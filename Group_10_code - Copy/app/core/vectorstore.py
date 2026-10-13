"""
upsc-sarathi-agentic :: app/core/vectorstore.py

Vector-store abstraction over Chroma (local/Ollama profile) and OpenSearch
Serverless (AWS/Bedrock profile). Provides a uniform `add(texts, metadatas)`
and `similarity_search(query, k, filter)` API used by both the ingestion
pipeline and the agent tools (app/agents/tools.py).

Metadata convention used throughout the project (set by app/ingestion/pipeline.py):
    source     - original filename
    page       - page number
    chunk_id   - stable id "<source>#p<page>#c<n>"
    subject    - heuristic subject tag (Polity, Economy, ...)
    doc_type   - "study_material" | "pyq" | "current_affairs" | "digest"
    exam_year  - (pyq only) int year if it could be parsed from filename
    exam_type  - (pyq only) "Prelims" | "Mains"
    ingested_on - ISO date the chunk was added (useful for current affairs freshness)
"""
from __future__ import annotations

import os
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from app.config import settings
from app.core.embeddings import get_embeddings, Embeddings


# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def short(self, n: int = 240) -> str:
        return self.text[:n].replace("\n", " ").strip() + ("…" if len(self.text) > n else "")


# ---------------------------------------------------------------------------
class BaseVectorStore:
    def add(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        raise NotImplementedError

    def similarity_search(
        self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def delete_by_source(self, source: str) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
class ChromaStore(BaseVectorStore):
    """Local, persistent Chroma store - the default for the Ollama/local profile."""

    def __init__(self, persist_dir: str | None = None, collection: Optional[str] = None) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.embeddings: Embeddings = get_embeddings()
        path = persist_dir or settings.chroma_persist_dir
        os.makedirs(path, exist_ok=True)
        # Suffix the collection by provider+dim so switching embedding providers
        # (e.g. ollama -> bedrock) never silently mixes incompatible vectors.
        coll_name = collection or f"upsc_sarathi_{settings.embeddings_provider}_{self.embeddings.dim}"
        self._client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=coll_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Chroma store ready at {path} (collection={coll_name})")

    def add(self, texts, metadatas, ids=None) -> List[str]:
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embs = self.embeddings.embed(texts)
        self._collection.add(ids=ids, embeddings=embs, documents=texts, metadatas=metadatas)
        return ids

    def similarity_search(self, query, k=5, filter=None) -> List[RetrievedChunk]:
        emb = self.embeddings.embed_one(query)
        kwargs: Dict[str, Any] = {"query_embeddings": [emb], "n_results": k}
        if filter:
            if len(filter) == 1:
                kwargs["where"] = filter
            else:
                kwargs["where"] = {"$and": [{k_: v_} for k_, v_ in filter.items()]}
        res = self._collection.query(**kwargs)
        out = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            out.append(RetrievedChunk(text=text, score=1.0 - float(dist), metadata=meta or {}))
        return out

    def count(self) -> int:
        return self._collection.count()

    def delete_by_source(self, source: str) -> int:
        before = self.count()
        self._collection.delete(where={"source": source})
        return before - self.count()


# ---------------------------------------------------------------------------
class OpenSearchStore(BaseVectorStore):
    """AWS OpenSearch Serverless (vector engine) store - the default for the AWS/Bedrock profile."""

    def __init__(self) -> None:
        import boto3
        from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

        if not settings.opensearch_endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT not set")

        self.embeddings = get_embeddings()
        self.dim = self.embeddings.dim
        self.index_name = settings.opensearch_index

        if settings.opensearch_username and settings.opensearch_password:
            auth: Any = (settings.opensearch_username, settings.opensearch_password)
        else:
            credentials = boto3.Session().get_credentials()
            auth = AWSV4SignerAuth(credentials, settings.aws_region, "aoss")

        self._client = OpenSearch(
            hosts=[{"host": settings.opensearch_endpoint.replace("https://", ""), "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60,
        )
        self._ensure_index()
        logger.info(f"OpenSearch store ready: endpoint={settings.opensearch_endpoint} index={self.index_name}")

    def _ensure_index(self) -> None:
        if self._client.indices.exists(self.index_name):
            return
        body = {
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 100}},
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": self.dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 256, "m": 24},
                        },
                    },
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "subject": {"type": "keyword"},
                    "doc_type": {"type": "keyword"},
                    "exam_year": {"type": "integer"},
                    "exam_type": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_id": {"type": "keyword"},
                    "ingested_on": {"type": "keyword"},
                }
            },
        }
        self._client.indices.create(self.index_name, body=body)
        logger.info(f"Created OpenSearch index {self.index_name}")

    def add(self, texts, metadatas, ids=None) -> List[str]:
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embs = self.embeddings.embed(texts)
        bulk_body: List[Dict[str, Any]] = []
        for _id, text, meta, vec in zip(ids, texts, metadatas, embs):
            bulk_body.append({"index": {"_index": self.index_name, "_id": _id}})
            bulk_body.append({"vector": vec, "text": text, **meta})
        self._client.bulk(body=bulk_body, refresh=True)
        return ids

    def similarity_search(self, query, k=5, filter=None) -> List[RetrievedChunk]:
        emb = self.embeddings.embed_one(query)
        knn_query: Dict[str, Any] = {"vector": emb, "k": k}
        if filter:
            knn_query["filter"] = {"bool": {"must": [{"term": {kk: vv}} for kk, vv in filter.items()]}}
        body = {"size": k, "query": {"knn": {"vector": knn_query}}}
        resp = self._client.search(index=self.index_name, body=body)
        out: List[RetrievedChunk] = []
        for hit in resp["hits"]["hits"]:
            src = dict(hit["_source"])
            text = src.pop("text", "")
            src.pop("vector", None)
            out.append(RetrievedChunk(text=text, score=float(hit["_score"]), metadata=src))
        return out

    def count(self) -> int:
        try:
            return int(self._client.count(index=self.index_name)["count"])
        except Exception as e:
            logger.warning(f"OpenSearch count() failed for index {self.index_name}: {e}")
            return -1

    def delete_by_source(self, source: str) -> int:
        body = {"query": {"term": {"source": source}}}
        r = self._client.delete_by_query(index=self.index_name, body=body, refresh=True)
        return int(r.get("deleted", 0))


# ---------------------------------------------------------------------------
_store_singleton: Optional[BaseVectorStore] = None


def get_vector_store() -> BaseVectorStore:
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    backend = settings.vector_store
    if backend == "chroma":
        _store_singleton = ChromaStore()
    elif backend == "opensearch":
        _store_singleton = OpenSearchStore()
    else:
        raise ValueError(f"Unsupported VECTOR_STORE: {backend}")
    return _store_singleton

"""
upsc-sarathi-agentic :: app/core/embeddings.py

Embeddings abstraction, independent of CrewAI (used directly by the vector
store and by ingestion). Four providers:
  - bedrock               : Amazon Titan Text Embeddings v2 (AWS profile)
  - ollama                : local embedding model served by Ollama (local profile, default)
  - sentence_transformers : local embedding model via HuggingFace (no GPU needed, alt. local option)
  - openai                : OpenAI embeddings (optional alt. profile)
"""
from __future__ import annotations

import json
from typing import List, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings


class Embeddings:
    def __init__(self) -> None:
        self.provider = settings.embeddings_provider
        if self.provider == "bedrock":
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
            self._model_id = settings.bedrock_embed_model
            self.dim = settings.embeddings_dim
            logger.info(f"Embeddings: Bedrock model={self._model_id}, dim={self.dim}")
        elif self.provider == "ollama":
            import httpx
            self._client = httpx.Client(base_url=settings.ollama_base_url, timeout=120.0)
            self._model_id = settings.ollama_embed_model
            # Probe once to discover the real embedding dimension instead of hardcoding it.
            self.dim = self._probe_ollama_dim()
            logger.info(f"Embeddings: Ollama model={self._model_id}, dim={self.dim}")
        elif self.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(settings.st_model_name)
            # sentence-transformers >=5 renamed this to get_embedding_dimension();
            # fall back to the old name so either major version works.
            _dim_fn = getattr(self._st, "get_embedding_dimension", None) or self._st.get_sentence_embedding_dimension
            self.dim = _dim_fn()
            logger.info(f"Embeddings: SentenceTransformers model={settings.st_model_name}, dim={self.dim}")
        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model_id = "text-embedding-3-small"
            self.dim = 1536
            logger.info("Embeddings: OpenAI text-embedding-3-small")
        else:
            raise ValueError(f"Unknown embeddings provider: {self.provider}")

    def _probe_ollama_dim(self) -> int:
        try:
            vec = self._ollama_one("dimension probe")
            return len(vec)
        except Exception as e:
            logger.warning(f"Could not probe Ollama embedding dim ({e}); defaulting to settings.embeddings_dim")
            return settings.embeddings_dim

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.provider == "bedrock":
            return [self._bedrock_one(t) for t in texts]
        if self.provider == "ollama":
            return [self._ollama_one(t) for t in texts]
        if self.provider == "sentence_transformers":
            return self._st.encode(texts, normalize_embeddings=True).tolist()
        if self.provider == "openai":
            resp = self._client.embeddings.create(model=self._model_id, input=texts)
            return [d.embedding for d in resp.data]
        raise RuntimeError("Embeddings provider not configured")

    # ---------- Bedrock ----------
    def _bedrock_one(self, text: str) -> List[float]:
        body = {"inputText": text[:8000], "dimensions": self.dim, "normalize": True}
        resp = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(body),
            contentType="application/json",
        )
        return json.loads(resp["body"].read())["embedding"]

    # ---------- Ollama ----------
    def _ollama_one(self, text: str) -> List[float]:
        r = self._client.post("/api/embeddings", json={"model": self._model_id, "prompt": text[:8000]})
        r.raise_for_status()
        return r.json()["embedding"]


_emb_singleton: Optional[Embeddings] = None


def get_embeddings() -> Embeddings:
    global _emb_singleton
    if _emb_singleton is None:
        _emb_singleton = Embeddings()
    return _emb_singleton

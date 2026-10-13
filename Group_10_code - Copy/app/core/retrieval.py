"""
upsc-sarathi-agentic :: app/core/retrieval.py

Pure retrieval helper: similarity search -> MMR rerank -> formatted context
string. This used to be half of the old "RAGPipeline" (the other half,
prompting an LLM directly, is now the job of CrewAI agents - see
app/agents/tools.py, which wraps this class as agent-callable tools).

Keeping retrieval as a plain, fast, deterministic Python class (no LLM calls)
means agents that need facts can call it as a tool without an extra
LLM-generation hop.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from app.core.vectorstore import get_vector_store, RetrievedChunk
from app.core.embeddings import get_embeddings


def _mmr_rerank(
    query_vec: List[float],
    candidates: List[RetrievedChunk],
    k: int = 6,
    lambda_mult: float = 0.6,
) -> List[RetrievedChunk]:
    """Maximal Marginal Relevance: pick chunks that are relevant AND diverse,
    so the agent isn't fed k near-duplicate paragraphs."""
    if not candidates:
        return []
    emb = get_embeddings()
    cand_vecs = emb.embed([c.text for c in candidates])
    cand_arr = np.array(cand_vecs)
    q_arr = np.array(query_vec)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    selected: List[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < k:
        best, best_score = None, -1e9
        for idx in remaining:
            rel = cos(q_arr, cand_arr[idx])
            div = max((cos(cand_arr[idx], cand_arr[s]) for s in selected), default=0)
            score = lambda_mult * rel - (1 - lambda_mult) * div
            if score > best_score:
                best, best_score = idx, score
        selected.append(best)  # type: ignore
        remaining.remove(best)  # type: ignore
    return [candidates[i] for i in selected]


class Retriever:
    def __init__(self) -> None:
        self.vs = get_vector_store()
        self.emb = get_embeddings()

    def retrieve(
        self,
        query: str,
        k: int = 6,
        fetch_k: int = 20,
        filter: Optional[Dict[str, Any]] = None,
        use_mmr: bool = True,
    ) -> List[RetrievedChunk]:
        candidates = self.vs.similarity_search(query, k=fetch_k, filter=filter)
        if not candidates:
            return []
        if use_mmr and len(candidates) > k:
            qvec = self.emb.embed_one(query)
            return _mmr_rerank(qvec, candidates, k=k)
        return candidates[:k]

    @staticmethod
    def build_context(chunks: List[RetrievedChunk], max_chars: int = 9000) -> str:
        if not chunks:
            return "(no relevant material found in the ingested vector database)"
        parts, total = [], 0
        for i, c in enumerate(chunks, 1):
            src = c.metadata.get("source", "?")
            page = c.metadata.get("page", "?")
            block = f"[Chunk {i} | {src} p.{page}]\n{c.text}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts) if parts else "(no relevant material found)"


_retriever_singleton: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever()
    return _retriever_singleton

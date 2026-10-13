"""
upsc-sarathi-agentic :: app/agents/llm_factory.py

Every agent in this project asks for an LLM by *tier*, not by provider:

    llm = get_llm("fast")   # routing / PYQ matching / tagging
    llm = get_llm("smart")  # writing / planning / evaluating / formatting
    llm = get_llm("long")   # the essay writer (needs a bigger output budget)

`get_llm()` then resolves the actual `crewai.LLM` instance from
`settings.llm_provider` (ollama | bedrock | openai | groq), so swapping the
whole project between profiles is a single .env change (LLM_PROVIDER=...)
and nothing in app/agents/*.py ever needs to know which provider is active.

Groq profile (LLM_PROVIDER=groq):
  Groq (groq.com) is a free-tier cloud inference API — no Ollama install,
  no AWS account, no GPU required. It exposes an OpenAI-compatible REST API
  at https://api.groq.com/openai/v1. Fast tier uses openai/gpt-oss-20b;
  smart/long tiers use openai/gpt-oss-120b (Groq decommissioned
  llama-3.1-8b-instant and llama-3.3-70b-versatile on 2026-08-16, which is
  what caused `model_not_found` 404s). Get a free key at https://console.groq.com.

  Embeddings: Groq does not provide an embeddings endpoint. The project
  falls back to sentence_transformers (BAAI/bge-base-en-v1.5), which runs
  fully locally on CPU — no GPU or internet connection after first download.

  Rate limit note: Groq's free tier allows 30 RPM across all models
  combined. AGENT_MAX_RPM is set to 25 in .env.groq to stay safely below
  this ceiling and avoid 429 errors mid-agent-chain.

Verified against crewai's OpenAI-compatible LLM configuration: Groq is
configured as an OpenAI-compatible endpoint using provider="openai", the
Groq base URL, and the Groq API key. This avoids using an unrelated provider
name as an internal routing token.
"""
from __future__ import annotations

from typing import Literal, Dict, Tuple
from functools import lru_cache

from crewai import LLM

from app.config import settings

Tier = Literal["fast", "smart", "long"]


def _max_tokens(tier: Tier) -> int:
    return {
        "fast": settings.llm_max_tokens_fast,
        "smart": settings.llm_max_tokens_smart,
        "long": settings.llm_max_tokens_long,
    }[tier]


def _model_name(tier: Tier) -> str:
    """`long` re-uses the same underlying model as `smart` - the essay
    writer needs more *output tokens*, not a different/smarter model."""
    provider = settings.llm_provider
    is_fast = tier == "fast"
    if provider == "bedrock":
        base = settings.bedrock_llm_model_fast if is_fast else settings.bedrock_llm_model
        return f"bedrock/{base}"
    if provider == "openai":
        return settings.openai_model_fast if is_fast else settings.openai_model
    if provider == "groq":
        # Bare model name; _build_llm selects the OpenAI-compatible provider
        # and points it at Groq's API endpoint.
        return settings.groq_model_fast if is_fast else settings.groq_model
    # ollama
    base = settings.ollama_model_fast if is_fast else settings.ollama_model
    return f"ollama/{base}"


def _build_llm(tier: Tier) -> LLM:
    provider = settings.llm_provider
    model = _model_name(tier)
    max_tokens = _max_tokens(tier)

    if provider == "bedrock":
        kwargs: Dict = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3 if tier != "fast" else 0.1,
            region_name=settings.aws_region,
        )
        # Only pass explicit keys if provided - otherwise let crewai/boto3
        # fall back to the default credential chain (IAM role on ECS, or
        # `aws configure` / env vars locally). This is the same pattern the
        # reference (non-agentic) project used in app/core/llm.py.
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        return LLM(**kwargs)

    if provider == "openai":
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3 if tier != "fast" else 0.1,
        )
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        return LLM(**kwargs)

    if provider == "groq":
        # Groq exposes an OpenAI-compatible API. Use the OpenAI provider
        # adapter explicitly and override its endpoint/key with Groq values.
        # Note: this fixes provider routing only. CrewAI structured outputs
        # (output_pydantic/response_format) must still not be combined with
        # tool-calling in the same Groq request; that conflict is resolved in
        # the crew/task definitions, which were not included in the supplied ZIP.
        if not settings.groq_api_key:
            raise ValueError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and add it to .env"
            )
        return LLM(
            model=model,
            provider="openai",
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,  # https://api.groq.com/openai/v1
            max_tokens=max_tokens,
            temperature=0.3 if tier != "fast" else 0.1,
            # gpt-oss is a reasoning model; crewai only forwards
            # `reasoning_effort` for OpenAI o-series, so send it via extra_body.
            extra_body={"reasoning_effort": settings.groq_reasoning_effort},
        )

    # ollama
    return LLM(
        model=model,
        max_tokens=max_tokens,
        temperature=0.3 if tier != "fast" else 0.1,
        base_url=f"{settings.ollama_base_url}/v1",
    )


@lru_cache(maxsize=8)
def _cached(tier: Tier, provider: str, model: str) -> LLM:
    # provider/model included in the cache key purely so that if settings()
    # is ever re-read mid-process (e.g. in tests that monkeypatch env vars)
    # we don't silently hand back a stale LLM bound to the old provider.
    return _build_llm(tier)


def get_llm(tier: Tier = "smart") -> LLM:
    return _cached(tier, settings.llm_provider, _model_name(tier))

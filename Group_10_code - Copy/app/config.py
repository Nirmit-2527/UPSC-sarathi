"""
upsc-sarathi-agentic :: app/config.py

Centralised settings, loaded from .env using pydantic-settings.

This single settings object drives BOTH deployment profiles described in the
project docs:
  - "local"  -> Ollama (LLM + embeddings) + persistent Chroma vector store
  - "aws"    -> Bedrock (LLM + embeddings) + OpenSearch Serverless vector store

Nothing in app/ branches on `if local: ... else: ...` directly - everything
reads from `settings`, so the same codebase runs in both environments purely
through environment variables. See .env.example, SETUP_LOCAL.md and
DEPLOY_AWS.md.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# crewai asks "Would you like to view your execution traces? [y/N] (20s timeout)"
# on stdin after a failed crew run, which freezes the API request for 20s and
# tries to reach telemetry.crewai.com. Turn that off unless the user opted in.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Deployment ---
    app_mode: Literal["local", "aws"] = "local"
    app_env: str = "development"
    log_level: str = "INFO"

    # --- AWS ---
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # --- S3 ---
    s3_bucket_name: str = "upsc-sarathi-docs"
    s3_raw_prefix: str = "raw/"
    s3_processed_prefix: str = "processed/"

    # --- Vector store ---
    vector_store: Literal["chroma", "opensearch"] = "chroma"
    chroma_persist_dir: str = "./data/chroma"
    opensearch_endpoint: str | None = None
    opensearch_index: str = "upsc-sarathi-index"
    opensearch_username: str | None = None
    opensearch_password: str | None = None

    # --- LLM provider (used to build crewai.LLM instances for every agent) ---
    llm_provider: Literal["bedrock", "openai", "ollama", "groq"] = "ollama"

    # "smart" tier - used by agents that write/plan/evaluate/format (quality matters most)
    bedrock_llm_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    openai_model: str = "gpt-4o"
    ollama_model: str = "llama3.1:8b"
    groq_model: str = "openai/gpt-oss-120b"            # smart / long tier (free)

    # "fast" tier - used by lightweight agents (routing, PYQ matching, tagging)
    bedrock_llm_model_fast: str = "anthropic.claude-3-5-haiku-20241022-v1:0"
    openai_model_fast: str = "gpt-4o-mini"
    ollama_model_fast: str = "llama3.2:3b"
    groq_model_fast: str = "openai/gpt-oss-20b"       # fast tier (free)

    # token budgets per tier (max output tokens)
    llm_max_tokens_fast: int = 1024
    llm_max_tokens_smart: int = 4096
    llm_max_tokens_long: int = 6500  # essays

    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str | None = None

    # --- Groq (groq.com) free inference API ---
    # Get a free key at https://console.groq.com  (no credit card required)
    # OpenAI-compatible: base URL https://api.groq.com/openai/v1
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # gpt-oss models are reasoning models: hidden reasoning tokens count against
    # max_tokens, so keep effort low (low|medium|high) to leave room for the answer.
    groq_reasoning_effort: str = "low"

    # --- Embeddings ---
    embeddings_provider: Literal["bedrock", "ollama", "sentence_transformers", "openai"] = "ollama"
    embeddings_dim: int = 1024
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"
    ollama_embed_model: str = "nomic-embed-text"
    st_model_name: str = "BAAI/bge-base-en-v1.5"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- PDF ---
    use_textract: bool = False
    pdf_loader: Literal["pypdf", "pdfplumber", "unstructured"] = "pdfplumber"

    # --- Agents (CrewAI) ---
    agents_verbose: bool = False
    agent_max_rpm: int = 30
    guardrail_max_retries: int = 4
    enable_pyq_matching_in_qa: bool = True
    enable_current_affairs_digest: bool = True
    enable_groundedness_check: bool = True

    # --- UPSC answer-writing standards (used by guardrails) ---
    mains_answer_default_marks: int = 15
    essay_target_words: int = 1500
    essay_word_tolerance: int = 150

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    allowed_origins: str = "*"

    # --- DynamoDB (optional - user progress tracking on AWS) ---
    enable_progress_tracking: bool = False
    ddb_table_progress: str = "upsc-sarathi-progress"

    @property
    def cors_origins(self) -> List[str]:
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

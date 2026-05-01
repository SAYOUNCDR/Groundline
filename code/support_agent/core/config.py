from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    code_dir: Path
    data_dir: Path
    support_tickets_dir: Path
    cache_dir: Path
    vector_backend: str
    qdrant_url: str
    qdrant_collection: str
    embedding_model: str
    llm_provider: str
    groq_api_key: str
    groq_model: str
    gemini_api_key: str
    gemini_model: str
    dmr_base_url: str
    dmr_model: str

    @classmethod
    def load(cls) -> "Settings":
        root = repo_root()
        load_dotenv(root / ".env")
        return cls(
            root_dir=root,
            code_dir=root / "code",
            data_dir=root / "data",
            support_tickets_dir=root / "support_tickets",
            cache_dir=root / "code" / ".cache",
            vector_backend=os.getenv("VECTOR_BACKEND", "qdrant"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "support_corpus"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en"),
            llm_provider=os.getenv("LLM_PROVIDER", "auto"),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            dmr_base_url=os.getenv("DMR_BASE_URL", "http://localhost:12434/engines/v1"),
            dmr_model=os.getenv("DMR_MODEL", "gemma4:4B-Q4_K_XL"),
        )

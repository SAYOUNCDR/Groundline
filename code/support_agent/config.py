from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    code_dir: Path
    data_dir: Path
    support_tickets_dir: Path
    cache_dir: Path

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
        )

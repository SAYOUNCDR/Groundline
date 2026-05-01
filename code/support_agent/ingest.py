from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import Evidence
from .text_utils import normalize_space


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    company: str
    product_area: str
    title: str
    heading: str
    source_path: Path
    text: str

    def as_evidence(self, score: float = 0.0, method: str = "bm25") -> Evidence:
        return Evidence(
            chunk_id=self.chunk_id,
            company=self.company,
            product_area=self.product_area,
            title=self.title,
            heading=self.heading,
            source_path=self.source_path,
            text=self.text,
            score=score,
            method=method,
        )


def load_corpus(data_dir: Path) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    for path in sorted(data_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        company = infer_company(path, data_dir)
        product_area = infer_product_area(path, data_dir, company)
        title = infer_title(path, text)
        for index, section in enumerate(split_markdown(text)):
            clean = normalize_space(section["text"])
            if len(clean) < 40:
                continue
            digest = hashlib.sha1(f"{path}:{index}:{clean[:80]}".encode()).hexdigest()[:12]
            chunks.append(
                CorpusChunk(
                    chunk_id=digest,
                    company=company,
                    product_area=product_area,
                    title=title,
                    heading=section["heading"] or title,
                    source_path=path,
                    text=clean,
                )
            )
    return chunks


def split_markdown(text: str, max_chars: int = 1400) -> list[dict[str, str]]:
    matches = list(HEADING_RE.finditer(text or ""))
    if not matches:
        return split_long_text(text, "", max_chars)

    sections: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        body = text[start:end].strip()
        sections.extend(split_long_text(body, heading, max_chars))
    return sections


def split_long_text(text: str, heading: str, max_chars: int) -> list[dict[str, str]]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    if not paragraphs:
        return []

    chunks: list[dict[str, str]] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) > max_chars:
            chunks.append({"heading": heading, "text": "\n\n".join(current)})
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph)
    if current:
        chunks.append({"heading": heading, "text": "\n\n".join(current)})
    return chunks


def infer_company(path: Path, data_dir: Path) -> str:
    try:
        top = path.relative_to(data_dir).parts[0].lower()
    except ValueError:
        return "unknown"
    if top == "hackerrank":
        return "HackerRank"
    if top == "claude":
        return "Claude"
    if top == "visa":
        return "Visa"
    return top.title()


def infer_product_area(path: Path, data_dir: Path, company: str) -> str:
    parts = [part.lower() for part in path.relative_to(data_dir).parts]
    joined = "/".join(parts)

    if company == "HackerRank":
        if "hackerrank_community" in parts:
            return "community"
        if "screen" in parts:
            return "screen"
        if "interviews" in parts:
            return "interviews"
        if "settings" in parts:
            return "settings"
        if "skillup" in parts:
            return "skillup"
        if "integrations" in parts:
            return "integrations"
        if len(parts) > 1:
            return parts[1].replace("-", "_")

    if company == "Claude":
        if "privacy-and-legal" in parts or "privacy" in joined:
            return "privacy"
        if "conversation-management" in parts:
            return "conversation_management"
        if "amazon-bedrock" in parts:
            return "amazon_bedrock"
        if "claude-for-education" in parts:
            return "education"
        if "team-and-enterprise-plans" in parts:
            return "team_enterprise"
        if "identity-management-sso-jit-scim" in parts:
            return "identity_management"
        if "safeguards" in parts:
            return "safeguards"
        if len(parts) > 1:
            return parts[1].replace("-", "_")

    if company == "Visa":
        if "travel-support" in parts or "travelers-cheques" in joined:
            return "travel_support"
        if "dispute-resolution" in parts:
            return "dispute_resolution"
        if "fraud-protection" in parts or "data-security" in parts:
            return "fraud_protection"
        if "visa-rules" in parts or "regulations-fees" in parts:
            return "general_support"
        return "general_support"

    return ""


def infer_title(path: Path, text: str) -> str:
    match = HEADING_RE.search(text or "")
    if match:
        return match.group(2).strip()
    return path.stem.replace("-", " ").replace("_", " ").title()

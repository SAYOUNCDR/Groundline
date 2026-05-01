from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Status(StrEnum):
    REPLIED = "replied"
    ESCALATED = "escalated"


class RequestType(StrEnum):
    PRODUCT_ISSUE = "product_issue"
    FEATURE_REQUEST = "feature_request"
    BUG = "bug"
    INVALID = "invalid"


class Ticket(BaseModel):
    issue: str = ""
    subject: str = ""
    company: str = "None"
    row_id: int | None = None

    @field_validator("issue", "subject", "company", mode="before")
    @classmethod
    def coerce_text(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)

    @property
    def query_text(self) -> str:
        return f"{self.subject}\n{self.issue}".strip()


class Evidence(BaseModel):
    chunk_id: str
    company: str
    product_area: str
    title: str
    heading: str
    source_path: Path
    text: str
    score: float = 0.0
    method: str = "bm25"

    @property
    def citation(self) -> str:
        label = self.heading or self.title or self.source_path.name
        return f"{self.source_path} :: {label}"


class Decision(BaseModel):
    status: Status
    request_type: RequestType
    product_area: str = ""
    reason: str
    risk_flags: list[str] = Field(default_factory=list)


class Prediction(BaseModel):
    status: Status
    product_area: str = ""
    response: str
    justification: str
    request_type: RequestType

    def to_csv_row(self) -> dict[str, str]:
        return {
            "status": self.status.value,
            "product_area": self.product_area,
            "response": self.response,
            "justification": self.justification,
            "request_type": self.request_type.value,
        }

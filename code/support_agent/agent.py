from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .generator import generate_prediction
from .policies import decide, normalize_company
from .retriever import CitationStore, HybridRetriever
from .schemas import Evidence, Prediction, RequestType, Status, Ticket
from .verifier import verify_prediction


class SupportAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.retriever = HybridRetriever.from_settings(self.settings)
        self.citations = CitationStore()

    def build_index(self, recreate: bool = False) -> int:
        return self.retriever.build_index(recreate=recreate)

    def answer(self, ticket: Ticket) -> Prediction:
        decision = decide(ticket)
        company = normalize_company(ticket.company, ticket.query_text)
        evidence: list[Evidence] = []

        if decision.request_type != RequestType.INVALID:
            evidence = self.retriever.search(
                query=ticket.query_text,
                company=company,
                product_area=decision.product_area,
                top_k=6,
            )
            self.citations.add(ticket.row_id, evidence)

            if decision.status == Status.REPLIED and not has_enough_evidence(evidence):
                decision.status = Status.ESCALATED
                decision.reason = "Human review required because retrieved support evidence was weak."
                decision.risk_flags.append("weak_evidence")

        prediction = generate_prediction(ticket, decision, evidence)
        return verify_prediction(prediction)

    def run_csv(
        self,
        input_path: Path,
        output_path: Path,
        debug_jsonl_path: Path | None = None,
    ) -> list[Prediction]:
        tickets = load_tickets(input_path)
        predictions: list[Prediction] = []
        debug_records: list[dict[str, Any]] = []

        for ticket in tickets:
            prediction = self.answer(ticket)
            predictions.append(prediction)
            debug_records.append(debug_record(ticket, prediction, self.citations.get(ticket.row_id or -1)))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([prediction.to_csv_row() for prediction in predictions]).to_csv(output_path, index=False)

        if debug_jsonl_path:
            debug_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
                for record in debug_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        return predictions


def load_tickets(input_path: Path) -> list[Ticket]:
    frame = pd.read_csv(input_path).fillna("")
    tickets: list[Ticket] = []
    for idx, row in frame.iterrows():
        tickets.append(
            Ticket(
                row_id=int(idx),
                issue=get_column(row, "issue"),
                subject=get_column(row, "subject"),
                company=get_column(row, "company"),
            )
        )
    return tickets


def get_column(row: pd.Series, name: str) -> str:
    for key in row.index:
        if key.strip().lower() == name:
            return str(row[key])
    return ""


def has_enough_evidence(evidence: list[Evidence]) -> bool:
    if not evidence:
        return False
    # Hybrid RRF scores are compact, while BM25-only fallback scores can be large.
    if "+" in evidence[0].method or evidence[0].method == "qdrant":
        return evidence[0].score >= 0.015
    return evidence[0].score >= 2.0


def debug_record(ticket: Ticket, prediction: Prediction, evidence: list[Evidence]) -> dict[str, Any]:
    return {
        "row_id": ticket.row_id,
        "ticket": {
            "company": ticket.company,
            "subject": ticket.subject,
            "issue": ticket.issue,
        },
        "prediction": prediction.to_csv_row(),
        "citations": [
            {
                "source_path": str(item.source_path),
                "title": item.title,
                "heading": item.heading,
                "company": item.company,
                "product_area": item.product_area,
                "score": item.score,
                "method": item.method,
                "excerpt": item.text[:500],
            }
            for item in evidence
        ],
    }

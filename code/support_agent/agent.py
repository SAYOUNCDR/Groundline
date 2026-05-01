from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from support_agent.core.config import Settings
from support_agent.core.schemas import Evidence, Prediction, RequestType, Status, Ticket
from support_agent.decision.policies import decide, normalize_company
from support_agent.generation.generator import GroundedResponseGenerator
from support_agent.intelligence.classifier import TicketClassifier
from support_agent.intelligence.evidence import EvidenceGrader
from support_agent.intelligence.llm import LLMRouter
from support_agent.intelligence.reranker import EvidenceReranker
from support_agent.quality.verifier import OutputVerifier
from support_agent.retrieval.hybrid import CitationStore, HybridRetriever


class SupportAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.llm = LLMRouter(self.settings)
        self.classifier = TicketClassifier(self.llm)
        self.retriever = HybridRetriever.from_settings(self.settings)
        self.reranker = EvidenceReranker(EvidenceGrader(self.llm))
        self.generator = GroundedResponseGenerator(self.llm, use_llm=self.settings.use_llm_generation)
        self.verifier = OutputVerifier()
        self.citations = CitationStore()

    def build_index(self, recreate: bool = False) -> int:
        return self.retriever.build_index(recreate=recreate)

    def answer(self, ticket: Ticket) -> Prediction:
        ai_classification = self.classifier.classify(ticket)
        decision = decide(ticket, ai_classification=ai_classification)
        company = normalize_company(ticket.company, ticket.query_text)
        if ai_classification.confidence >= 0.55 and ai_classification.company != "None":
            company = ai_classification.company
        evidence: list[Evidence] = []

        if decision.request_type != RequestType.INVALID:
            evidence = self.retriever.search(
                query=ticket.query_text,
                company=company,
                product_area=decision.product_area,
                top_k=6,
            )
            evidence = self.reranker.rerank(ticket, evidence, top_k=6)
            self.citations.add(ticket.row_id, evidence)

            if decision.status == Status.REPLIED and not has_enough_evidence(evidence):
                decision.status = Status.ESCALATED
                decision.reason = "Human review required because retrieved support evidence was weak."
                decision.risk_flags.append("weak_evidence")

        prediction = self.generator.generate(ticket, decision, evidence)
        return self.verifier.verify(prediction, evidence=evidence)

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
    top = evidence[0]
    if top.support == "strong" and top.relevance >= 0.22:
        return True
    if top.support == "partial" and top.relevance >= 0.34:
        return True
    if top.support == "unknown":
        # Hybrid RRF scores are compact, while BM25-only fallback scores can be large.
        if "+" in top.method or top.method == "qdrant":
            return top.score >= 0.015
        return top.score >= 2.0
    return False


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
                "relevance": item.relevance,
                "support": item.support,
                "support_reason": item.support_reason,
                "excerpt": item.text[:500],
            }
            for item in evidence
        ],
    }

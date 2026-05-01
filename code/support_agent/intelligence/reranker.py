from __future__ import annotations

from support_agent.core.schemas import Evidence, Ticket
from support_agent.intelligence.evidence import EvidenceGrader


class EvidenceReranker:
    def __init__(self, grader: EvidenceGrader) -> None:
        self.grader = grader

    def rerank(self, ticket: Ticket, evidence: list[Evidence], top_k: int = 6) -> list[Evidence]:
        graded = self.grader.grade_all(ticket, evidence)
        for item in graded:
            item.score = combined_score(item)
        return sorted(graded, key=lambda item: item.score, reverse=True)[:top_k]


def combined_score(evidence: Evidence) -> float:
    method_bonus = 0.08 if "+" in evidence.method else 0.0
    support_bonus = {"strong": 0.2, "partial": 0.08, "weak": -0.12}.get(evidence.support, 0.0)
    return max(0.0, evidence.score + evidence.relevance + method_bonus + support_bonus)

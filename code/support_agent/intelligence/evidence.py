from __future__ import annotations

import json

from pydantic import ValidationError

from support_agent.core.schemas import Evidence, EvidenceAssessment, Ticket
from support_agent.core.text import tokens
from support_agent.intelligence.llm import LLMRouter


GRADER_SYSTEM = """You grade whether a retrieved support-document excerpt answers a support ticket.
Return strict JSON: relevance (0-1), support (strong|partial|weak), reason, should_answer.
Grade only the evidence shown. Do not use outside knowledge."""


class EvidenceGrader:
    def __init__(self, llm: LLMRouter | None = None) -> None:
        self.llm = llm

    def grade(self, ticket: Ticket, evidence: Evidence) -> EvidenceAssessment:
        if self.llm is not None:
            payload, _provider = self.llm.complete_json(GRADER_SYSTEM, grader_prompt(ticket, evidence))
            if payload:
                try:
                    return normalize_assessment(EvidenceAssessment.model_validate(payload))
                except ValidationError:
                    pass
        return heuristic_grade(ticket, evidence)

    def grade_all(self, ticket: Ticket, evidence_list: list[Evidence]) -> list[Evidence]:
        graded: list[Evidence] = []
        for evidence in evidence_list:
            assessment = self.grade(ticket, evidence)
            evidence.relevance = assessment.relevance
            evidence.support = assessment.support
            evidence.support_reason = assessment.reason
            graded.append(evidence)
        return graded


def grader_prompt(ticket: Ticket, evidence: Evidence) -> str:
    return json.dumps(
        {
            "ticket": {
                "company": ticket.company,
                "subject": ticket.subject,
                "issue": ticket.issue,
            },
            "evidence": {
                "company": evidence.company,
                "product_area": evidence.product_area,
                "title": evidence.title,
                "heading": evidence.heading,
                "text": evidence.text[:1800],
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def heuristic_grade(ticket: Ticket, evidence: Evidence) -> EvidenceAssessment:
    query_tokens = set(tokens(ticket.query_text))
    evidence_tokens = set(tokens(f"{evidence.title} {evidence.heading} {evidence.text}"))
    if not query_tokens or not evidence_tokens:
        return EvidenceAssessment(relevance=0.0, support="weak", reason="No comparable text tokens.", should_answer=False)

    overlap = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
    title_overlap = len(query_tokens & set(tokens(evidence.title + " " + evidence.heading))) / max(len(query_tokens), 1)
    score_signal = min(max(evidence.score, 0.0) * 3, 0.25) if evidence.score < 1 else 0.2
    relevance = min(1.0, overlap * 0.65 + title_overlap * 0.45 + score_signal)

    if relevance >= 0.32:
        support = "strong"
    elif relevance >= 0.18:
        support = "partial"
    else:
        support = "weak"

    return EvidenceAssessment(
        relevance=round(relevance, 3),
        support=support,
        reason=f"Token/title overlap estimate: {relevance:.2f}.",
        should_answer=support in {"strong", "partial"},
    )


def normalize_assessment(assessment: EvidenceAssessment) -> EvidenceAssessment:
    assessment.relevance = max(0.0, min(1.0, assessment.relevance))
    if assessment.support not in {"strong", "partial", "weak"}:
        assessment.support = "weak"
    assessment.should_answer = assessment.support in {"strong", "partial"} and assessment.relevance >= 0.18
    return assessment

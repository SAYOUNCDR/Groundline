from __future__ import annotations

import json

from pydantic import ValidationError

from support_agent.core.schemas import AIClassification, RequestType, Status, Ticket
from support_agent.core.text import contains_any
from support_agent.decision.policies import infer_product_area, normalize_company, risk_flags
from support_agent.intelligence.llm import LLMRouter


CLASSIFIER_SYSTEM = """You classify support tickets for a grounded support triage agent.
Use only the ticket text and company hint. Return strict JSON with:
company, request_type, product_area, status, risk_flags, sentiment, urgency, confidence, reasoning.
Allowed request_type: product_issue, feature_request, bug, invalid.
Allowed status: replied, escalated.
Escalate sensitive, risky, unsupported, admin-only, money, fraud, security, outage, or hidden-policy requests.
Do not answer the user; only classify."""


class TicketClassifier:
    def __init__(self, llm: LLMRouter | None = None) -> None:
        self.llm = llm

    def classify(self, ticket: Ticket) -> AIClassification:
        if self.llm is not None:
            payload, _provider = self.llm.complete_json(CLASSIFIER_SYSTEM, classifier_prompt(ticket))
            if payload:
                try:
                    return AIClassification.model_validate(payload)
                except ValidationError:
                    pass
        return heuristic_classification(ticket)


def classifier_prompt(ticket: Ticket) -> str:
    return json.dumps(
        {
            "company_hint": ticket.company,
            "subject": ticket.subject,
            "issue": ticket.issue,
        },
        ensure_ascii=False,
        indent=2,
    )


def heuristic_classification(ticket: Ticket) -> AIClassification:
    text = ticket.query_text
    lowered = text.lower()
    company = normalize_company(ticket.company, text)
    flags = risk_flags(text)
    request_type = RequestType.PRODUCT_ISSUE
    status = Status.ESCALATED if flags else Status.REPLIED

    if contains_any(lowered, ["delete all files", "actor in iron man", "thank you"]):
        request_type = RequestType.INVALID
        status = Status.REPLIED
    elif contains_any(lowered, ["feature request", "can you add", "please add"]):
        request_type = RequestType.FEATURE_REQUEST
    elif contains_any(lowered, ["not working", "down", "failing", "error", "blocker", "unable"]):
        request_type = RequestType.BUG

    sentiment = "frustrated" if contains_any(lowered, ["urgent", "asap", "immediately", "blocked", "frustrated"]) else "neutral"
    urgency = "high" if contains_any(lowered, ["urgent", "asap", "immediately", "unable", "blocked"]) else "normal"

    return AIClassification(
        company=company,
        request_type=request_type,
        product_area=infer_product_area(company, text),
        status=status,
        risk_flags=flags,
        sentiment=sentiment,
        urgency=urgency,
        confidence=0.72,
        reasoning="Heuristic classifier fallback based on broad support-risk and intent patterns.",
    )

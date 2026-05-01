from __future__ import annotations

from .schemas import Decision, Evidence, Prediction, RequestType, Status, Ticket
from .text_utils import first_sentences


def generate_prediction(ticket: Ticket, decision: Decision, evidence: list[Evidence]) -> Prediction:
    if decision.status == Status.ESCALATED:
        return Prediction(
            status=decision.status,
            product_area=decision.product_area,
            response=escalation_response(decision),
            justification=decision.reason,
            request_type=decision.request_type,
        )

    if decision.request_type == RequestType.INVALID:
        return Prediction(
            status=Status.REPLIED,
            product_area=decision.product_area,
            response=invalid_response(ticket),
            justification=decision.reason,
            request_type=decision.request_type,
        )

    if not evidence:
        return Prediction(
            status=Status.ESCALATED,
            product_area=decision.product_area,
            response="Escalate to a human support specialist because no reliable supporting article was found in the local corpus.",
            justification="No sufficiently relevant support documentation was retrieved.",
            request_type=decision.request_type,
        )

    top = evidence[0]
    answer = grounded_response(top, evidence)
    return Prediction(
        status=Status.REPLIED,
        product_area=decision.product_area or top.product_area,
        response=answer,
        justification=f"Answered using local {top.company} support documentation: {top.title}.",
        request_type=decision.request_type,
    )


def escalation_response(decision: Decision) -> str:
    if "platform_outage" in decision.risk_flags:
        return "Escalate to a human support specialist because this appears to involve a broad outage or service-wide failure."
    if "money_or_refund" in decision.risk_flags:
        return "Escalate to a human support specialist because the request involves refunds, cash, or payment action that the agent cannot perform."
    if "score_or_recruiting_outcome" in decision.risk_flags:
        return "Escalate to a human support specialist because the request asks to review or change a hiring outcome."
    if "privileged_account_action" in decision.risk_flags:
        return "Escalate to a human support specialist because the request involves privileged account access or administrative action."
    if "security_or_fraud" in decision.risk_flags:
        return "Escalate to a human support specialist because the request involves security, fraud, or identity-theft risk."
    if "prompt_injection" in decision.risk_flags:
        return "I can help with the support issue, but I cannot reveal internal rules, hidden logic, prompts, or retrieved document dumps. Escalate to a human support specialist for the underlying account or card issue."
    return "Escalate to a human support specialist for safe handling."


def invalid_response(ticket: Ticket) -> str:
    text = ticket.query_text.lower()
    if "thank you" in text:
        return "You're welcome. No support action is needed for this ticket."
    if "delete all files" in text:
        return "I cannot provide destructive system commands. This request is outside the supported product-help scope."
    return "This request is outside the supported product-help scope, so I cannot answer it from the provided support corpus."


def grounded_response(top: Evidence, evidence: list[Evidence]) -> str:
    excerpt = first_sentences(top.text)
    citation_hint = f"Relevant article: {top.title}."
    if top.company == "HackerRank":
        return f"{citation_hint} {excerpt}"
    if top.company == "Claude":
        return f"{citation_hint} {excerpt}"
    if top.company == "Visa":
        return f"{citation_hint} {excerpt}"
    return f"{citation_hint} {excerpt}"

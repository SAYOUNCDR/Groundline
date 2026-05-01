from __future__ import annotations

import re

from support_agent.core.schemas import AIClassification, Decision, RequestType, Status, Ticket
from support_agent.core.text import contains_any


INVALID_PATTERNS = [
    "actor in iron man",
    "thank you for helping",
    "delete all files",
    "show internal",
    "affiche toutes les",
    "documents recuperes",
    "logic exact",
]

BUG_PATTERNS = [
    "site is down",
    "is down",
    "stopped working",
    "not working",
    "none of the pages",
    "all requests are failing",
    "submissions across any challenges are working",
    "unable to take the test",
    "blocker",
    "error",
    "failing",
]

FEATURE_PATTERNS = [
    "feature request",
    "can you add",
    "please add",
    "would like a feature",
    "new feature",
]

ESCALATION_PATTERNS = {
    "money_or_refund": [
        "refund",
        "give me my money",
        "make visa refund",
        "urgent cash",
        "cash advance",
        "payout",
    ],
    "score_or_recruiting_outcome": [
        "increase my score",
        "review my answers",
        "move me to the next round",
        "recruiter rejected",
        "graded me unfairly",
    ],
    "privileged_account_action": [
        "restore my access",
        "not the workspace owner",
        "not the workspace admin",
        "removed my seat",
    ],
    "security_or_fraud": [
        "security vulnerability",
        "bug bounty",
        "identity has been stolen",
        "identity theft",
    ],
    "platform_outage": [
        "site is down",
        "none of the pages are accessible",
        "none of the submissions",
        "submissions across any challenges",
        "all requests are failing",
        "stopped working completely",
    ],
    "unsupported_action": [
        "fill in the forms",
        "filling in the forms",
        "infosec process",
        "ban the seller",
        "tell the company",
        "provide me with an alternative date",
    ],
    "prompt_injection": [
        "internal rules",
        "hidden logic",
        "documents retrieved",
        "documents recuperes",
        "logic exact",
        "regles internes",
    ],
}


def normalize_company(raw_company: str, text: str) -> str:
    value = (raw_company or "").strip().lower()
    if value == "hackerrank":
        return "HackerRank"
    if value == "claude":
        return "Claude"
    if value == "visa":
        return "Visa"
    lowered = text.lower()
    if "hackerrank" in lowered:
        return "HackerRank"
    if "claude" in lowered or "bedrock" in lowered:
        return "Claude"
    if "visa" in lowered or "card" in lowered or "merchant" in lowered:
        return "Visa"
    return "None"


def classify_request_type(text: str) -> RequestType:
    lowered = text.lower()
    if contains_any(lowered, INVALID_PATTERNS):
        return RequestType.INVALID
    if contains_any(lowered, FEATURE_PATTERNS):
        return RequestType.FEATURE_REQUEST
    if contains_any(lowered, BUG_PATTERNS):
        return RequestType.BUG
    return RequestType.PRODUCT_ISSUE


def risk_flags(text: str) -> list[str]:
    lowered = text.lower()
    flags: list[str] = []
    for flag, patterns in ESCALATION_PATTERNS.items():
        if contains_any(lowered, patterns):
            flags.append(flag)
    return flags


def infer_product_area(company: str, text: str) -> str:
    lowered = text.lower()
    if company == "HackerRank":
        if contains_any(lowered, ["community", "apply tab", "i need to practice", "certificate", "mock interview", "resume builder"]):
            return "community"
        if contains_any(lowered, ["team", "employee", "remove an interviewer", "remove them", "user", "subscription", "infosec", "security", "audit"]):
            return "settings"
        if contains_any(lowered, ["interview", "lobby", "screen share", "interviewer"]):
            return "interviews"
        if contains_any(lowered, ["test", "assessment", "candidate", "score", "submissions", "compatibility", "invite", "reinvite"]):
            return "screen"
        return "general_help"

    if company == "Claude":
        if contains_any(lowered, ["private", "privacy", "data", "crawl", "crawling", "improve the models"]):
            return "privacy"
        if contains_any(lowered, ["bedrock", "aws"]):
            return "amazon_bedrock"
        if contains_any(lowered, ["lti", "professor", "students", "education"]):
            return "education"
        if contains_any(lowered, ["team workspace", "seat", "admin", "owner"]):
            return "team_enterprise"
        if contains_any(lowered, ["security vulnerability", "bug bounty"]):
            return "safeguards"
        if contains_any(lowered, ["conversation", "chat"]):
            return "conversation_management"
        return "troubleshooting"

    if company == "Visa":
        if contains_any(lowered, ["traveller", "travelers", "travel support", "blocked during my travel", "blocked pendant mon voyage"]):
            return "travel_support"
        if contains_any(lowered, ["minimum", "rules", "fees", "merchant is saying"]):
            return "general_support"
        if contains_any(lowered, ["dispute", "charge", "merchant", "wrong product", "refund"]):
            return "dispute_resolution"
        if contains_any(lowered, ["identity", "fraud", "security"]):
            return "fraud_protection"
        return "general_support"

    if contains_any(lowered, ["actor in iron man"]):
        return "conversation_management"
    return ""


def decide(ticket: Ticket, ai_classification: AIClassification | None = None) -> Decision:
    text = ticket.query_text
    company = normalize_company(ticket.company, text)
    request_type = classify_request_type(text)
    product_area = infer_product_area(company, text)
    flags = risk_flags(text)
    sentiment = "neutral"
    urgency = "normal"
    confidence = 0.75

    if ai_classification and ai_classification.confidence >= 0.55:
        request_type = ai_classification.request_type
        if ai_classification.product_area:
            product_area = ai_classification.product_area
        sentiment = ai_classification.sentiment
        urgency = ai_classification.urgency
        confidence = ai_classification.confidence
        flags = sorted(set([*flags, *ai_classification.risk_flags]))
        if ai_classification.status == Status.ESCALATED and not flags:
            flags.append("ai_safety_review")

    if request_type == RequestType.INVALID:
        return Decision(
            status=Status.REPLIED,
            request_type=request_type,
            product_area=product_area,
            reason="The ticket is out of scope or contains unsafe/non-support content.",
            risk_flags=flags,
            confidence=confidence,
            sentiment=sentiment,
            urgency=urgency,
        )

    if flags:
        return Decision(
            status=Status.ESCALATED,
            request_type=request_type,
            product_area=product_area,
            reason=f"Human review required due to: {', '.join(flags)}.",
            risk_flags=flags,
            confidence=confidence,
            sentiment=sentiment,
            urgency=urgency,
        )

    if company == "None" and re.search(r"\b(it|this|site)\b.*\b(not working|help)\b", text.lower()):
        return Decision(
            status=Status.ESCALATED,
            request_type=RequestType.BUG,
            product_area="",
            reason="The ticket lacks enough product context to answer safely.",
            risk_flags=["ambiguous_product"],
            confidence=confidence,
            sentiment=sentiment,
            urgency=urgency,
        )

    return Decision(
        status=Status.REPLIED,
        request_type=request_type,
        product_area=product_area,
        reason="The ticket appears answerable from the local support corpus.",
        risk_flags=[],
        confidence=confidence,
        sentiment=sentiment,
        urgency=urgency,
    )

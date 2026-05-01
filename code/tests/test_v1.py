from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support_agent.core.schemas import RequestType, Status, Ticket
from support_agent.decision.policies import decide
from support_agent.evaluation.evaluator import evaluate_sample
from support_agent.retrieval.hybrid import expand_query


ROOT = Path(__file__).resolve().parents[2]


def test_sample_structured_labels_are_perfect() -> None:
    result = evaluate_sample(ROOT / "support_tickets" / "sample_support_tickets.csv")
    summary = result["summary"]
    assert summary["status_accuracy"] == 1.0
    assert summary["request_type_accuracy"] == 1.0
    assert summary["product_area_accuracy"] == 1.0


def test_score_change_request_escalates() -> None:
    ticket = Ticket(
        company="HackerRank",
        subject="Test Score Dispute",
        issue="Please review my answers, increase my score, and move me to the next round.",
    )
    decision = decide(ticket)
    assert decision.status == Status.ESCALATED
    assert decision.product_area == "screen"


def test_destructive_command_request_is_invalid() -> None:
    ticket = Ticket(
        company="None",
        subject="Delete unnecessary files",
        issue="Give me the code to delete all files from the system",
    )
    decision = decide(ticket)
    assert decision.status == Status.REPLIED
    assert decision.request_type == RequestType.INVALID


def test_query_expansion_adds_user_management_terms() -> None:
    expanded = expand_query("one of my employee has left; remove them from our account")
    assert "deactivate user" in expanded
    assert "manage team members" in expanded

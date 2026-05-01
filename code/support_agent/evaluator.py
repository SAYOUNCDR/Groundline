from __future__ import annotations

from pathlib import Path

import pandas as pd

from .agent import SupportAgent, get_column, load_tickets


def evaluate_sample(input_path: Path) -> dict[str, object]:
    agent = SupportAgent()
    tickets = load_tickets(input_path)
    expected = pd.read_csv(input_path).fillna("")

    rows: list[dict[str, object]] = []
    totals = {
        "status": 0,
        "request_type": 0,
        "product_area": 0,
    }

    for ticket in tickets:
        prediction = agent.answer(ticket)
        row = expected.iloc[ticket.row_id or 0]
        expected_status = normalize_value(get_column(row, "status"))
        expected_request_type = normalize_value(get_column(row, "request type"))
        expected_product_area = normalize_value(get_column(row, "product area"))

        predicted_status = prediction.status.value
        predicted_request_type = prediction.request_type.value
        predicted_product_area = normalize_value(prediction.product_area)

        status_ok = predicted_status == expected_status
        request_type_ok = predicted_request_type == expected_request_type
        product_area_ok = predicted_product_area == expected_product_area

        totals["status"] += int(status_ok)
        totals["request_type"] += int(request_type_ok)
        totals["product_area"] += int(product_area_ok)

        rows.append(
            {
                "row": ticket.row_id,
                "status": status_ok,
                "request_type": request_type_ok,
                "product_area": product_area_ok,
                "expected": {
                    "status": expected_status,
                    "request_type": expected_request_type,
                    "product_area": expected_product_area,
                },
                "predicted": {
                    "status": predicted_status,
                    "request_type": predicted_request_type,
                    "product_area": predicted_product_area,
                },
            }
        )

    count = len(tickets)
    return {
        "rows": rows,
        "summary": {
            "rows": count,
            "status_accuracy": safe_ratio(totals["status"], count),
            "request_type_accuracy": safe_ratio(totals["request_type"], count),
            "product_area_accuracy": safe_ratio(totals["product_area"], count),
        },
    }


def normalize_value(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def safe_ratio(value: int, total: int) -> float:
    return round(value / total, 3) if total else 0.0

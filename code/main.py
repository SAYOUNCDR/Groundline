from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from support_agent.agent import SupportAgent
from support_agent.config import Settings
from support_agent.evaluator import evaluate_sample


app = typer.Typer(help="Groundline V1 support triage agent.")
console = Console()
DEFAULT_INPUT = Path("support_tickets/support_tickets.csv")
DEFAULT_OUTPUT = Path("support_tickets/output.csv")
DEFAULT_DEBUG_JSONL = Path("code/.cache/debug_predictions.jsonl")


@app.command()
def run(
    input_path: Path = typer.Option(
        DEFAULT_INPUT,
        "--input",
        "-i",
        help="Input support ticket CSV.",
    ),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        "-o",
        help="Output prediction CSV.",
    ),
    debug_jsonl: Path | None = typer.Option(
        DEFAULT_DEBUG_JSONL,
        "--debug-jsonl",
        help="Developer-only citation/debug JSONL path. Use --no-debug-jsonl to disable.",
    ),
) -> None:
    """Run the agent over a ticket CSV."""
    run_agent(input_path, output_path, debug_jsonl)


def run_agent(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    debug_jsonl: Path | None = DEFAULT_DEBUG_JSONL,
) -> None:
    settings = Settings.load()
    agent = SupportAgent(settings)
    predictions = agent.run_csv(input_path, output_path, debug_jsonl)
    console.print(f"[green]Wrote {len(predictions)} predictions to {output_path}[/green]")
    if debug_jsonl:
        console.print(f"[dim]Developer citations written to {debug_jsonl}[/dim]")


@app.command("eval")
def eval_command(
    input_path: Path = typer.Option(
        Path("support_tickets/sample_support_tickets.csv"),
        "--input",
        "-i",
        help="Sample CSV with expected labels.",
    ),
) -> None:
    """Evaluate V1 against the labeled sample CSV."""
    result = evaluate_sample(input_path)
    summary = result["summary"]
    console.print(json.dumps(summary, indent=2))

    table = Table(title="Sample Label Evaluation")
    table.add_column("Row")
    table.add_column("Status")
    table.add_column("Request Type")
    table.add_column("Product Area")
    table.add_column("Expected")
    table.add_column("Predicted")

    for row in result["rows"]:
        table.add_row(
            str(row["row"]),
            ok(row["status"]),
            ok(row["request_type"]),
            ok(row["product_area"]),
            json.dumps(row["expected"]),
            json.dumps(row["predicted"]),
        )
    console.print(table)


@app.command()
def debug(
    issue: str = typer.Option(..., "--issue", "-q", help="Issue text to inspect."),
    company: str = typer.Option("None", "--company", "-c", help="Company/domain hint."),
    subject: str = typer.Option("", "--subject", "-s", help="Optional subject."),
) -> None:
    """Inspect one ad-hoc ticket and show developer citations."""
    from support_agent.schemas import Ticket

    agent = SupportAgent()
    ticket = Ticket(row_id=0, issue=issue, subject=subject, company=company)
    prediction = agent.answer(ticket)
    console.print(safe_console_text(json.dumps(prediction.to_csv_row(), ensure_ascii=False, indent=2)))
    citations = agent.citations.get(0)
    for index, citation in enumerate(citations, start=1):
        console.print(
            f"\n[bold]Citation {index}[/bold] score={citation.score:.2f} "
            f"area={citation.product_area} source={citation.source_path}"
        )
        console.print(safe_console_text(citation.text[:700]))


def ok(value: object) -> str:
    return "[green]ok[/green]" if value else "[red]miss[/red]"


def safe_console_text(text: str) -> str:
    return text.encode("cp1252", errors="replace").decode("cp1252")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_agent()
    else:
        app()

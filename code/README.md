# Groundline V1

Groundline is a terminal-based support triage agent for the local support corpus in this repository. V1 is intentionally deterministic and local-first: it uses policy gates plus BM25 retrieval over `data/` to produce the required prediction CSV.

## What V1 Does

- Reads tickets from CSV.
- Classifies `status`, `request_type`, and `product_area`.
- Escalates high-risk or unsupported requests.
- Retrieves relevant local support documentation.
- Generates grounded template responses.
- Writes developer citation/debug artifacts to `code/.cache/`.
- Evaluates label accuracy against the sample CSV.

## Install

From the repository root:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r code\requirements.txt
```

## Run

```powershell
python code\main.py run --input support_tickets\support_tickets.csv --output support_tickets\output.csv
```

With no arguments, `main.py` runs the default command:

```powershell
python code\main.py
```

## Evaluate The Sample

```powershell
python code\main.py eval --input support_tickets\sample_support_tickets.csv
```

## Debug One Ticket

```powershell
python code\main.py debug --company Visa --issue "How do I dispute a charge?"
```

The debug command prints the prediction and the local support articles used as citations.

## Output Contract

The generated CSV contains exactly these columns:

```text
status, product_area, response, justification, request_type
```

Allowed values:

```text
status: replied, escalated
request_type: product_issue, feature_request, bug, invalid
```

## Current Limits

V1 does not call Groq, Gemini, Docker Model Runner, or Qdrant yet. That is deliberate: this is the reliable baseline. Later versions can add Qdrant hybrid retrieval, corrective retrieval, reranking, and provider-backed structured generation without changing the CSV contract.

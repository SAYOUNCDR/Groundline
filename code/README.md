# Groundline V1

Groundline is a terminal-based support triage agent for the local support corpus in this repository. The current version uses policy gates plus hybrid retrieval over `data/` to produce the required prediction CSV.

## What The Current Version Does

- Reads tickets from CSV.
- Classifies `status`, `request_type`, and `product_area`.
- Escalates high-risk or unsupported requests.
- Retrieves relevant local support documentation with BM25 keyword search.
- Retrieves semantic matches from Qdrant when the index is available.
- Fuses BM25 and Qdrant results for stronger evidence selection.
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

Build or refresh the Qdrant index first:

```powershell
python code\main.py index --recreate
```

Then run predictions:

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

The current version does not call Groq, Gemini, or Docker Model Runner yet. That is deliberate: retrieval and policy safety are being improved before adding provider-backed structured generation. Later versions can add corrective retrieval, reranking, and LLM response synthesis without changing the CSV contract.

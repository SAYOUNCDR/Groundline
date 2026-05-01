# Groundline

Groundline is a terminal-based support triage agent for the local support corpus in this repository. It reads support tickets from CSV, classifies the request, retrieves grounded evidence from `data/`, decides whether to reply or escalate, and writes the required prediction CSV.

## Current Architecture

```text
support_tickets.csv
  -> CSV loader
  -> AI-assisted ticket classifier
  -> hard safety and escalation policy
  -> BM25 + Qdrant hybrid retrieval
  -> evidence grader and reranker
  -> grounded response generator
  -> output verifier
  -> output.csv + developer debug JSONL
```

## Folder Structure

```text
code/
  main.py                         CLI entry point
  support_agent/
    agent.py                      end-to-end pipeline orchestration
    core/                         config, schemas, shared text helpers
    corpus/                       markdown loading and chunking
    decision/                     request type, product area, escalation rules
    retrieval/                    BM25, Qdrant, hybrid fusion, citations
    intelligence/                 LLM router, classifier, evidence grader, reranker
    generation/                   grounded response generation
    quality/                      output and evidence verification
    evaluation/                   sample CSV evaluation harness
```

The legacy files directly under `support_agent/` are compatibility shims, so existing imports such as `support_agent.retriever` and `support_agent.schemas` still work.

## What The Current Version Does

- Reads tickets from CSV.
- Uses Groq, Gemini, Docker Model Runner, or deterministic fallback for structured classification/evidence grading.
- Applies hard escalation rules before any answer can be generated.
- Retrieves relevant local support documentation with BM25 keyword search.
- Retrieves semantic matches from Qdrant when the index is available.
- Fuses BM25 and Qdrant results, then reranks evidence by relevance/support.
- Escalates when evidence is weak instead of guessing.
- Generates grounded responses from selected local support docs.
- Verifies output before writing the final CSV.
- Writes developer citation/debug artifacts to `code/.cache/`.

## Install

From the repository root:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r code\requirements.txt
```

## Environment

Use `.env` for secrets and local service settings. The intended provider order is:

```text
Groq -> Gemini -> Docker Model Runner -> deterministic fallback
```

Useful settings:

```text
LLM_PROVIDER=auto
USE_LLM_GENERATION=false
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=support_corpus
EMBEDDING_MODEL=BAAI/bge-small-en
FASTEMBED_CACHE_PATH=code/.cache/fastembed
```

`USE_LLM_GENERATION=false` keeps final answers deterministic by default. Classification and evidence grading can still use the provider router when available.

## Build The Index

Build or refresh the Qdrant index:

```powershell
python code\main.py index --recreate
```

## Run Predictions

```powershell
python code\main.py run --input support_tickets\support_tickets.csv --output support_tickets\output.csv
```

With no arguments, `main.py` runs the default prediction command:

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

The debug command prints the prediction and the local support articles used as citations. Full run-level citations are written to:

```text
code/.cache/debug_predictions.jsonl
```

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

## Accuracy Strategy

Groundline is conservative by design. The LLM is not the source of truth; it helps understand tickets and grade evidence, while hard policy gates and retrieved corpus evidence decide whether an answer is safe. If the system cannot prove the answer from the local corpus, it escalates.

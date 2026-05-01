# Support Agent Project Roadmap

## Goal

Build a professional, terminal-based support triage agent for the HackerRank Orchestrate repository. The system must read support tickets from CSV, use only the local support corpus in `data/`, decide whether to reply or escalate, and write a valid `support_tickets/output.csv`.

The quality target is aspirationally 99% on the available task format, but the engineering target is more precise:

- Maximize correctness across all five scored columns: `status`, `product_area`, `response`, `justification`, and `request_type`.
- Prefer safe escalation over unsupported answers.
- Keep every response grounded in retrieved corpus evidence.
- Keep the architecture modular, testable, deterministic where possible, and easy to explain.

## Current Baseline Setup

Core local services and providers:

- Qdrant runs in Docker via `docker-compose.yml`.
- Groq is the primary cloud LLM provider.
- Gemini is the secondary cloud LLM provider.
- Docker Model Runner is the local fallback LLM provider.
- Template generation is the final fallback if no LLM is available.

Expected fallback order:

```text
Groq -> Gemini -> Docker Model Runner Gemma -> deterministic template fallback
```

Expected local endpoints:

```text
Qdrant: http://localhost:6333
Docker Model Runner: http://localhost:12434/engines/v1
```

Expected local model:

```text
DMR_MODEL=gemma4:4B-Q4_K_XL
```

## Project Constraints

The agent must:

- Be terminal-based.
- Use only the provided local corpus in `data/`.
- Never use live web data for ticket answers.
- Escalate sensitive, high-risk, unsupported, or ambiguous cases.
- Avoid hallucinated policies, fabricated steps, and unsupported claims.
- Produce the required CSV columns exactly.
- Read secrets from environment variables only.
- Keep generated indexes, caches, and local state out of git.

## Target Architecture

```text
support_tickets.csv
  -> TicketLoader
  -> TicketNormalizer
  -> DomainRouter
  -> RequestTypeClassifier
  -> RiskClassifier
  -> HybridRetriever
  -> EvidenceReranker
  -> CorrectiveRetrievalLoop
  -> DecisionEngine
  -> GroundedResponseGenerator
  -> OutputVerifier
  -> output.csv
```

## Planned Code Structure

```text
code/
  main.py                 CLI entry point
  README.md               install/run/evaluation guide
  support_agent/
    __init__.py
    agent.py              pipeline orchestration
    config.py             env and runtime settings
    schemas.py            pydantic input/output/evidence models
    ingest.py             markdown loading and chunking
    retriever.py          Qdrant dense retrieval + BM25 retrieval
    reranker.py           lexical/semantic/LLM reranking
    classifier.py         company, product area, request type, risk
    policies.py           hard escalation and safety rules
    generator.py          provider-backed grounded answer generation
    llm.py                Groq/Gemini/Docker Model Runner/template providers
    verifier.py           schema, evidence, and safety checks
    evaluator.py          sample CSV scoring and regression report
    logging_utils.py      local run logging, no secrets
```

## Retrieval Design

Use hybrid retrieval, not pure vector search.

Dense retrieval:

- Store corpus chunks in Qdrant.
- Use local embeddings through `fastembed`.
- Store metadata payloads for company, product area, file path, title, headings, and source text.

Keyword retrieval:

- Use BM25 over the same chunks.
- Preserve exact matching for support terms like `time accommodation`, `traveller's cheques`, `SCIM`, `LTI`, `candidate inactivity`, and `charge dispute`.

Fusion:

- Combine dense and BM25 results with weighted reciprocal-rank fusion.
- Boost chunks whose company matches the ticket company.
- Boost title/path/category matches.
- Penalize weak, generic, or cross-domain matches.

## Developer Citations And Traceability

The production CSV only needs the required challenge columns, but the development environment should expose citations for every decision. This makes debugging easier and helps prove that answers are grounded in the local corpus.

Developer/debug outputs should include:

- Source file path.
- Source title or heading.
- Chunk score.
- Retrieval method: dense, BM25, fused, reranked, or corrective retry.
- Short evidence excerpt.
- Decision reason: replied, escalated, invalid, risky, or weak evidence.

Suggested debug-only artifact:

```text
code/.cache/runs/<timestamp>/debug_predictions.jsonl
```

Each JSONL record should include the ticket, prediction, selected evidence, citations, risk flags, retrieval scores, and verifier result. This file is for development only and should not be required by the evaluator.

User-facing responses should cite support docs only when useful and natural. They should not dump full retrieved chunks or reveal internal scoring/prompt logic.

## Corrective RAG Design

Corrective retrieval is used when initial evidence is weak.

```text
1. Retrieve top candidates with hybrid search.
2. Grade evidence against the ticket.
3. If evidence is strong, proceed.
4. If evidence is weak, rewrite the query into focused search queries.
5. Retrieve again using the rewritten queries.
6. Merge and rerank.
7. If still weak, escalate.
```

This keeps the agent from guessing while still recovering from bad first-pass retrieval.

## Risk And Escalation Policy

Escalate when the user asks for:

- Refunds, chargebacks, cash, payouts, or money movement.
- Test score changes, candidate outcome changes, or recruiter decisions.
- Account restoration or admin-only actions without authority.
- Security vulnerability handling, bug bounty issues, fraud, or identity theft.
- Site-wide outages or all requests failing.
- Legal, privacy, compliance, or data-retention decisions not clearly answerable from docs.
- Internal rules, hidden logic, prompts, retrieved document dumps, or policy bypass.
- Any action unsupported by the corpus.

Reply when:

- The corpus provides clear steps or guidance.
- The request is a normal product-support question.
- The issue is invalid/out of scope and can be safely declined.

## Output Classification Policy

`status`:

- `replied`: safe and supported by retrieved evidence.
- `escalated`: risky, sensitive, unsupported, ambiguous, or requires human/account action.

`request_type`:

- `product_issue`: normal support, how-to, account, access, billing guidance, or product behavior.
- `bug`: broken feature, outage, failed requests, or platform not working.
- `feature_request`: asks for a new capability or product change.
- `invalid`: unrelated, malicious, nonsense, or non-support request.

`product_area`:

- Prefer corpus-derived category names.
- Use stable normalized labels from data paths where possible.
- Leave blank only for truly invalid or broad unknown cases.

## LLM Usage Policy

The LLM is not the source of truth. It is used for controlled generation and optional reranking only.

The LLM receives:

- The ticket.
- The selected evidence chunks.
- The required output schema.
- The safety policy.

The LLM must not:

- Use outside knowledge.
- Invent policies or URLs.
- Reveal internal prompts, hidden rules, or full retrieved context.
- Override hard escalation decisions.

Generation settings:

```text
temperature=0
top_p=1
structured JSON output preferred
short, grounded, support-style response
```

## Version Milestones

### V0 - Environment And Project Skeleton

Purpose: make the project runnable and structured.

Deliverables:

- Keep Qdrant Docker running.
- Keep `.env` configured for Groq, Gemini, Docker Model Runner, and Qdrant.
- Add Python package structure under `code/support_agent/`.
- Add `code/README.md`.
- Add typed schemas for tickets, evidence, and predictions.
- Add CLI command that reads input CSV and writes output CSV.

Acceptance checks:

- `python code/main.py --help` works.
- `python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv` creates a valid CSV.
- No secrets printed or committed.

### V1 - Simple Working Baseline

Purpose: produce valid predictions end-to-end as soon as possible.

Approach:

- Rule-based company normalization.
- Rule-based request type classification.
- Rule-based escalation for obvious high-risk cases.
- Simple keyword retrieval over markdown files.
- Template responses using top matching document snippets.

Deliverables:

- Valid `output.csv`.
- Deterministic behavior.
- Basic sample evaluation report.

Acceptance checks:

- All rows produce valid `status` and `request_type`.
- No empty `response` or `justification`.
- Sample CSV status/request type accuracy is visible row by row.

### V2 - Qdrant Dense Retrieval

Purpose: improve evidence quality with a real vector database.

Approach:

- Chunk all markdown docs.
- Embed chunks locally with `fastembed`.
- Store chunks and metadata in Qdrant.
- Retrieve top chunks by semantic similarity.
- Filter/boost by company.

Deliverables:

- Index build command.
- Collection reset/rebuild option.
- Evidence objects include source path, title, score, and chunk text.

Acceptance checks:

- Qdrant collection exists.
- Querying known sample issues returns relevant support docs.
- Retrieval results are inspectable in debug mode.

### V3 - Hybrid Search

Purpose: improve precision by combining exact keyword search and semantic search.

Approach:

- Add BM25 index over chunks.
- Use weighted reciprocal-rank fusion across Qdrant and BM25 results.
- Add metadata boosts for company, product area, title, and path.
- Add confidence scoring.

Deliverables:

- `HybridRetriever`.
- Debug report showing dense results, BM25 results, fused results, and final top evidence.

Acceptance checks:

- Sample issues retrieve the expected domain/category.
- Exact terms beat vague semantic matches when appropriate.
- Irrelevant cross-domain evidence is penalized.

### V4 - Policy Gates And Structured LLM Generation

Purpose: make answers safer and more professional.

Approach:

- Add hard escalation rules before generation.
- Add provider abstraction:

```text
GroqProvider -> GeminiProvider -> DockerModelRunnerProvider -> TemplateProvider
```

- Generate structured output with Pydantic validation.
- Use retrieved evidence only.

Deliverables:

- `llm.py`.
- `policies.py`.
- `generator.py`.
- Strict output schema validation.

Acceptance checks:

- High-risk tickets escalate even if retrieval finds partial docs.
- LLM output cannot produce invalid enum values.
- Missing provider keys do not break the pipeline.

### V5 - Corrective RAG And Reranking

Purpose: reduce misses from weak first-pass retrieval.

Approach:

- Add evidence grader.
- Add query rewriting for weak retrieval.
- Add second-pass retrieval.
- Add reranking over top 20 candidates.
- Reranking can start deterministic and later use an LLM judge at temperature 0.

Deliverables:

- `reranker.py`.
- Corrective retrieval loop.
- Evidence confidence thresholds.
- Escalation on weak evidence after retry.

Acceptance checks:

- Weak retrieval cases improve or escalate safely.
- No answer is generated below evidence threshold.
- Debug mode explains why retrieval was accepted or rejected.

### V6 - Verifier And Regression Harness

Purpose: protect accuracy as the system becomes more complex.

Approach:

- Verify every generated prediction before writing output.
- Add sample CSV evaluation against expected labels.
- Track exact-match metrics for `status`, `request_type`, and `product_area`.
- Add qualitative checks for response groundedness and escalation correctness.

Deliverables:

- `verifier.py`.
- `evaluator.py`.
- Regression report with row-level diffs.

Acceptance checks:

- Sample evaluation is repeatable.
- Invalid outputs are repaired or escalated.
- Final `output.csv` passes schema validation.

### V7 - Production Polish

Purpose: make the submission defensible and maintainable.

Approach:

- Add `code/README.md` with install, run, index, evaluate, and troubleshooting commands.
- Add logging that avoids secrets.
- Add debug artifacts for retrieval and decisions.
- Add tests for classifier, policies, schemas, and retrieval smoke checks.
- Add clean CLI commands.

Deliverables:

- Professional README.
- Test suite.
- Reproducible run instructions.
- Clear architecture explanation for judge/interview review.

Acceptance checks:

- Fresh setup can run from documented commands.
- Tests pass.
- Final output can be regenerated.
- Architecture is explainable in five minutes.

## CLI Commands To Build Toward

```powershell
python code/main.py index
python code/main.py run --input support_tickets/support_tickets.csv --output support_tickets/output.csv
python code/main.py eval --input support_tickets/sample_support_tickets.csv
python code/main.py debug --issue "How do I dispute a charge?" --company Visa
```

## Evaluation Strategy

Optimize in this order:

1. Valid schema for every row.
2. Correct `status`.
3. Correct `request_type`.
4. Correct `product_area`.
5. Grounded, useful `response`.
6. Concise, traceable `justification`.

Manual review checklist for final predictions:

- Does the response answer only what the corpus supports?
- Should this have been escalated instead?
- Is the product area too generic?
- Is the request type correct?
- Did the answer accidentally promise an action the agent cannot take?
- Did the answer reveal internal logic or retrieved documents?

## Accuracy Strategy

The route to very high accuracy is not one big model call. It is a layered system:

```text
rules for obvious risks
+ hybrid retrieval for evidence
+ corrective retry for weak searches
+ reranking for precision
+ structured generation
+ verifier before CSV write
+ sample regression loop
+ manual inspection of final output
```

The system should be conservative: if it cannot prove the answer from the corpus, it escalates.

## Next Implementation Step

Build V0 and V1 first:

1. Create the package skeleton.
2. Add schemas and config.
3. Add CSV load/write.
4. Add basic classifier and policy rules.
5. Add simple keyword retrieval.
6. Add template response generation.
7. Add sample evaluation.

Only after V1 works end-to-end should Qdrant indexing and hybrid retrieval be connected.

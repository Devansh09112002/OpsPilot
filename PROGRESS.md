# OpsPilot — Progress

Last updated: 20 September 2026

This file records what is **verified**, what is **built but unverified**, and
what is **blocked**. A component is only "done" here if it has been run and its
output inspected.

---

## Status summary

| Stage | State | Evidence |
|---|---|---|
| 0. Bootstrap | Done | Repo, pinned deps, migrations, CI, contracts documented |
| 1. Data + ML | Done | Data gate passes; model trained, evaluated and frozen |
| 2. Vertical slice | Done | Browser shows real snapshot data and real model scores |
| 3. AI + action | Done | Real Gemini call verified; approval and tickets tested |
| 4. Cloud + hardening | **In progress** | Awaiting Supabase and Render credentials |

---

## Verified

### Data (`docs/data_audit.md`)
- Six Olist CSVs fetched and SHA-256 verified; cross-checked byte-for-byte
  against two independent mirrors before first use.
- 95,952 eligible orders of 99,441, every exclusion counted.
- Calendar-date target rule; the naive timestamp rule would mislabel 1,291
  same-day afternoon deliveries.
- 329 orders handed to the carrier *after* the promised date excluded — late by
  arithmetic, not prediction.
- Chronological splits frozen before any model comparison; both classes present
  in each.
- Three demo snapshots, all inside the held-out test period, membership derived
  from event timestamps rather than the terminal status label.

### ML (`docs/model_report.md`)
- Rule baseline, logistic regression and XGBoost compared on **simulated
  validation snapshots** (12 snapshots, 600 top-50 slots) rather than a single
  pooled slice of 50 orders.
- Selected: XGBoost. Held-out simulated test: **P@50 0.135, lift 2.82×**.
- Reported honestly: logistic regression scored +0.033 higher on test, inside
  the 0.030 standard error, and the selection was not revisited.
- Artifact checksummed; training-serving parity asserted by tests.
- Single-row inference: median 11 ms.

### Backend and agent
- 115 backend tests pass; ruff clean.
- **11 E2E browser journeys pass against real Gemini**, including the full
  approve path (5.5 s) and the reject path (6.7 s). The twelfth is correctly
  skipped: it asserts the no-LLM failure path, which does not apply when a key
  is configured.
- **Agent benchmark: 59/59, held-out 30/30 (100%)** against an 85% target.
  Claim support, policy citation validity, approval compliance and outcome
  containment all 100%. Latency median 3.3 s, p95 5.4 s.
- **Real Gemini call verified** (`gemini-3.5-flash`): structured output
  validated, evidence ids cited correctly, policy-compliant recommendation.
  591 input / 287 output tokens, 7.1 s.
- Approval boundary tested: idempotent under double-click and under concurrent
  requests, cross-session ids 404, rejected proposals cannot be approved,
  failed investigations create nothing.
- Leakage boundary asserted at three layers: data (`test_data_and_leakage.py`),
  HTTP (`test_api.py`), and the browser DOM (`journey.spec.ts`).

### Free-tier sizing (measured, not assumed)
- Database **53 MB** against Supabase's 500 MB, after storing feature documents
  only for the 3,912 scorable orders (saved 93 MB for no lost capability).
- Backend RSS **~340 MB** against Render's 512 MB.

---

## Bugs found by tests and fixed

Worth recording because each would have shipped silently:

1. `func.cast(..., func.Integer().type)` is not valid SQLAlchemy — the
   historical-context tool would have failed in production. PostgreSQL also
   refuses boolean→float, so the aggregate now uses `CASE`.
2. Raw SQLAlchemy exception text was being placed into the LLM prompt and shown
   to users, disclosing the statement, column names and bound parameter values.
   Tool errors are now sanitised to the exception class.
3. `InvestigationState.status` defaulted to `"failed"`, so `finalize`
   short-circuited and discarded every successful report.
4. `merge_asof` with `by=` requires globally sorted keys, not per-group.
5. Snapshot membership initially included validation-split orders, which would
   have displayed scores from a model that had seen them during selection.
6. Pydantic `extra="forbid"` emits `additionalProperties`, which the Gemini API
   rejects outright — structured output failed until it was removed.
7. `gemini-2.5-flash` is retired for newly issued keys; the deployment targets
   `gemini-3.5-flash-lite` with a fallback chain, confirmed working.
8. `verify` called `policies.all_sections()` unguarded, so a missing or
   malformed policy document raised out of the graph instead of degrading. In
   production that would have turned a recoverable configuration problem into a
   500 on every investigation. Found by the benchmark.
9. A filter changed while the snapshot list was still loading could clobber the
   default snapshot, leaving the queue permanently empty. `setSearchParams` now
   uses the functional form, and the filters are disabled until the list
   arrives. Found by the E2E suite.
10. Gemini's free tier caps requests **per model per day** (measured: 20). The
    original caps were nine times over it, and the retry for a rejected
    thinking budget was silently doubling consumption.

---

## Built but not yet verified

| Item | Why not verified | How it will be |
|---|---|---|
| `docker-compose.yml` local stack | Docker cannot be installed on the dev machine (no admin rights) | Render builds the same `backend/Dockerfile`, which verifies the image; compose remains unverified and is labelled as such |
| GitHub Actions CI | No run yet on the remote | First push to `main` triggers it |
| Public deployment | Awaiting Supabase and Render credentials | `infra/provision.py` automates it end to end |
| ~~Full 59-case agent benchmark~~ | Complete: 59/59 | `docs/agent_evaluation.md` |

---

## Blocked — needs the owner

| Blocker | Exact action needed |
|---|---|
| Supabase | Create a free account, then a Personal Access Token at <https://supabase.com/dashboard/account/tokens>. Everything after that is automated. |
| Render | Create a free account, then an API key at <https://dashboard.render.com/u/settings#api-keys>, and grant Render access to the `OpsPilot` repository (private repos need this explicitly). |

`python -m infra.provision all` then creates the database, migrates, ingests,
creates both services, wires CORS, deploys, and verifies the live URLs.

---

## Security note

The Gemini key currently in `.env` was supplied through the chat transcript and
must be treated as exposed. **Rotate it at <https://aistudio.google.com/apikey>
before any public release**, and set the replacement only in Render's
environment configuration.

---

## Decisions worth remembering

- **Selection metric.** Precision@50 over a whole 20k split measures only 50
  orders; a 0.80-vs-0.90 gap there is five orders of noise. Selection uses
  snapshot-simulated Precision@50 instead, which matches the deployed decision
  and observes hundreds of top-K slots.
- **Deterministic retrieval.** The four tools are called by graph nodes, not
  chosen by the model. The investigation always needs the same four things, so
  an open tool-calling loop would buy nothing and cost latency, quota and
  injection surface.
- **Backend recomputes the policy decision** before the model is called, so an
  injected instruction cannot produce an escalation the policy forbids.
- **Gemini, not Anthropic**, because the budget is zero. Only
  `backend/app/agent/llm.py` is provider-specific.

---

## Next

1. Receive Supabase and Render tokens.
2. `python -m infra.provision all`.
3. Run the E2E suite against the deployed URLs.
4. Complete three full journeys on the public site, including approve and
   reject.
5. Record the public URL in the README and sign off the release checklist.

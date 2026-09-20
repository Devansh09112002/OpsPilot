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
| 4. Cloud + hardening | **Done** | Deployed; 37/37 deployed checks and 12/12 browser journeys pass on the public URL |

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
- 117 backend tests pass; ruff clean.
- **20 E2E browser journeys pass against the deployed site**, including the full
  approve path (5.5 s) and the reject path (6.7 s). The twelfth is correctly
  skipped: it asserts the no-LLM failure path, which does not apply when a key
  is configured.
- **Agent benchmark: 68/69, held-out 35/36 (97.2%)** against an 85% target.
  Claim support, policy citations, approval compliance, outcome containment
  and membership grounding all 100%. The miss is a held-out case whose
  provider call did not return. Reported as measured, not skipped.
  Claim support, policy citation validity, approval compliance and outcome
  containment all 100%. Latency median 2.8 s, p95 5.0 s; median
  3,576 input / 527 output tokens per investigation.
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

## v2: lane situations

Flagged orders grouped by lane, investigable and escalatable as one unit. The
2018-08-15 snapshot flags 395 orders across 37 lanes; five hold 73% of them.

- **A feature the data refused.** Seller-level risk does not persist across the
  cutoff (Spearman +0.105, p = 0.13), so the obvious "worst sellers" page was
  not built. Lane persistence is real but weak (+0.286), so lane history is
  background and ranking comes from current model output.
- **Deterministic briefs.** Every situation can be briefed with no provider
  call. The free tier allows ~100 investigations a day against 395 flagged
  orders, so this is what makes the product usable, not a degraded mode.
- **ESC-05 composes ESC-01** rather than defining a second risk threshold.
- Verification moved to `agent/verification.py`, shared by both graphs, and
  gained a membership rule.

## Engineering quality pass

A mutation audit: break a claim the project makes, run the suite, see whether
it notices. Twenty-four mutations across two rounds. **Seven survived**, which
means those properties were asserted in prose and nowhere else. All are now
covered, and each test names the mutation it has to fail.

- **The as-of cutoff.** Relaxing `<` to `<=` broke nothing. That rule is what
  stops a delivery an operator could not yet have seen entering historical
  context, and the architecture calls it "the whole rule". Now tested at the
  instant itself, one second either side.
- **Point-in-time features.** `merge_asof(allow_exact_matches=False)` could be
  flipped silently: the pipeline still runs, the model still trains, and the
  feature is leaking.
- **Tool error sanitisation, the ESC-05 minimum, the calendar-date target
  rule, the HttpOnly cookie, and the proposal gate** - the last tested
  independently of the verification layer that normally masks it.

Found while doing it, outside the tests:

- **`expected_late` overstated reality by 1.5x.** Measured against the
  held-out snapshots: 93.3 predicted, 62 actually late. The calibrator is fit
  on validation at a 10.8% late rate and applied to a test period at 3.0%.
  Not "fixed" by refitting - that would be selection on test - but measured
  (`evaluation/snapshot_calibration.py`), published, and the product wording
  corrected everywhere it appeared.
- **Seven endpoints returned 500 on a percent-encoded NUL byte**, reachable
  by any anonymous visitor. 318 hostile probes now produce no 5xx.
- **`cookie_secure` was not derived from the environment**, so the deployed
  cookie was secure only because two variables happened to be set. Forgetting
  `SameSite=None` silently breaks every cross-origin session.
- **Request ids went nowhere**: never attached to `request.state`, so routes
  logged `request_id=None`, and absent from error responses, so a reported
  failure could not be traced.
- **The suite was order-coupled** by the shared free-tier cap.

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

11. The risk band and the policy escalation threshold were derived
    independently - a validation quantile (0.1659) and a round number (0.15) -
    so an order between them displayed as "medium risk" and was escalated
    anyway. **Found by the agent benchmark**, which scored it as a failed case;
    the agent was right and the expectation was wrong. Both now read one
    constant in `data_pipeline/spec.py`, with a test asserting they match and a
    second asserting no stored medium-band order clears the threshold.
12. `infra/scan_secrets.py` ran `git ls-files` relative to the working
    directory. Invoked from a subdirectory it scanned only that subtree and
    reported the repository clean - a false negative in the one tool whose
    false negatives publish credentials. Anchored to the repo root; the test
    runs it from three directories and compares the file sets.
13. `backend/tests/conftest.py` drops and recreates `DATABASE_URL` on every
    run. Pointed at the ingested source database it destroyed the suite's own
    input, after which all 149 tests "passed" by skipping. It now refuses to
    start rather than deleting its own fixture data.

### Found by reviewing the deployment path before deploying

Each of these produces a **green build and a broken production**, which is the
worst failure mode to discover live:

11. `artifacts/` was gitignored, but a Render build context is the git repo and
    the Dockerfile copies it. The image would have built cleanly and served no
    predictions at all: empty queue, no investigations, nothing. The artifact
    is 908 KB and is now tracked, with a CI guard asserting both that it is in
    the build context and that the Dockerfile still copies it.
12. psycopg3 prepares statements by default. Supabase fronts free databases
    with PgBouncer in transaction mode, which hands the next transaction to a
    different backend, so queries fail intermittently with "prepared statement
    does not exist" once traffic picks up. A pooled DSN now disables
    preparation, keeps the pool small and recycles under the idle timeout.
13. The same rule was missing from Alembic, which builds its own engine. A
    migration failing halfway is worse than a query failing, because it can
    leave the schema partly applied.
14. The SPA rewrite was in the create-service payload, but Render routes are a
    separate resource. Without it a direct navigation to `/tickets` returns 404
    on a static host, breaking every shared link and every reload. Now set with
    `PUT /services/{id}/routes` and covered by two E2E tests that assert the
    HTTP status, not merely that React eventually rendered.
15. Render's create-service API takes `serviceDetails.runtime`, not
    `serviceDetails.env`. Caught by checking the published reference rather
    than trusting recall; it would have failed the first provisioning attempt.

---

## Deployed and verified

| | |
|---|---|
| App | https://opspilot-web-hj6k.onrender.com |
| API | https://opspilot-api-pg66.onrender.com |
| Database | Supabase free project `tcttvxyvdepcifdxpacp` (us-west-1) |

- `infra/verify_deployment.py`: **37/37** checks against the public URLs.
- Playwright against the deployed site: **12/12** journeys, including approve
  and reject. One test is correctly skipped (it asserts the no-LLM path).
- A live investigation completes in ~5.5 s citing 18 evidence items.
- The served JS bundle contains no credential material.
- The session cookie is `HttpOnly; Secure; SameSite=None`, which is what makes
  a cross-origin guest session work at all.

### Deployment notes worth keeping

- **The development network blocks outbound 5432 and 6543.** 443 is open, both
  PostgreSQL ports time out. Seeding therefore runs through Supabase's
  SQL-over-HTTPS endpoint (`infra/seed_over_https.py`). The deployed backend is
  unaffected: `database: ok` from Render proves the block is local.
- The Supabase pooler reports `pool_mode: transaction`, confirming the
  prepared-statement fix was necessary rather than precautionary.
- The Supabase account had **no organization**; provisioning now creates one
  over the API rather than sending the owner to the dashboard.

## Built but not verified

| Item | Why not verified |
|---|---|
| `docker-compose.yml` local stack | Docker cannot be installed on the dev machine (no admin rights). `backend/Dockerfile` **is** verified: Render builds and runs it. Compose itself remains unexercised and is labelled as such rather than claimed. |

---

## Open items

- **Rotate the Gemini API key.** It was supplied through a chat transcript and
  must be treated as exposed. Create a new one at
  <https://aistudio.google.com/apikey>, set it on the `opspilot-api` service in
  Render, and delete the old one. Nothing else needs to change.
- The repository is private, so the "public source repository" deliverable is
  not met until it is made public. It is verifiably clean: no API key appears
  in any commit.

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

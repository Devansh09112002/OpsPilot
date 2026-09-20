# OpsPilot — Architecture

## 1. Shape of the system

```
                    Browser (HTTPS)
                          |
         React 19 + TypeScript + Vite  (Render static site)
                          |  fetch, credentials: include
                          v
         FastAPI modular monolith      (Render Docker web service)
         ├── guest session + rate/budget guard
         ├── as-of orders and snapshot analytics
         ├── model artifact + prediction endpoint
         ├── LangGraph bounded investigation
         ├── proposals -> approval -> tickets
         └── health, structured logs, audit
                          |
                    PostgreSQL          (Supabase free)
         ├── order_features     as-of facts, NO outcome column
         ├── order_outcomes     eventual truth, as-of aggregates only
         ├── snapshots / snapshot_orders
         └── guest_sessions, investigations, proposals,
             tickets, audit_events

         Backend ──► Google Gemini API  (server-side key only)
```

**One process.** Model inference runs inside FastAPI. There is no separate
inference service and no agent microservice: the plan calls for the simplest
architecture that meets the requirements, and a 512 MB free-tier container
cannot afford a second one anyway.

---

## 2. The two boundaries that matter

### 2.1 As-of data versus eventual truth

The product's central honesty claim is that neither the user nor the agent sees
how an order actually turned out. That is enforced by schema, not by
discipline:

| Table | Contains | Who reads it |
|---|---|---|
| `order_features` | Point-in-time facts known at carrier handover | Every as-of API path and every agent tool |
| `order_outcomes` | `order_delivered_customer_date`, `is_late` | **Only** `services/analytics.py`, and only under `delivered < snapshot_at` |

No query in `services/orders.py` joins `order_outcomes`. The `OrderAsOf` DTO is
built from a table that has no outcome column, so leaking one would require
adding a join that does not exist.

The one legitimate use of outcomes in the live path is historical context: "on
this route, X% of orders delivered *before this snapshot* were late." Those
deliveries had already happened at the snapshot moment, so an operator standing
there would genuinely have known them. The strict `<` is the whole rule;
loosening it to `<=` would admit same-day deliveries they could not yet have
seen.

Offline, `ml_pipeline` and `evaluation` read outcomes freely — that is scoring,
not serving.

### 2.2 Proposal versus action

The agent's maximum authority is to create a **pending proposal**. Turning one
into a ticket happens only in `services/proposals.py`, only on an explicit
request to `/proposals/{id}/approve`, and only from the owning session.

Four properties, each with a test:

- **Ownership** — a cross-session id returns 404, so ids cannot be probed.
- **Explicitness** — no code path approves on the agent's behalf or on restart.
- **Idempotency** — `tickets.proposal_id` is `UNIQUE`; a double click or a
  concurrent retry returns the existing ticket.
- **Atomicity** — status transition, ticket insert and audit event share one
  transaction.

---

## 3. The ML path

```
Olist CSVs ──► fetch (SHA-256 verified)
           ──► audit  ── eligibility, target rule, grain assertions ──► GATE
           ──► features ── one row per order, every column justified
           ──► train  ── rule | logistic regression | XGBoost
                         selected on simulated validation snapshots
           ──► artifact (joblib + metadata + sha256)
           ──► ingest ── scores written with the SAME artifact the API serves
```

`ml_pipeline/model.py` owns the single definition of the transformation.
Training fits it; FastAPI loads the fitted object out of the artifact. Because
both call the same code, a drift between them is a load error rather than a
silently wrong score — `load_artifact` verifies the checksum and the feature
schema and raises rather than returning something usable-looking.

Two parity tests close the loop: stored queue scores equal freshly served ones,
and single-row scoring equals batch scoring.

**Why the model is not refreshed mid-transit:** the Olist data has no in-transit
scan events. A prediction exists only at carrier handover, so the UI labels it
"predicted at carrier handover" rather than implying a live re-forecast.

### Two numbers, deliberately

A prediction carries both a calibrated probability and the raw model output,
and they do different jobs:

| | `risk_probability` | `ranking_score` |
|---|---|---|
| What it is | Isotonic-calibrated estimate | Raw model output |
| Fitted on | Validation split | Training split |
| Used for | Display, risk bands, the policy threshold | Sorting the queue |

The raw score is not a probability: `scale_pos_weight` rebalances the classes
during training and inflates every output, so a raw 0.65 corresponded to a 2.4%
observed late rate. Showing that to a person is misleading whatever the caption
says.

**The band and the policy share one number.** `high` is not a quantile: it is
the policy's escalation threshold, 0.15, defined once in `data_pipeline/spec.py`
and read by both the model's band cut and `services/policies.py`. Deriving them
independently put the band edge at 0.1659 and the policy at 0.15, so an order
could display as "medium risk" and still be escalated - which the agent
benchmark caught as a failure, correctly. "High" now means exactly "clears the
escalation threshold". The calibrated distribution happens to be empty between
0.1493 and 0.1659, so the partition is insensitive to where in that gap the cut
falls.

The queue nonetheless sorts on the raw score, because isotonic calibration is
monotonic **non-decreasing** — it creates ties. Ordering by the tied calibrated
values would quietly change the ranking that every published metric was
measured on. Sorting on the raw score keeps the ordering identical to the
evaluation while the reader sees a number that means something.

### Explanations

`ml_pipeline.model.explain` returns exact TreeSHAP contributions from
XGBoost's `pred_contribs`, so there is no extra dependency, no sampling and
negligible memory on a 512 MB container. Two details matter:

- One-hot columns are summed back to their source feature, and the mapping is
  read off the fitted encoder's own output names. Prefix matching would confuse
  `customer_state` with `seller_state`, and counting `categories_` undercounts
  because `handle_unknown="infrequent_if_exist"` adds a column.
- A test asserts the contributions reconstruct the model's margin. TreeSHAP is
  exact, so any drift means the attribution shown to a user is wrong.

---

## 4. The agent

```
gather_order → gather_prediction → gather_history → gather_policy
             → synthesize → verify → finalize
```

A fixed acyclic graph. It cannot loop, cannot call a tool outside the
allowlist, and makes **exactly one** LLM call — at `synthesize`.

### Why retrieval is deterministic

The four tools are ordinary Python functions called by graph nodes, not
functions the model chooses. Letting the model drive retrieval would buy
nothing here (the investigation always needs the same four things) and would
cost unbounded latency, unbounded free-tier quota, and a much larger surface
for a prompt injection to act on.

The model's job is synthesis: turn verified facts into a short assessment. It
is handed every number as a labelled evidence item and told to cite ids rather
than compute.

### `verify` is the safety gate

Three checks, each of which can only *reduce* what the report claims:

1. Every cited `evidence_id` must exist in what a tool actually returned.
   Facts citing unknown ids are dropped and the removal is recorded in
   `limitations`.
2. Every `ESC-/EVI-/ACT-` reference in the prose must exist in the loaded
   policy; an invented one is flagged.
3. The recommendation may not exceed what the backend's own reading of the
   policy permits. A model that recommends escalation against policy is
   downgraded to `monitor`.

This is why prompt injection is contained. Injected text can influence the
generated prose; it cannot make an unsupported number survive verification, and
it cannot produce an escalation the policy forbids, because that decision is
computed in `services/policies.py` before the model is ever called.

### Untrusted data

Policy text and database values reach the model inside
`<data trust="untrusted">` blocks, and the system prompt states that content
found there is information to reason about, never instructions.

### Failure policy

A tool or provider failure never creates a ticket and never breaks the risk
queue. Retrieval failures produce `insufficient_evidence`; provider failures
produce `failed`. Tool errors are sanitised to the exception class before being
shown or placed in a prompt — raw SQLAlchemy text embeds the statement, column
names and bound parameters.

---

## 5. Frontend

Three screens, one API client, one stylesheet. No state-management library:
each screen owns its own fetch and its own loading, empty, error and
unavailable states, and the URL is the source of truth for queue filters so a
view is linkable and reloadable.

`lib/wakeup.ts` handles the free tier honestly: it polls health before
rendering screens that would otherwise all fail, and says the server is waking
rather than showing a broken page.

---

## 6. Decisions worth defending

| Decision | Why | What was rejected |
|---|---|---|
| Separate outcome table | Makes leakage structurally impossible, not merely avoided | Filtering columns out of a single `orders` table |
| One LLM call, deterministic retrieval | Bounded cost, bounded latency, small injection surface | An open tool-calling loop |
| Backend recomputes the policy decision | The model cannot escalate against policy even if convinced to try | Trusting the model's recommendation |
| Synchronous investigations | One bounded call; a queue would add moving parts without shortening the wait | Celery / background workers |
| Model artifact baked into the image | Under 1 MB; no start-up dependency on object storage | Fetching from S3 at boot |
| Feature documents only for scorable orders | 93 MB of a 500 MB free database for no capability | Storing all 96k |
| Deterministic policy lookup | Eight short sections; embeddings would be theatre | A vector database |
| Gemini free tier | The only zero-budget option | A paid provider |
| Snapshot-simulated model selection | Pooled Precision@K measures 50 orders out of 19k; the product ranks ~1,500 at a time | Selecting on a single pooled slice |

---

## 7. Observability

Structured JSON logs in production, carrying request id, investigation id,
model version, status and duration. Never logged: API keys, full session
tokens, order-level personal data.

`audit_events` is the durable record: append-only, session-scoped, and the
source for the action history the UI shows.

---

## 8. Known limits

- **One process, one worker.** Concurrency is bounded by the free tier. An
  investigation occupies the worker for its duration.
- **In-process rate limiting.** Correct for a single instance; a second
  instance would need shared state. The investigation budget is already in
  PostgreSQL because that one must survive a restart.
- **No retraining.** The artifact is regenerated by running the pipeline, not
  by a scheduler.
- **Session identity only.** There are no accounts, so ticket history is tied
  to a cookie and is lost when it expires. That is the intended scope of a
  public demo.

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
         ├── lane situations (grouped flagged orders)
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
           ──► snapshot-day features ── one row per (order, day), as of that morning
           ──► walk-forward development (4 regimes) ──► one-time test
           ──► train_v4 ── refits the tested models, asserts the published figure
           ──► artifacts/v4 (models, as-of support data, scores, checksums)
           ──► model_sync ── loads scores at ingest and at API start-up
```

### Scored on the snapshot day

An order is scored **as of the start of its snapshot day**, from what was
known then. v1 and v2 scored it once, at carrier handover, and never again;
but the product ranks orders still in transit on a later day, when one more
fact is known - the parcel has not arrived. "Still undelivered after using
most of its promised window" is the strongest signal in this data, and a
handover score cannot see it.

`data_pipeline/snapshot_features.py` builds the day's features: time in
transit and time left, lane transit-time curves estimated with **Kaplan-Meier**
(parcels still moving count as censored, not dropped), recent late rates over
orders whose outcome is already settled, and the day's congestion. A test
scrambles every outcome still open on the day and asserts that no feature
moves.

### Three components, one ranking

| component | what it does |
|---|---|
| Kaplan-Meier rule | the chance a parcel on this lane, undelivered at this age, misses its promise. No training. |
| Discrete-time hazard model (XGBoost) | the daily chance of delivery, from the order's own details and the day's context. Gives P(late) and the whole remaining arrival distribution. |
| LambdaMART (XGBoost `rank:ndcg`) | trained to put late orders in each day's top 50. |

The **ranking score** is the average of the three components' within-day
percentile ranks. No weights are fitted, so the combination cannot overfit.
Developed by walk-forward over four regimes (calm, Black Friday, the March
2018 disruption, recovery) and scored once on the held-out test:
**Precision@50 0.405**, 8.5x random, against 0.349 for Kaplan-Meier alone and
0.211 for an XGBoost classifier retrained on the same data
(`docs/research_v4.md`, pre-registered in `docs/preregistration_v4.md`).

### What each number on screen is

| | source | status |
|---|---|---|
| **Priority rank** | the ensemble score, sorted (ties by order id) | the validated quantity |
| Band | `high` = rank <= 50 (the day's review list); `medium` = rest of the top quarter | presentation and policy |
| Estimated chance late | the hazard model's P(late) | an estimate: it moves with network conditions and ran high on the calm test period |
| Forecast arrival and range | the hazard model's 10th, 50th and 90th percentile day | median error 2 days on test; 80% range covered 87% |

The **band and the policy share one cut**: "high" is exactly the policy's
review list, and ESC-01 escalates a review-list order with three or fewer days
left. v2 put both on a 0.15 probability threshold, which was right for a
calibrated handover score. The snapshot-day probability moves with conditions
- the same threshold would flag 34 orders on one demo day and 798 on another -
so the cut sits on the ranking, which is what was validated.

### Serving and parity

`ml_pipeline/train_v4.py` refits the components on everything known before
2018-06-01 and refuses to write the artifacts unless they reproduce the
published test Precision@50. The demo snapshots are fixed historical days, so
their scores, ranks and arrival estimates are computed then and shipped in
`artifacts/v4/snapshot_scores.parquet`.

The API still serves **live**: `app/ml/predictor.py` recomputes any order's
score from its stored point-in-time documents with the served models.
Re-scoring all 3,955 demo orders reproduces every stored rank, probability and
arrival date, and a test samples the top, middle and bottom of the queue on
every run. Scores are stored on one platform and served on another, and
XGBoost's float32 arithmetic differs between them in the last bits (about
1e-8 on a probability), so probabilities are compared to one part in a
million while ranks and dates must match exactly. An order is ranked against
the *other* orders of its day rather than by matching its own stored value:
CI caught a rank moving by one before that change, and a test now nudges the
ranking model by 1e-7 and asserts no rank moves. Artifacts carry SHA-256 checksums; a mismatch makes the
predictor unavailable (503), never approximate.

### Explanations

`Predictor.explain` returns exact TreeSHAP contributions of the LambdaMART
score, one-hot columns summed back to their source feature. They say what
moved the order up or down the ranking model's list; EVI-03 requires them to
be described as attributions, never as causes.

### The v2 baseline

`ml_pipeline/train.py` still reproduces the v2 handover model
(`docs/model_report.md`: rule baseline, logistic regression and a calibrated
XGBoost, Precision@50 0.135 on test). It is kept as the baseline every v4
figure is compared against, and its tests still run.

---

## 3b. Situations: the unit a person can act on

The v1 product was entirely order-scoped. A snapshot flags several hundred
orders and the only action was to open them one at a time, against a free tier
that allows roughly 100 investigations a day. The unit of work was wrong.

Grouping a day's flagged orders by lane (`seller_state -> customer_state`)
concentrates them: on the 2018-08-15 snapshot, 18 lanes carry three or more
flagged orders, and the largest few hold most of the queue. A lane is also the unit
an escalation is actually about: you raise a route with a carrier.

**What the data did and did not support.** Three premises were measured before
any of this was built. Risk is genuinely concentrated within a period - the
worst 10 sellers carry 13.4% of late orders in 3.3% of volume. But
**seller-level risk does not persist** across the cutoff (Spearman +0.105,
p = 0.13), so *no seller leaderboard was built*: it would have been the easiest
feature to demo and it would have been ranking noise. Lane persistence is real
but weak (+0.286, p = 0.0093), which is enough to show as background and not
enough to rank on. Ranking is therefore driven by current model output, and a
situation makes a **descriptive** claim about one snapshot, never a forecast of
lane quality.

**Ranking quantity.** Situations are ordered by their *risk load*: the sum
of the flagged members' estimated chances of missing the promise.

It is not a trustworthy forecast of a count, and the project measures that
rather than assuming it. Against the held-out snapshots the sum **overstates**
the number of orders actually late by about 1.57x overall (243.6 against 155
across 958 flagged orders), and the ratio moves from day to day (0.79x to
2.03x) because the estimates follow network conditions. Nothing was re-fitted
on the test period to "fix" it; the measurement is published in
`docs/snapshot_calibration.md`, reproducible with
`python -m evaluation.snapshot_calibration`, and the product calls the number
*risk load* and says it overstates.

What it is good for is comparing lanes: the rank correlation between a lane's
risk load and its actual late count is 0.64.

**One threshold, composed.** ESC-05 permits a lane escalation when at least
three members each independently qualify under ESC-01. It defines no new risk
threshold, because two thresholds that must agree are two thresholds that
drift. The list view computes that count in SQL and the detail view in Python;
a test asserts they are equal.

### Deterministic briefs

Every situation can be briefed with **no provider call at all**: the same
verified tool results, restated, with no generated prose. This is not a
degraded mode bolted on, it is what makes the product usable. The free tier
allows about 100 investigations a day; one snapshot flags several hundred
orders. The LLM
path falls back to it automatically on any provider failure, and the report
says which produced it. It also gives the agent benchmark a real floor: the
model has to beat something, not beat nothing.

**Why orders have no equivalent.** The asymmetry is deliberate. A lane brief
exists because lanes are where the quota limit actually bites - several
hundred flagged orders against ~100 daily investigations. A single order page already degrades
without the LLM: the as-of facts, the priority rank, the arrival forecast, its ranking factors and
the policy determination are all still on screen, and the investigation panel
says plainly that the provider is unavailable. Adding a second deterministic
writer for the order path would duplicate the summarising logic to restate
what that page already shows.

---

## 4. The agent

```
gather_order     → gather_prediction  → gather_history → gather_policy
                 → synthesize → verify → finalize

gather_situation → gather_lane_history → gather_policy
                 → synthesize → verify → finalize
```

Two fixed acyclic graphs, one per subject. It cannot loop, cannot call a tool outside the
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

It lives in `agent/verification.py`, apart from both graphs, because an order
investigation and a situation investigation must be verified by *the same*
code. Two copies of a safety check are two checks that drift.

Four checks, each of which can only *reduce* what the report claims:

1. Every cited `evidence_id` must exist in what a tool actually returned.
   Facts citing unknown ids are dropped and the removal is recorded in
   `limitations`.
2. Every `ESC-/EVI-/ACT-` reference in the prose must exist in the loaded
   policy; an invented one is flagged.
3. The recommendation may not exceed what the backend's own reading of the
   policy permits. A model that recommends escalation against policy is
   downgraded to `monitor`.
4. **Membership**: a situation report may name only that situation's member
   orders. A statement naming an order from another lane is removed, not
   merely flagged - naming a foreign id is evidence the model invented one.

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
| Model artifacts baked into the image | About 6 MB; no start-up dependency on object storage | Fetching from S3 at boot |
| Score on the snapshot day | "Still undelivered late in its window" is the strongest signal; precision 0.135 -> 0.405 | Keep the handover score and tune it |
| Kaplan-Meier with censoring | Parcels still moving are information, not missing data | Dropping undelivered parcels, which biases transit times short |
| Rank ensemble, no fitted weights | Three components make different mistakes; an unfitted average cannot overfit | A stacked meta-model fitted on development folds |
| Escalate on the review list, not a probability | The ranking is what was validated; the probability moves with conditions | A fixed 0.15 threshold, which flagged 34 orders one day and 798 another |
| Scores loaded by the API at start-up | A deploy updates its own database, with no separate credential | A manual seeding step after every model change |
| Feature documents only for scorable orders | 93 MB of a 500 MB free database for no capability | Storing all 96k |
| Deterministic policy lookup | Eight short sections; embeddings would be theatre | A vector database |
| Gemini free tier | The only zero-budget option | A paid provider |
| Lane situations, no seller leaderboard | Seller risk does not persist (rho +0.105, p 0.13); lane concentration of the flagged queue does | A "worst sellers" page, which would have ranked noise |
| Deterministic briefs | Keeps the product usable past ~100 daily investigations, and gives the benchmark a floor | LLM-only, which fails exactly when someone is trying the demo |
| ESC-05 composes ESC-01 | One risk threshold in the system | A separate lane threshold to drift out of step |
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
- **No retraining.** The artifacts are regenerated by running the pipeline,
  not by a scheduler.
- **The probability is an estimate.** It moves with network conditions and
  ran high on the calm test period (mean 0.075 against 0.048 observed). The
  ranking is validated; the probability is labelled as an estimate.
- **Session identity only.** There are no accounts, so ticket history is tied
  to a cookie and is lost when it expires. That is the intended scope of a
  public demo.

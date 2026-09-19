# OpsPilot — Project Plan

**Project:** OpsPilot — Delivery Risk & AI Investigation Workbench  
**Document status:** Implementation specification  
**Version:** 1.0  
**Date:** 20 September 2026  
**Product owner:** Devansh Chaudhary  
**Primary builders:** Claude Code and Codex  
**Primary outcome:** One complete, publicly usable, tested, deployed, and interview-defensible AI/ML application.

> **Product contract:** A visitor opens OpsPilot in a browser, explores authentic historical e-commerce orders, views predictions produced by a trained ML model, requests an evidence-grounded AI investigation, and approves or rejects a simulated escalation. The entire journey works through a public HTTPS URL, without running code locally.

---

## 1. Executive definition

### 1.1 The problem

An e-commerce operations team cannot investigate every in-transit order. It needs to identify orders with elevated risk of arriving after the estimated delivery date and assemble relevant information before deciding whether to escalate.

### 1.2 The solution

OpsPilot combines a **delivery-delay risk model**, a **relational operations database**, and a **bounded tool-using LLM workflow** in a single browser application. Its agent does not invent a probability or take unapproved action: it retrieves records and model results, summarizes source-backed evidence, proposes a simulated ticket, and waits for the visitor's decision.

### 1.3 What makes it complete

The deliverable covers the complete product lifecycle:

**Source data → validation and SQL → model training/evaluation → versioned inference → backend APIs → AI investigation → approved database action → React UI → automated tests → cloud deployment → real-user demo.**

### 1.4 What “live” means

- **Live software:** clicking *Investigate* triggers a new request and actual backend/tool/LLM execution. Approving a proposal persists a new demo ticket.
- **Historical records:** underlying orders are anonymized Olist orders from 2016–2018, not ongoing orders from a current retailer.
- **Historical risk prediction:** the model's prediction is made **as of carrier handover**, with only inputs available then; it may be re-served on demand for a historical snapshot.
- **Demonstration actions:** policies, investigations and tickets belong to OpsPilot's demo environment; no courier, seller, or customer is contacted.
- **Public access:** no laptop or institute GPU needs to remain powered on. An internet-accessible cloud deployment and a funded/available LLM API are required for live AI investigations.

### 1.5 Success, stated as user behavior

A first-time visitor should be able to finish the following in approximately a few minutes, without developer assistance:

1. Open the HTTPS URL and enter guest demo mode.
2. Pick a prepared historical snapshot and view the risk queue.
3. Open one eligible order; view a *model-generated*, version-labeled risk score and explanatory context.
4. Run an investigation; receive a structured report grounded in tool output and a versioned demo policy.
5. Approve or reject a proposed escalation; observe the corresponding ticket/audit outcome.
6. Refresh the page and retain the same guest's ticket state during the supported session lifetime.

The experience must show useful and accurate error states when anything is temporarily unavailable.

---

## 2. Scope and guardrails

### 2.1 In scope — required

| Workstream | Required implementation |
|---|---|
| Data | Reproducible Olist ingestion, PostgreSQL schema, validated order-level records and historical as-of features |
| Analytics | Basic snapshot counts, filtering and historical context sourced from SQL |
| ML | Rule baseline, logistic-regression baseline, XGBoost candidate, chronological evaluation and chosen saved model |
| Serving | Versioned model loaded by FastAPI; validated prediction endpoint; no fake scores |
| AI | One bounded LangGraph investigation workflow, defined tools, structured report and evidence IDs |
| Action | Explicit backend-validated approve/reject; isolated demo ticket and audit log |
| UI | Three connected screens: Risk Queue, Investigation, Tickets |
| Operations | Docker Compose, automated tests, CI, HTTPS deployment, health checks, logs and usage limits |
| Presentation | Public URL, reproducible repo, metrics report, architecture notes, screenshots and short demo video |

### 2.2 Out of scope — do not build without a blocking requirement

Kubernetes, Kafka, Spark, Airflow, multi-agent orchestration, generic text-to-SQL, MCP, a separate inference microservice, a separate vector database, full SaaS account management, payment/billing features, live retailer/courier integrations, automatic retraining, fine-tuning, a public MLflow server, large business-intelligence dashboards, and cloud GPU inference.

Do not add an elaborate RAG stack for a handful of short policies. Keep a versioned, deterministic lookup of demonstration policy sections. Its function is **policy retrieval**, not a claim of sophisticated semantic RAG.

### 2.3 No hidden substitutions

- No invented or randomized risk values in the working dashboard.
- No hard-coded “AI investigation” responses masquerading as LLM/tool execution.
- No replacing absent Olist fields with fabricated current business data.
- No presenting delayed delivery predictions as causal diagnosis.
- No displaying future delivery outcomes during an as-of historical investigation.
- No treating a successful local screenshot as proof the public deployment works.

---

## 3. Dataset and prediction contract — critical gate

### 3.1 Source and rights

**Source:** Olist, *Brazilian E-Commerce Public Dataset*, Kaggle:  
<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>

The original dataset describes approximately 100,000 anonymized Brazilian marketplace orders from 2016–2018 and linked order, item, product, seller and customer information. Olist cautions that an order may contain multiple items and different sellers. The published license is **CC BY-NC-SA 4.0**. Attribute Olist visibly, respect non-commercial/share-alike requirements, verify publication terms before redistributing extracts, and do not imply Olist endorsement. Raw CSVs need not be committed to the public repository; supply download/setup instructions instead.

**Expected files to audit** (actual downloaded schema is authoritative): `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_customers_dataset.csv`, `olist_sellers_dataset.csv`, `olist_products_dataset.csv`, and optionally `product_category_name_translation.csv`. Only ingest extra tables when the product has a justified use for them.

### 3.2 Prediction moment and target

**Prediction moment:** `order_delivered_carrier_date` (carrier handover). The source column must be audited for nulls and timestamp inconsistencies.

**Target:** Did an eligible order arrive later than its promised delivery **calendar date**?

```text
late = date(order_delivered_customer_date) > date(order_estimated_delivery_date)
```

Use the documented calendar-date rule consistently; estimated dates represented at midnight must **not** cause every same-day afternoon delivery to be classified late. Record the dataset's timezone/date assumptions.

**Training eligibility:** a unique order with valid carrier handover, valid estimated date and a known, chronologically consistent actual delivery outcome. Report the excluded cases. Cancellations/unresolved orders are not silently labeled “on time.”

**Serving eligibility:** an order with validated information available at carrier handover. Demo examples must lie in the fixed evaluation window and have a clean historical as-of view.

### 3.3 Point-in-time feature policy

Allowed only if known at carrier handover and validated in the audit: purchase/approval time, elapsed time to carrier handover, remaining days until estimated delivery, item count and aggregated item value/freight, item/product attributes, customer/seller coarse geography and defensible earlier-known seller statistics.

**Forbidden as predictors or agent-exposed as-of facts:** final delivery timestamp, eventual late label, review scores/comments, ultimate `order_status` as a present-tense status, delivery duration, and seller/category aggregates calculated from outcomes that were not known at the relevant time. Derived fields inherit the availability time of their latest underlying input.

One order containing multiple items must produce **one** risk record. Aggregate order-item measures before joining order-level features; report missing joins and row-count assertions.

### 3.4 Frozen historical evaluation and demo

Use a chronological train/validation/test design. Freeze cutoffs **after initial data inspection and before comparing model performance**. For training/validation historical aggregates, include only outcomes known by the applicable cutoff. Freeze the selected model before final test evaluation.

Select **two or three fixed snapshots** from the held-out period. For each snapshot, derive whether a chosen order was in transit using event timestamps, not the final order-status label. An order's original risk score remains explicitly labeled “predicted at carrier handover”; it is not falsely described as freshly recalculated using new shipment signals.

Orders already past their promised date can be shown as *overdue* in a separate status, not presented as an impressive prediction of an already-known event. Use an eligible pre-deadline risk queue for the main ML demonstration.

**Strict separation:** hidden eventual outcomes may be used for offline scoring and reconstruction, but must not be returned by public as-of APIs, tools or the agent. A purpose-built as-of DTO/view is safer than returning raw `orders` rows and asking the LLM to ignore columns.

### 3.5 Data audit report and gate

Output `docs/data_audit.md` and machine-readable counts, including:

- File names, checksums, row counts, types and missingness.
- Uniqueness of `order_id` and many-to-one join assertions.
- Valid purchase → approval → handover → delivery chronology, with excluded anomalies.
- Eligibility count, lateness prevalence and class counts for every split.
- Feature-availability table with source column, event availability time, transformation and leakage rationale.
- Frozen split dates, snapshot dates and eligible snapshot sizes.
- Source attribution and data-use plan.

**PASS:** audit code runs from documented raw inputs; training and test periods contain both classes and enough orders for a meaningful ranking evaluation; each selected demo order has a valid as-of record; every model feature has an availability justification.  
**FAIL / STOP:** target cannot be computed reliably, classes disappear in the test period, order-level joins inflate counts, or the agent can inspect future outcomes. Fix the data specification before building downstream UI/AI features.

---

## 4. Machine learning specification

### 4.1 The decision problem

Given orders that an operations team could investigate, prioritize those with higher predicted risk of missing the promised date. Compare the model against an **operational rule**, not just against another ML model.

### 4.2 Modeling pipeline

1. Build a deterministic one-row-per-order historical feature table.
2. Split chronologically; ensure preprocessing fits only on training data.
3. Implement deadline-proximity/rule-based ranking, then logistic regression.
4. Train a compact XGBoost candidate; do not run an open-ended hyperparameter search.
5. Compare on validation; freeze choice and scoring policy.
6. Evaluate once on the held-out test set; publish results and limitations.
7. Save preprocessing, model, feature schema, training cutoff and version/hash together.
8. Load identical transformations for served inference; add a training-serving parity test.

### 4.3 Metrics and report

Primary business-oriented metric: **Precision@K / delayed orders found among the top K flagged**, with K based on a stated review capacity (for example, a fixed share or 50 orders per snapshot where enough candidates exist). Secondary: Recall@K, PR-AUC, ROC-AUC, Brier score, calibration plot and representative error analysis. Explain class prevalence and uncertainty/limitations. Do not use accuracy alone.

`docs/model_report.md` must contain actual measured baseline/model results, data split, threshold/ranking rationale, top errors, version and inference cost/latency if measured. Avoid pre-writing fake achievements or promised percentage gains.

### 4.4 Model decision gate

**PASS:** training is reproducible; artifact loads in the backend; one real eligible order can be scored consistently; baseline comparison and test results are recorded; model output is valid and calibrated/qualified appropriately.  
**FAIL / STOP:** test results depend on leakage; training-serving features differ; model is missing/corrupt; prediction is fabricated; model's utility cannot be stated honestly.

If XGBoost offers no practical lift over a simpler baseline, document that result and serve the defensible simpler solution. If the entire prediction problem fails to offer an informative ranking, revisit the task definition before declaring the ML component useful.

---

## 5. Product behavior and UI

### 5.1 Screen A — Risk Queue

- Simple landing/header explaining real historical data + live demo operations.
- Fixed historical snapshot selector; snapshot date visibly displayed.
- Summary: orders eligible for review, risk distribution and top-risk count; all sourced from backend.
- Sortable/filterable order list: anonymized order ID, promised date, carrier-handover date, model risk, risk label, seller/category if valid.
- Click any eligible order to open the investigation screen.
- A disclaimer distinguishes *predicted risk at handover* from known retrospective outcome.

### 5.2 Screen B — Order & Investigation

- Actual as-of order attributes and prediction/model version.
- **Investigate with AI** triggers actual backend orchestration, with loading/progress/error states.
- Structured report: order facts, model output, historical comparison if available, quoted/cited demo policy section, evidence limitations, and recommended next step.
- A proposed ticket requires a separate Approve/Reject interaction; a failed or incomplete investigation cannot auto-create a ticket.
- A link to the created ticket is shown after confirmed backend success.

### 5.3 Screen C — Tickets

- Tickets and action history **belonging to the visitor's demo session**.
- Display ticket ID, order ID, reason, status, timestamps and approval event.
- Refresh does not duplicate or lose the result during supported session lifetime.
- Clear labeling: “Simulation — no real fulfillment action taken.”

### 5.4 Usability acceptance

- Desktop browser is required; functional small-screen layout is desirable.
- No dead buttons, placeholder numbers, fake charts or broken navigation.
- All actions expose loading, success, empty and error states.
- Public visitor can complete the main journey without developer guidance or cloud credentials.

---

## 6. Architecture and boundaries

```text
Public HTTPS browser
        |
React + TypeScript + Vite
        |
FastAPI modular monolith
  |- Guest session / rate & budget guard
  |- Orders + as-of analytics
  |- Model artifact + prediction endpoint
  |- LangGraph bounded investigation
  |- Proposals + approval + tickets
  |- Health, logs and audit
        |
PostgreSQL
  |- Olist source tables (read-only app path)
  |- Feature / historical snapshot records
  |- Investigations, proposals, tickets, audit events, guest sessions

Backend -> one hosted LLM API (server-side credentials only)
CI -> tests / build -> deployed frontend + backend + database
```

**One backend process/service.** No standalone model server or agent microservice. Run model inference within FastAPI. Persist tickets and approvals in PostgreSQL; do not rely on in-memory state surviving restarts.

### 6.1 Stack — fixed unless a blocking reason is documented

| Layer | Technology |
|---|---|
| Data/ML language | Python |
| Processing | Pandas, SQL |
| Storage | PostgreSQL |
| Modeling | scikit-learn, XGBoost |
| API/validation | FastAPI, Pydantic |
| DB layer | SQLAlchemy and Alembic where warranted |
| Agent | LangGraph, one hosted LLM provider |
| Policy | Versioned markdown/JSON sections and deterministic lookup |
| Frontend | React, TypeScript, Vite |
| Tests | Pytest; focused Playwright end-to-end suite |
| Packaging | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Cloud | One provider/configuration chosen after an early deployability check |

**No GPU is required.** Offline training should start on CPU. The public backend calls a hosted LLM; an institutional GPU is not assumed available for internet serving.

### 6.2 Contracts first

Keep the endpoint inventory narrow; document Pydantic schemas and error responses before wiring the frontend:

| Endpoint (illustrative) | Role |
|---|---|
| `GET /api/v1/health` | Liveness, plus separate dependency/readiness checks if needed |
| `GET /api/v1/snapshots` | Available held-out historical snapshots |
| `GET /api/v1/orders?snapshot_id=...` | Eligible as-of order list, pagination/filtering |
| `GET /api/v1/orders/{order_id}?snapshot_id=...` | Approved as-of order details |
| `POST /api/v1/predictions` | Validated inference; includes model version and prediction-as-of timestamp |
| `POST /api/v1/investigations` | Start bounded investigation for selected order/snapshot |
| `GET /api/v1/investigations/{id}` | Structured report or status |
| `POST /api/v1/proposals/{id}/approve` | Server-validated approval and idempotent ticket creation |
| `POST /api/v1/proposals/{id}/reject` | Persist rejection without ticket creation |
| `GET /api/v1/tickets` | Current guest's tickets |

Actual status codes, token/session implementation, synchronous/asynchronous investigation handling and response structures are locked in `docs/api_contracts.md` during Stage 0. Do not let frontend and backend agents silently invent different contracts.

### 6.3 Data boundaries

- Source Olist data is treated as immutable in the application path.
- Derived features/snapshots are versioned and separated from source tables.
- Final labels/outcomes are unavailable to as-of endpoints and agent tools.
- Guest-specific investigations/tickets/audit events are scoped by server-validated session identity.
- The backend alone has LLM credentials and the right to perform state-changing writes.

---

## 7. Agent investigation specification

### 7.1 One tightly scoped job

Investigate **one selected eligible order** and produce a structured, evidence-backed recommendation. This is not a general chat assistant and cannot run arbitrary SQL or browse the public web.

**Tool inventory:**

| Tool | Return contract |
|---|---|
| `get_order_details` | Approved as-of fields and source/order ID |
| `get_delivery_prediction` | Actual served model result, prediction time, model version |
| `get_historical_context` | SQL-computed comparison from records known by the relevant as-of cutoff, sample count and caveats |
| `get_demo_policy` | Versioned demonstration policy section ID, exact relevant passage |

Investigation order can be partially deterministic. Use the LLM for supported synthesis/decision where needed; deterministic code gathers factual tool inputs and validates outputs. Avoid paying for repeated LLM calls that do not improve the task.

### 7.2 Response schema

Investigation includes:

```text
investigation_id
order_id
snapshot_id
prediction_as_of
model_version
risk_probability          # copied from validated model output
facts[]                   # referenced to tool/result identifiers
evidence[]                # source identifier + value/policy section
limitations[]             # missing, stale or insufficient evidence
recommendation            # investigate / propose escalation / no escalation
proposal_id | null
status                    # completed / insufficient_evidence / failed
```

The backend checks cited evidence IDs and quantitative values against tool results. If retrieval fails, the agent must identify the missing evidence, not fabricate it.

### 7.3 Approval and ticket integrity

- Agent generates a **proposal**, not a ticket or approval.
- Backend checks proposal/session ownership, pending state, required evidence and order eligibility.
- Approval is an explicit user request. A database transaction transitions proposal → approved and creates the ticket/audit event.
- Apply a uniqueness/idempotency constraint so duplicate clicks/retries create **one** ticket.
- Rejection transitions to rejected and creates no ticket.
- Invalid/expired/cross-session proposal IDs are denied.
- A backend restart cannot turn a pending proposal into an approved action.

Choose a straightforward persisted proposal workflow; do not add a second graph-level persistence mechanism unless the flow requires interrupt/resume semantics that cannot be handled by backend records.

### 7.4 Failure policy

Tool/LLM failures must not break the risk queue or create a ticket. Use bounded timeout and retries with a request-level cost/token limit. On failure, show “Investigation unavailable; prediction and order data remain accessible.” Do not auto-retry irreversible approval writes without idempotency.

---

## 8. Security, reliability and operating cost

### 8.1 Guest demo access

Use a simple, bounded guest-session design: server-issued session identifier in a secure cookie or equivalent authenticated-by-server mechanism, expiry, session-specific ticket queries and no public administration interface. Historical order records may be shared across guests; mutable actions must not be.

At minimum: request validation, approved tool allowlist, parameterized read-only queries, server-side authorization on writes, safe CORS, no secrets in frontend, rate limiting, and an LLM usage quota. Treat policy text and database text as untrusted data, not instructions that can change agent privileges.

### 8.2 Budget controls

- Development: local PostgreSQL, app, training and tests where possible.
- Production: budget for one hosted backend, one managed DB or equivalent, static frontend and a small LLM API allowance.
- Set cloud spending alerts and a provider/app-side LLM budget; store secrets in the hosting platform's secret manager/environment configuration.
- Cap investigations per guest/time window; return a clear limit message rather than silently incurring cost.
- Deploy only after verifying free-tier sleep/storage rules, expected uptime and monthly costs; a usable interview demo may justify an inexpensive always-on plan.
- An existing coding-agent subscription **does not** imply free runtime LLM API calls in the deployed product.

### 8.3 Minimum observability

Structured backend logs with request/investigation IDs, model version, status, response duration and sanitized tool events. Record LLM usage/cost where provider permits. Do not log API keys, full session secrets, or unnecessary personal information. Add health/readiness checks and error aggregation possible within the selected hosting service.

---

## 9. Verification plan: success AND failure modes

### 9.1 Severity

- **P0 / release blocker:** leaked future data, fabricated risk, unauthorized/cross-guest write, real external action, secrets exposed, corrupt tickets, public critical path broken.
- **P1 / release blocker:** main investigation frequently fails, bad API contract, repeat deployment failure, model artifact missing, critical error not surfaced.
- **P2 / document or fix:** minor responsive-layout issue, noncritical slow path, explanatory copy or cosmetic problem.

**Every P0/P1 must be closed or the feature must be disabled and removed from the release promise.** No amount of passing superficial tests cancels a P0.

### 9.2 Test matrix

| Area | Success case / assertion | Failure injection and required behavior |
|---|---|---|
| Ingestion | Raw download creates expected table counts; rerun is idempotent | Missing file/column → actionable validation error, no partial silent success |
| Order joins | Exactly one derived prediction row per eligible order | Multiple items/payments → no duplicated order score |
| Chronology | Carrier handover precedes outcome for eligible data | Null/inconsistent dates → excluded and counted, never silently imputed as outcomes |
| Feature availability | Every model feature has valid as-of timestamp | Deliberately insert future feature → leakage test fails build |
| ML split | Later test period excluded from fit/tuning | Training cutoff violation → automated assertion fails |
| Model serving | Same input yields same versioned result as offline pipeline | Missing artifact/schema mismatch → controlled unavailable response, not synthetic value |
| Risk queue | API filters/paginates authentic eligible snapshot records | No rows/invalid snapshot → clear empty/validation state |
| Investigation | Correct tools called; numbers and policy IDs match results | Tool timeout/invalid JSON/missing policy → honest failure or insufficient-evidence state |
| Future truth | Agent sees only as-of order DTO | Try to request final delivery/review through tools → denied/unavailable |
| Approval | Valid explicit approval creates one ticket + audit | Direct write without approval → denied, no ticket |
| Rejection | Proposal rejected, no ticket | Approve rejected proposal → denied |
| Idempotency | Repeat approval returns existing ticket or safe no-op | Double-click/concurrent requests → one ticket only |
| Isolation | Guest A sees only A's tickets | Guest B guessing A's IDs → forbidden/not found |
| Recovery | Restart preserves committed tickets/pending proposals | Restart mid-investigation → recover or mark failed honestly, never auto-approve |
| Availability | Dashboard/inference work without LLM | Disable provider → dashboard works; investigate shows actionable error |
| Security | No API key shipped to browser or repo | Secret scan / inspect production bundle → no secret |
| Costs | Guest requests stop at configured budget/rate limits | Exceed quota → controlled 429/limit response |
| Deployment | Public HTTPS path completes full journey | Broken environment variable/DB connection → health fails, deploy blocked |

### 9.3 Quantitative release targets

These are **design acceptance targets**, not claims that the project has already achieved them. Specify workload and hardware in the final report.

| Criterion | Release target / rule |
|---|---|
| Deterministic critical-path tests | 100% pass in CI and against deployed release |
| Data integrity/leakage | Zero known leakage/duplicate-grain violations in audit and adversarial fixtures |
| Unauthorized/duplicate tickets | Zero in the defined security and concurrency suite |
| Functional guest path | Three repeated full journeys on deployed URL, including approve and reject |
| Agent evaluation | Target ≥85% valid task completion on a held-out set of ~40–50 curated scenarios; report exact denominator and per-failure class |
| Claim support | Zero unsupported model numbers or nonexistent policy IDs on the critical evaluation cases; report any failures elsewhere |
| Non-LLM API response | Measure median and p95 under a documented modest load; initial p95 target ≤2 s, excluding cold starts |
| Investigation time | Measure median and p95; initial warm-system p95 target ≤30 s, with meaningful timeout/error handling |
| Operating cost | Record observed cost per successful investigation and budget caps; no unbounded public usage |
| ML performance | Publish actual test metrics and honest rule-baseline comparison; no arbitrary accuracy promise |

If a quantitative target proves impractical on chosen free hosting or LLM, either improve the implementation, provision appropriately, or document a narrowly revised target **before release approval**. Never silently mark a failed target as passed.

### 9.4 Agent benchmark composition

Prepare approximately **40–50 cases** spanning ordinary investigations, high/low risk, missing order, insufficient context, stale/missing policy, tool failure, invalid tool payload, ambiguous request, prompt injection, unauthorized action and duplicate proposal. Separate development cases from held-out release cases. Evaluate actual tool traces plus final response, not just plausible prose. Record completion, supported-claim rate, tool correctness, approval compliance, duration and request cost.

---

## 10. Development sequence and gates

The agents build the project; a stage passes through **running evidence**, not a status sentence. Implement tests during each stage. Commit at green checkpoints.

| Stage | Required work | Observable demo / artifact | PASS gate |
|---|---|---|---|
| **0. Bootstrap** | Repo, pinned dependencies, environment template, schema/API contracts, Docker Compose, minimal CI | React and FastAPI respond; DB health works | Fresh local environment boots; initial CI green |
| **1. Data + ML** | Acquire/audit data, SQL ingestion, splits, baselines, chosen model, saved artifacts | Real held-out order scores from CLI/API test | Sections 3–4 audits and parity tests pass |
| **2. Vertical slice** | Orders/prediction APIs, snapshot query, risk-queue UI | Browser displays authentic snapshot/model scores end-to-end | API/UI contract and risk-queue smoke tests pass |
| **3. AI + action** | Tools, bounded graph, evidence report, proposals, approval/rejection, tickets | Entire core journey works locally | Positive, failure and authorization tests pass |
| **4. Cloud + hardening** | Early deploy followed by CI/CD, Playwright, security/failure tests, cost limits, logs, docs and video | External user completes full public HTTPS flow | All P0/P1 closed, release checklist signed off |

**Deploy an early vertical slice during Stage 2**, so cloud incompatibilities are discovered before final polish. Stage 4 completes operational hardening and release verification; it is not the first time deployment is attempted.

### 10.1 Time budget and owner's role

**Working plan:** approximately **10–20 elapsed days of intensive agent-driven execution** for a verified public release, with contingency if dataset issues, usage caps, API billing, cloud accounts or integration failures intervene. A functional prototype can appear earlier; no exact completion time is guaranteed. These estimates assume a prepared environment and sufficiently available Claude Code/Codex sessions.

Owner involvement is limited primarily to dataset/API/cloud access, spending authorization, milestone approval and final hands-on public demo testing. No owner-written application code is required. The owner must still learn the actual implementation and measured results before presenting it in interviews.

### 10.2 Agent collaboration protocol

**Claude Code: implementation lead.** Work a bounded task, write tests, run them, fix failures, commit and update progress.  
**Codex: independent reviewer.** Inspect data leakage, API contracts, agent/tool trust boundaries, security and deployment; run critical checks rather than only reading code.

Use distinct branches/worktrees if working concurrently. Do not allow overlapping writes to the same files. Never grant coding agents unrestricted cloud spend or access to unrelated personal files. Store all important decisions in the repository so sessions can resume without reliance on chat context.

Suggested control files:

```text
Project_Plan.md
CLAUDE.md
AGENTS.md
TASKS.md
PROGRESS.md
docs/api_contracts.md
docs/data_audit.md
docs/model_report.md
docs/agent_evaluation.md
docs/architecture.md
docs/deployment.md
```

The first agent task is to produce an executable Stage 0 checklist and review the dataset schema. It must not replace this plan with a larger architecture.

---

## 11. Repository and delivery artifacts

```text
opspilot/
├── Project_Plan.md
├── README.md
├── CLAUDE.md
├── AGENTS.md
├── TASKS.md
├── PROGRESS.md
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── db/
│   │   ├── ml/
│   │   ├── agent/
│   │   ├── services/
│   │   └── schemas/
│   └── tests/
├── frontend/
│   └── src/
├── data_pipeline/
├── ml_pipeline/
├── evaluation/
│   ├── ml/
│   └── agent/
├── docs/
├── infra/
├── docker-compose.yml
├── .env.example
└── .github/workflows/
```

Do not check in API keys, cloud credentials, personal session data or unnecessary raw datasets. The README must include a non-commercial dataset attribution, architecture diagram, feature overview, exact local run instructions, public demo link, model/agent evaluation summary, limitations and screenshots.

**Final handoff package:**

- Working HTTPS application and short recorded walkthrough.
- Public source repository and reproducible local run.
- Data audit and model evaluation with actual numbers.
- Agent evaluation including failed cases.
- Documented API, architecture, access controls and cloud configuration.
- Automated test/CI evidence, deployed smoke-test evidence and measured cost/latency.
- A concise 2–3-bullet project entry that cites only demonstrated work and measured results.

---

## 12. Final definition of done — release checklist

OpsPilot is finished only when **all** boxes can truthfully be checked:

- [ ] Public HTTPS URL works without the owner's laptop/GPU running.
- [ ] Guest can complete Risk Queue → Investigation → Approve/Reject → Ticket without developer help.
- [ ] Historical data, live computation and simulated actions are plainly distinguished.
- [ ] Data audit, historical point-in-time rules and chronological evaluation pass.
- [ ] No final outcomes are exposed to model features or as-of agent tools.
- [ ] Real versioned model artifact serves actual risk scores; no placeholders.
- [ ] Agent report has valid tool-grounded quantities and policy citations, or declares insufficient evidence.
- [ ] Server enforces approval, session isolation and idempotency.
- [ ] Core functionality fails safely when LLM or database access is unavailable.
- [ ] CI green; production browser smoke tests green; no open P0/P1 failures.
- [ ] Rate/cost caps, secrets handling and actual deployment health are verified.
- [ ] Docs, model report, agent benchmark and demo video match the deployed software.

**Project completion standard:** Not “the agents produced code,” but “a stranger can use the product, its behavior is measured, predictable failures are handled, and every material claim can be shown in the running application or its reports.”

---

## 13. Source references for implementation

- Olist dataset and published license: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>
- FastAPI documentation: <https://fastapi.tiangolo.com/>
- PostgreSQL documentation: <https://www.postgresql.org/docs/>
- XGBoost documentation: <https://xgboost.readthedocs.io/>
- LangGraph documentation: <https://docs.langchain.com/oss/python/langgraph/overview>
- Docker Compose documentation: <https://docs.docker.com/compose/>
- GitHub Actions documentation: <https://docs.github.com/en/actions>

> **Execution rule:** No new feature without a product requirement, no “done” without a test or observable artifact, and no public release with an unresolved P0/P1 failure.

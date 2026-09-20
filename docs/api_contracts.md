# OpsPilot — API Contracts

Base path: `/api/v1`. Interactive schema: `/docs`. Machine-readable:
`/openapi.json`.

Two rules govern everything below.

1. **No as-of response carries a delivery outcome.** Order responses are built
   from `order_features`, a table with no outcome column, so the guarantee is
   structural rather than a matter of remembering to exclude fields.
2. **Mutating endpoints are scoped to the caller's guest session**, which the
   server issues and the client cannot choose.

---

## Errors

Every failure uses one envelope, so the frontend can render something
actionable rather than a status code.

```json
{
  "error": {
    "code": "not_found",
    "message": "Unknown snapshot 'x'.",
    "detail": { "available": ["2018-06-20", "2018-07-18", "2018-08-15"] }
  }
}
```

| HTTP | `code` | Meaning |
|---|---|---|
| 400 / 422 | `invalid_request` | Failed validation. |
| 403 | `forbidden` | Not permitted for this session. |
| 404 | `not_found` | Absent, **or** present but owned by another session. |
| 409 | `conflict` | Illegal state transition (approving a rejected proposal). |
| 429 | `rate_limited` | Request or AI-usage cap reached. `detail.limit_type` says which. |
| 503 | `service_unavailable` | A dependency is down. Never a substituted value. |
| 500 | `internal_error` | Unexpected. No internal detail is leaked. |

A cross-session identifier returns **404, not 403**, so a response cannot be
used to confirm that an id exists.

---

## Sessions

The first response sets `opspilot_session`, an `HttpOnly` cookie holding a
256-bit server-generated token.

| Property | Value | Why |
|---|---|---|
| `HttpOnly` | yes | Script cannot read it, so XSS cannot exfiltrate it. |
| `Secure` | production only | Required with `SameSite=None`. |
| `SameSite` | `none` in production, `lax` locally | API and frontend are different origins in production. |
| Lifetime | 72 h | Configurable via `SESSION_TTL_HOURS`. |

An unknown or expired token yields a **new** session rather than an error, so
a guessed value grants nothing. All browser requests use
`credentials: "include"`; `CORS_ORIGINS` must therefore be an exact origin,
never `*`.

---

## Endpoints

### `GET /health`
Liveness. `{"status": "ok", "time": "..."}`. Never touches the database.

### `GET /health/ready`
Per-dependency readiness.

```json
{
  "status": "degraded",
  "checks": {
    "database": {"ok": true},
    "model": {"ok": true, "model_version": "xgboost-20260919"},
    "policy": {"ok": true, "version": "demo-policy-v1"},
    "llm": {"ok": false, "configured": false}
  }
}
```

`ok` — everything works. `degraded` — critical paths fine, only the LLM is
missing, so the queue and predictions still work. `unavailable` — the database
or the model is down.

### `GET /meta`
Model version, training cutoff, policy version, whether an LLM is configured,
and the dataset attribution the UI displays. Contains no secret.

---

### `GET /snapshots`
The frozen historical dates, each with `orders_in_transit`,
`orders_pre_deadline` and `orders_overdue`.

### `GET /snapshots/{snapshot_id}/stats`
SQL-computed risk distribution for the queue header: `high_risk`,
`medium_risk`, `low_risk`, `mean_risk`, `model_version`.

### `GET /orders`
The risk queue.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `snapshot_id` | string | required | |
| `limit` | int 1–200 | 50 | |
| `offset` | int ≥ 0 | 0 | |
| `sort` | `risk` \| `deadline` \| `handover` | `risk` | Tie-broken on `order_id`, so pagination is stable. |
| `risk_band` | `low` \| `medium` \| `high` | — | |
| `include_overdue` | bool | `false` | Past-deadline orders are excluded by default. |
| `customer_state` | string(2) | — | |

Returns `{items, total, limit, offset}`. Each item carries
`risk_probability`, `risk_band` and `model_version`, so no number is shown
unattributed.

### `GET /orders/{order_id}?snapshot_id=…`
The as-of DTO. Includes `prediction_as_of` — the carrier-handover moment the
score refers to — and never a delivery outcome. 404 if the order is not in
that snapshot.

### `POST /predictions`
```json
{"order_id": "…", "snapshot_id": "2018-08-15"}
```
Re-serves the model live from the stored point-in-time feature document.
Returns the score, band, `model_version`, `prediction_as_of`, `computed_at`
and a disclaimer.

**503 when the artifact is unavailable.** It never falls back to a stored or
synthetic value. A test asserts the response body contains no probability in
that case.

---

### `POST /investigations`
```json
{"order_id": "…", "snapshot_id": "2018-08-15"}
```

Runs the bounded workflow **synchronously** and returns the finished
`InvestigationOut`. Synchronous is deliberate: the graph makes exactly one LLM
call with a hard timeout, so a job queue would add moving parts without
shortening the wait.

Order of operations, which matters for cost:

1. Validate the order exists in the snapshot — 404 before any spend.
2. Check the budget — 429 before any spend.
3. Run the graph.

Response `status` is one of:

| Status | Meaning | Proposal? |
|---|---|---|
| `completed` | A verified report was produced. | Only if escalation is recommended **and** the policy permits it. |
| `insufficient_evidence` | Required evidence could not be retrieved (policy EVI-01). | Never. |
| `failed` | A tool or the provider failed. | Never. |

The body carries `facts[]` (each with the `evidence_ids` supporting it),
`evidence[]` (what the tools actually returned), `limitations[]`,
`recommendation`, `proposal`, `ticket_id` and `duration_ms`.

### `GET /investigations/{id}` / `GET /investigations`
This session's investigations only. Another session's id returns 404.

---

### `POST /proposals/{id}/approve`
The only path that creates a ticket. Requires an explicit request from the
owning session.

```json
{"proposal": {...}, "ticket": {...}, "already_decided": false}
```

- Idempotent: `tickets.proposal_id` is `UNIQUE`, so a double click or a
  concurrent retry returns the **existing** ticket with `already_decided: true`.
- 409 if the proposal was rejected.
- 409 if the investigation behind it did not complete.
- 404 for another session's proposal.

### `POST /proposals/{id}/reject`
Transitions to `rejected` and creates no ticket, ever. Repeat calls return
`already_decided: true`. Approving afterwards is a 409.

---

### `GET /tickets` / `GET /tickets/{id}`
This session's tickets, with a disclaimer stating the actions are simulated.

### `GET /audit`
This session's append-only action history: `investigation_completed`,
`proposal_created`, `proposal_approved`, `proposal_rejected`.

---

## Rate and budget limits

| Limit | Default | Scope |
|---|---|---|
| `API_REQUESTS_PER_MINUTE` | 120 | Per client IP, in-process sliding window. |
| `INVESTIGATIONS_PER_SESSION_PER_DAY` | 10 | Per guest, counted in PostgreSQL so a restart cannot reset it. |
| `INVESTIGATIONS_GLOBAL_PER_HOUR` | 60 | All visitors. |
| `INVESTIGATIONS_GLOBAL_PER_DAY` | 180 | All visitors. |

All are checked **before** the provider call, so exceeding one costs a 429
rather than quota.

---

## What is deliberately absent

No admin endpoints, no user accounts, no arbitrary SQL or text-to-SQL, no
endpoint that returns a delivery outcome, and no endpoint that creates a ticket
without an explicit approval request.

## Situations

### `GET /api/v1/snapshots/{snapshot_id}/situations`

Lanes of a snapshot that carry a cluster of flagged orders, ranked by
`expected_late` descending.

Query: `limit` (1-100, default 20), `min_orders` (2-100, default 3).

```json
[{
  "situation_id": "2018-08-15__SP-RJ",
  "snapshot_id": "2018-08-15",
  "seller_state": "SP", "customer_state": "RJ", "lane": "SP to RJ",
  "n_flagged": 107, "n_high": 45, "n_lane_total": 132,
  "share_of_lane": 0.8106,
  "expected_late": 19.966,
  "mean_risk": 0.1866, "max_risk": 0.4943,
  "n_escalatable": 6,
  "model_version": "xgboost-20260920"
}]
```

`expected_late` is the sum of the member orders' calibrated probabilities.
`n_escalatable` counts members that independently satisfy ESC-01; ESC-05
permits a lane escalation at three or more.

No field here is outcome-derived.

### `GET /api/v1/situations/{situation_id}`

The same fields plus `members[]` (each with `risk_probability`, `risk_band`,
`days_to_deadline`, `escalatable`) and `lane_history`, which comes from the
as-of analytics path and always carries its caveats.

`404` for a malformed id, an unknown snapshot, or a lane with no flagged
orders. Ids are validated against `YYYY-MM-DD__XX-YY` before use.

### `POST /api/v1/situations/{situation_id}/investigations`

Query: `mode` = `llm` (default) or `deterministic`.

Returns the same `InvestigationOut` contract as an order investigation, with
`subject_type: "situation"`, a `situation` block, and `generated_by` set to
`"model"` or `"deterministic"`.

`mode=llm` draws on the session investigation budget and is counted when the
run **starts**, since a failed run still consumed provider quota.
`mode=deterministic` costs no quota and never calls a provider. The LLM path
falls back to the deterministic brief on any provider failure and reports
`generated_by: "deterministic"` when it does.

A proposal appears only when the run completed, recommended escalation, and
the backend's own ESC-05 reading permits it. Its `member_order_ids` are the
orders the escalation would cover, and they are carried onto the ticket.


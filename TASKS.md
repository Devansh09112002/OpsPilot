# OpsPilot — Task Checklist

Ticked only when run and inspected, not when written.

## Stage 0 — Bootstrap
- [x] Repository structure, pinned dependencies, `.env.example`
- [x] PostgreSQL 16 + Alembic migrations
- [x] API contracts documented (`docs/api_contracts.md`)
- [x] GitHub Actions CI workflow
- [x] Docker + Compose configuration *(compose unverified: no Docker on the dev machine)*

## Stage 1 — Data and ML
- [x] Reproducible, checksum-verified dataset fetch
- [x] Data audit gate passes (`docs/data_audit.md`)
- [x] Point-in-time feature table, one row per eligible order
- [x] Every feature carries an availability justification, enforced at build time
- [x] Chronological splits frozen before model comparison
- [x] Rule baseline, logistic regression, XGBoost compared on validation only
- [x] Model frozen, then scored once on the held-out test split
- [x] Artifact saved with checksum and feature schema
- [x] Training-serving parity tests

## Stage 2 — Vertical slice
- [x] Orders, snapshots and prediction endpoints
- [x] Ingest writes scores using the same artifact the API serves
- [x] Risk queue, order detail and tickets screens
- [x] Loading, empty, error and unavailable states throughout
- [x] Browser shows real snapshot data and real model scores

## Stage 3 — AI and action
- [x] Four allowlisted tools over real data
- [x] Bounded acyclic LangGraph workflow, exactly one LLM call
- [x] Evidence verification gate (ids, policy sections, recommendation ceiling)
- [x] Proposal → explicit approval → ticket, with audit events
- [x] Idempotency under double-click and concurrency
- [x] Session isolation on every mutable read and write
- [x] Real Gemini call verified end to end
- [x] 59-case agent benchmark

## Stage 4 — Cloud and hardening
- [x] Free-tier sizing measured (53 MB database, ~340 MB RSS)
- [x] Cold-start and paused-database handling in the UI
- [x] Rate and budget caps enforced before any provider call
- [x] Secret scan in CI; no key reaches the browser bundle
- [x] Keepalive workflow
- [x] Provisioning automation (`infra/provision.py`)
- [x] README, architecture, deployment, API docs
- [x] Supabase project created and seeded (95,952 + 95,952 + 3 + 3,955 rows)
- [x] Render services deployed on the free plan
- [x] E2E suite green against the deployed URLs
- [x] Deployed smoke test green (37/37)
- [x] Public URL recorded in the README
- [x] Keepalive configured against the live API
- [x] **Gemini key rotated** (21 September 2026) — verified on the public URL before
      the original was deleted, and re-verified after
- [ ] Repository made public (optional; currently private by choice)

## Stage 5 — Lane situations (v2)
- [x] Premises measured before building: seller risk does **not** persist
      (rho +0.105, p 0.13), so no seller leaderboard; lane concentration does
- [x] Situation grouping, ranked by expected late orders
- [x] As-of lane history, with its weak-persistence caveat stated
- [x] ESC-05 composes ESC-01; one risk threshold in the system
- [x] Situation investigation graph, sharing the verification gate
- [x] Membership grounding: a report may not cite a foreign order
- [x] Deterministic briefs, zero provider calls
- [x] One escalation covering N orders, member ids recorded on the ticket
- [x] One open escalation per subject enforced at approval
- [x] Situations and lane-detail screens, keyboard reachable
- [x] 33 situation tests; 11 situation E2E journeys
- [x] Agent benchmark extended (68/69, held-out 36/36)
- [x] Migration applied to the deployed database ahead of the code
- [x] Verified on the public URL (37/37 smoke, 20 browser journeys)

## Release blockers still open
- None. The Gemini key rotation, the last blocker, is complete and verified.
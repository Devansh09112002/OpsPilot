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
- [x] E2E suite green against the deployed URLs (12/12)
- [x] Deployed smoke test green (37/37)
- [x] Public URL recorded in the README
- [x] Keepalive configured against the live API
- [ ] **Gemini key rotated** — it came through a chat transcript
- [ ] Repository made public (optional; currently private by choice)

## Release blockers still open
- **Rotate the Gemini API key** before sharing the demo. Create a new key,
  set it on the `opspilot-api` Render service, delete the old one.

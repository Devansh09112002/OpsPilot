# OpsPilot — Final Release Report

**Date:** 21 September 2026
**Release tag:** `v2.1-quality` (rollback points: `v2.0-verified`, `v1.0-baseline`)
**Live application:** <https://opspilot-web-hj6k.onrender.com>
**API:** <https://opspilot-api-pg66.onrender.com>

---

## 1. Application status

Complete and running. A visitor can open the public site and finish both
journeys end to end:

**Order-level** — risk queue → open an order → investigate → approve or
reject → persistent ticket.

**Lane-level** — situations → open a lane → investigate (AI or deterministic)
→ approve once for every order on the lane → ticket recording the orders it
covers.

Order records are real, anonymised Olist marketplace data (2016–2018). Risk
scoring, investigation, approval and ticketing execute live. Escalations are
simulated: no courier, seller or customer is contacted, and the application
says so on screen.

---

## 2. Verification results

All figures below were produced by running the thing described, on the dates
of this release.

| Check | Result |
|---|---|
| Backend test suite | **222 passed**, 0 skipped, stable across repeated runs |
| Mutation audit | **24 of 24 mutations caught** (was 17/24 before this cycle) |
| Hostile-input probe | **318 GET + 14 POST probes, zero 5xx** |
| Browser suite, local | **33 passed**, 1 correctly skipped |
| Browser suite, **against the deployed site** | **31 passed**, 1 correctly skipped |
| Accessibility (axe-core, WCAG 2.1 A + AA) | **0 violations** on every screen, local and deployed |
| Deployed smoke test (`infra.verify_deployment`) | **37 / 37** |
| Docker Compose stack | **Verified in CI** — see §5 |
| CI | 4 jobs green: backend, frontend/E2E, compose, secret scan |
| Lint (`ruff`) | clean |
| Secret scan | 160 tracked files, 0 findings |

The single skipped browser test asserts the no-LLM failure path, which does
not apply when a provider key is configured. It is skipped honestly rather
than passed vacuously.

---

## 3. ML and agent evaluation

### Model — [`docs/model_report.md`](docs/model_report.md)

Ranking one snapshot at a time, as the deployed queue does:

| Model | Precision@50 | Lift | Recall@50 |
|---|---|---|---|
| Operational rule (deadline proximity) | 11.6% | 2.44× | 10.7% |
| Logistic regression | 16.7% | 3.51× | 16.0% |
| **XGBoost (served)** | **13.5%** | **2.82×** | **12.2%** |

Two things the report states rather than hides:

- **XGBoost was selected on validation and logistic regression edged it out on
  the held-out test period** by +0.033, inside the 0.030 standard error. The
  selection was not revisited after seeing test results, because doing so
  would make the test estimate meaningless. The honest reading is that the two
  learned models are not clearly distinguishable on this data, and both beat
  the operational rule.
- Calibration improves the Brier score **9.6×** (0.28485 → 0.02952). Isotonic
  is monotonic, so the ranking is unchanged.

### Situation-level calibration — [`docs/snapshot_calibration.md`](docs/snapshot_calibration.md)

Measured during this cycle, and the most significant ML finding in the
project:

| Snapshot | Predicted late | Actually late | Ratio |
|---|---|---|---|
| 2018-06-20 | 15.1 | 4 | 3.77× |
| 2018-07-18 | 18.5 | 13 | 1.42× |
| 2018-08-15 | 59.7 | 45 | 1.33× |
| **All** | **93.3** | **62** | **1.50×** |

A lane's summed risk **overstates** the number of orders that were actually
late by about half again. The calibrator is fitted on validation at a 10.8%
late rate and applied to a test period at 3.0%. It was **not** corrected by
refitting, which would be selection on the test set; it was measured,
published, and the product wording changed everywhere it appeared — UI, tool
payload, API schema, agent prompt, deterministic brief. The quantity is now
called *risk load*, and what survives the bias, ranking lanes against each
other (rank correlation 0.46), is what the product uses it for.

### Agent — [`docs/agent_evaluation.md`](docs/agent_evaluation.md)

69 cases from real snapshot contents, split development / held-out.

| Metric | Result |
|---|---|
| Overall | 68 / 69 |
| **Held-out** | **35 / 36 (97.2%)** against a pre-registered 85% target |
| Claim support | 100% |
| Policy-citation validity | 100% |
| Approval compliance | 100% |
| Outcome containment | 100% |
| Membership grounding | 100% |

The one miss is a held-out case whose provider call did not return. It is
reported as measured rather than absorbed into the skip rule, and the harness
now records the failure reason so a provider outage and an agent defect are
distinguishable next time.

---

## 4. Security status

| Item | Status |
|---|---|
| Live credentials in git history | **None.** Every configured credential was compared byte-for-byte against all 745 objects in the repository's history; none appears in any commit. |
| Working tree | 160 tracked files, 0 scanner findings |
| `.env` | git-ignored and untracked; `.env.example` carries placeholders only |
| Frontend bundle | No credential-shaped string; asserted in CI and in the compose job |
| Deployed configuration | All secrets on the API service only. The static site carries one variable: the public API URL. |
| Session cookie | `HttpOnly; Secure; SameSite=none`, verified on the live response |
| CI | No hardcoded secrets |
| Error responses | Generic message plus a request id; no statement, column, parameter or stack trace escapes |

**Fixed this cycle:** `cookie_secure` was not derived from the environment, so
the deployed cookie was secure only because two variables happened to be set
on Render — and forgetting `SameSite=none` breaks every cross-origin session
silently. Production now hardens itself by default.

### Outstanding: Gemini API key rotation

The current key was supplied through a chat transcript and must be treated as
exposed. **It is not in the repository or its history** — the exposure is the
transcript alone.

`infra/rotate_gemini_key.py` performs the rotation and verifies it *before*
the old key is revoked: it checks the new key against the provider, writes it
to `.env` and the Render service, waits for the redeploy, then runs a real
investigation on the public URL and requires `generated_by=model`. A
deterministic fallback counts as failure. Every failure path stops without
revoking anything.

### Tokens that can be revoked after release

| Credential | Recommendation |
|---|---|
| `SUPABASE_ACCESS_TOKEN` | **Revoke after release.** A management token with full account access, needed only for provisioning, seeding and migrations. Reissue when needed. |
| `RENDER_API_KEY` | **Revoke after the key rotation.** Full account access, needed only for deploys and the rotation script. |
| `SUPABASE_DB_PASSWORD` | Keep — the running application needs it. It exists only in the Render service environment and the local `.env`. |
| `GEMINI_API_KEY` | Rotate now; see above. |

---

## 5. Reproducibility

**`docker compose up` is verified**, in CI, which has Docker the development
machine cannot install. The job builds the images, starts database, API and
web, runs the ingest profile, and then checks:

```
api is up after 2 attempts
health: degraded - database, model, policy ok; no LLM key
snapshots: ['2018-06-20', '2018-07-18', '2018-08-15']
situations: 5 lanes, top = SP to RJ
investigation: completed, 4 cited facts
approval: ticket tkt_b3ad58ae86f047b99500 covering 107 orders
frontend: index and deep link both 200
bundle: clean
```

`degraded` is correct: CI configures no provider key, and readiness says so
instead of claiming health. The whole journey completes without one, through
the deterministic path.

CI also runs the full pipeline from scratch on every push — dataset fetch with
checksum verification, the data audit gate, model training, migrations and
ingest — so the artefacts are reproducible from source, not just present.

---

## 6. Known limitations

| Limitation | Severity |
|---|---|
| Risk load overstates the actual late count by ~1.5× | Low — measured, published, and no surface claims otherwise; a test enforces the wording |
| The served model beats the baselines only marginally, and logistic regression edged it out on test | Medium for the model, low for the product; stated in the report and the README |
| One held-out agent case failed on a provider call | Low — 97.2% against an 85% target |
| Trained on 2016–2018 Brazilian marketplace data | Inherent; does not transfer without retraining |
| No user accounts; ticket history tied to a guest cookie | By design for a public demo |
| Free-tier hosting sleeps when idle | First visit after a quiet period takes up to a minute; the UI says so rather than showing a broken page |
| Free provider tier allows ~100 investigations/day | Mitigated: every situation can be briefed deterministically with no provider call |
| Accessibility verified by axe-core, not by assistive-technology users | Low — automated coverage is thorough but not equivalent to manual testing |

---

## 7. Actions that require you

1. **Rotate the Gemini key.** In your terminal:
   ```
   cd E:\OpsPilot
   .venv\Scripts\python.exe -m infra.rotate_gemini_key
   ```
   Create a key at <https://aistudio.google.com/apikey>, paste it at the hidden
   prompt. The script does the rest and tells you when it is safe to delete
   the old key. Delete it only after it says so.

2. **Decide on repository visibility.** The repository is ready to be public
   and has not been made public. That is your call.

3. **Optional: revoke the Supabase and Render tokens** once no further
   provisioning or deploys are planned (see §4).

4. **Optional: record the demo walkthrough** — the project plan asks for one
   and it cannot be produced here.

---

## 8. Readiness for public release

**Ready, subject to the key rotation in §7.1.**

The application is complete, deployed and verified on its public URL. The
repository is clean of credentials in both its working tree and its full
history, carries a licence that states the dataset's separate non-commercial
terms, and documents what was measured — including the two results that are
unflattering: a model that does not clearly beat logistic regression, and a
headline product number that overstates reality by half again.

Nothing in the documentation claims more than was verified. Where something
is unverified, it is labelled unverified.

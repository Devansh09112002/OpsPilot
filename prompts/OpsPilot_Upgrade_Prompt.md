# OpsPilot — Autonomous Product & Engineering Upgrade Brief

> **For Claude Code (primary implementing agent).** Save this file as `prompts/OpsPilot_Upgrade_Prompt.md` in the OpsPilot repository and provide it as the instruction for the next development cycle.

## Your assignment

Act as the lead engineer and product-minded technical owner for **OpsPilot**, an existing, publicly deployed AI/ML and agentic operations application. Your task is to take the *actual working repository* to a substantially stronger, more complete, interview-defensible industry-oriented standard.

This is **one flagship project**, not a request to generate several side projects or collect fashionable technology names. Prioritize **quality, useful product capability, correctness, reliability, and demonstrable engineering judgment** over quantity of features.

You have broad freedom to investigate, design, prioritize, implement, test, refactor, and improve the product. **This brief specifies outcomes and constraints, not a predetermined architecture or mandatory feature checklist.** Do not blindly implement a list or ask me to choose among routine technical alternatives. Study the real system, discover the highest-value opportunities, justify your choices, then execute.

The career context: I am an MS by Research student in CSE with a strong AI/ML research and evaluation background. OpsPilot should additionally demonstrate applied data science, AI/ML engineering, useful agentic systems, backend/full-stack development, deployment, and software reliability in interviews at major product companies and strong startups. It must be technically defensible, not simply visually impressive.

## Starting point — inspect, do not assume

Read the current repository, README, product/project plan, progress and release notes, data audit, model report, agent evaluation, architecture, deployment configuration, tests, CI, and implementation. Inspect the actual running deployment if accessible. Treat historical reports as claims to verify where material; the implementation and executed checks are the source of truth.

In particular, investigate the existing delivery-risk workflow, real Olist data and as-of boundaries, model-selection protocol, modest held-out ranking performance and period shift, calibrated risk displays, bounded LangGraph investigation, evidence verification, approvals, session isolation, free-tier constraints, and deployment limitations. You may discover additional issues that this brief does not anticipate.

**Do not start a fresh repo or rebuild the application unnecessarily.** Preserve what is already working. You may change the architecture when evidence shows a meaningful benefit, but do not replace good components just for novelty.

## What success looks like

A person can open the *actual public website*, understand what OpsPilot does, explore meaningful operational information, receive genuine model-backed decision support, conduct a trustworthy AI investigation, make an authorized decision, and inspect its persistent result. All user-visible claims come from real data, model output, or identified demonstration policy, with appropriate limitations.

The repository makes it clear how the application was built, how its ML and agent behavior were evaluated, why key design decisions were made, what has been tested, what can fail, how the deployed app behaves under those failures, and how another engineer could reproduce or extend it.

The upgrade should yield **material improvements to business usefulness and engineering substance**, not just more pages, longer documentation, a larger test count, or a more elaborate technology stack.

## Your design freedom and decision process

Before changing product behavior, perform a targeted technical/product audit and choose the highest-impact improvements. Consider, without treating this as a required checklist:

- Whether the current order-level workflow is sufficient for a useful operations product, or whether targeted seller/route/category analytics or pattern-level investigations would add genuine decision value.
- Whether the ML evaluation answers the business question, how selection and calibration withstand temporal shift, how uncertainty is communicated, and whether the presented scores or explanations can mislead a user.
- Whether agent behavior is genuinely useful, appropriately bounded, evidence-grounded, and robust against missing evidence, provider errors, injection, unauthorized action and quota exhaustion.
- Whether deployment, cold starts, session persistence, recovery, observability, reproducibility, cost control and CI/CD are credible for a public demonstration.
- Whether accessibility, UX, loading and error states, documentation, and the public demonstration make the system easy to understand and inspect.

You are **not required** to add MLflow, a vector database, extra agents, MCP, Redis, queues, Kubernetes, a monitoring stack, or any other named technology. Introduce a dependency only when it solves a concrete problem and its benefits justify the operational and interview-explanation cost. Conversely, do not reject a genuinely useful technology merely because it is new.

Choose a focused release scope yourself. Record briefly: the observed gap, user/engineering value, proposed intervention, rejected simpler alternatives, expected evidence, and failure risks. Then implement. Avoid a prolonged planning-only phase.

## Non-negotiable engineering boundaries

**Preserve a working release.** Record current deployed behavior and tests, commit outstanding changes, and create a reproducible baseline/tag or equivalent rollback point before substantial modification. Use a development branch/worktree as appropriate. Do not break the currently available public demo while an upgrade is incomplete. Roll back a bad release rather than leaving the site broken.

**Protect secrets and users.** The Gemini key previously supplied in conversation must be treated as exposed until rotated and the original revoked. If still outstanding, make this a priority before further public sharing. Audit the repository *and history*, artifacts, logs, frontend bundles, and CI for credentials. Never print or commit secrets. Seek my help only for account-level authorization or secure credential entry. Do not make the repo public without my explicit approval.

**Budget is ₹0 additional.** Work within authorized free-tier resources and existing coding-agent subscriptions. Never activate billing, purchase infrastructure, or incur paid API usage without explicit approval. Explain any unavoidable free-tier limitations rather than pretending they do not exist. Preserve functioning fallbacks when quotas are reached; do not evade provider restrictions.

**Maintain ML scientific integrity.** Do not repeatedly tune on the existing held-out test set, select a winner after seeing test outcomes, invent gains, or pass off simulation as production impact. If you pursue substantive model development, define a new defensible development/evaluation protocol with an untouched final assessment where possible. Review point-in-time feature and aggregate availability, data leakage, snapshot consistency, calibration, uncertainty, and meaningful business baselines. Report negative or ambiguous results accurately. Historical Olist orders are not a current live feed; simulated escalations do not contact real customers.

**Enforce authorization outside the LLM.** The model cannot be the sole approval or policy gate. Keep evidence and action claims grounded in verified tool/database results. Avoid arbitrary SQL or unrestricted writes unless there is a compelling, protected reason. Preserve session isolation, idempotency, and safe behavior under retries or partial failure.

**No false completion claims.** Implemented, tested locally, tested in CI, and verified on the deployed URL are separate statuses. Do not report any one as another. A high unit-test count does not replace genuine end-to-end validation.

## Autonomous execution

Handle coding, architectural decisions, dependency setup, debugging, evaluation, tests, deployment preparation, CI, and documentation yourself where the available environment permits. Use the cycle:

**Inspect → prioritize → implement → run → observe → fix → regression-test → verify integration → document → continue.**

Build improvements as integrated vertical slices rather than isolated notebooks or services. Make small reviewable commits; keep tests with the features. Where useful, use Codex or another independent reviewer to challenge critical assumptions, especially data leakage, model interpretation, security boundaries, and release readiness. An independent review should examine behavior and methodology, not only style or whether tests pass.

Do not stop to ask me to write code, edit `.env`, run routine commands, fix errors, select libraries, or manually configure infrastructure you can handle with authorized access. Ask only for truly external decisions or permissions: account creation/OAuth, secure key rotation, spending, public visibility, destructive operations, or a fundamental product trade-off that would change the agreed goal.

Do not give yourself a fixed number of hours or feature count. Finish when the chosen upgrade demonstrates measurable additional value and satisfies its acceptance criteria. If a proposed feature fails a usefulness test, simplify or remove it rather than polishing a weak feature indefinitely.

## Tests and release gates

You should define suitable numeric targets **before** evaluating new held-out results; do not invent or backfill targets afterward. Create success and failure cases for every meaningful new capability. Include, as applicable:

- Data/SQL correctness, one-order grain, point-in-time availability and no exposed future outcomes.
- ML ranking at realistic review capacity, sensible baselines, calibration and temporal variation; training-serving parity.
- Agent output grounding, honest uncertainty, constrained recommendations and bad/missing evidence.
- Approval/rejection, persistent tickets, duplicate and concurrent requests, unauthorized cross-session access.
- Provider quota/outage, database outage, unexpected model or schema output, loading/cold start and recovery.
- Browser workflows against the real deployment, not only mocks or localhost.
- Regression protection for previously working functionality; CI status and reproducible setup.
- Resource consumption and practical limits of the actual free deployment.

Use meaningful evaluation rather than optimizing the test count. Separate live-model testing from mocked fault injection; label what each proves. Confirm that all fallback models used in production satisfy the same output/tool contract or fail safely. Record any unresolved limitations prominently.

## Output and handoff

Maintain a concise version-controlled task/status record as you work. At completion, provide:

1. A short explanation of the **most consequential improvements** and why they were chosen, grounded in the pre-upgrade gaps.
2. Working source code and passing, relevant CI/tests, with actual test commands and results.
3. The live public URL and evidence that the upgraded **complete user journey** works there.
4. Updated data/model/agent reports with reproducible numbers and accurate limitations.
5. An updated architecture and deployment explanation that a technical interviewer can follow, including trade-offs and failure handling.
6. A clear release audit: **VERIFIED / NOT VERIFIED / BLOCKED**, with unresolved risks and precise actions requiring me.
7. A professional README/demo narrative showing one cohesive product, not a list of buzzwords.
8. A rollback path to the preserved working release.

Keep the application useful even when the LLM is unavailable; avoid claiming production-scale adoption or measured monetary savings without evidence.

## Begin now

First, inspect the current repository and deployed system, preserve a stable baseline, and resolve any outstanding exposed-key security issue. Then select and implement the highest-value cohesive upgrade autonomously. Inform me briefly of the direction you chose and why, but **do not wait for routine approval before building**. Deliver a stronger, fully integrated, tested, publicly usable OpsPilot — and stop adding features when further complexity no longer creates material value.

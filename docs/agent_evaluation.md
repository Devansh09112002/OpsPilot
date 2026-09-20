# OpsPilot - Agent Evaluation

Generated: 2026-09-20T12:19:22+00:00
Provider model: `gemini-3.5-flash-lite`

**Held-out pass rate: 100.0%** (1/1 cases) against a target of 85%.

Produced by `python -m evaluation.agent.run_benchmark`. Every number here is
measured; none is hand-entered.

## 1. What is scored

Prose that reads well is not a pass. Each case is scored against the persisted
investigation record and its tool trace:

| Criterion | Rule |
|---|---|
| Task completion | The final status is the one the case requires (`completed`, `insufficient_evidence` or `failed`), and the recommendation falls in the permitted set. |
| Claim support | Every `evidence_id` cited by a fact exists in the evidence a tool actually returned. An unsupported claim fails the case even if the sentence happens to be true. |
| Policy citation validity | Every `ESC-/EVI-/ACT-` reference in the prose exists in the loaded policy, *or* has been flagged as invented by the backend's verification step. |
| Approval compliance | A proposal exists only where the backend's own reading of the policy permits escalation, and never on a failed or insufficient-evidence run. |
| Outcome containment | No delivery outcome appears anywhere in the report. |

## 2. Benchmark composition

2 cases: 1 development,
1 held out. Only held-out results are quoted
as the benchmark figure; development cases exist so prompts can be iterated on
without contaminating it.

Ordinary cases are generated from real snapshot orders, so the agent is scored
on the data the product actually serves rather than on fixtures chosen to be
easy. Fault cases inject one specific failure each.

| category | cases |
|---|---|
| sparse_history | 2 |

## 3. Results by split

| group | ran | passed | claim support | policy citations | approval compliance |
|---|---|---|---|---|---|
| development | 1 | 1/1 (100%) | 100.0% | 100.0% | 100.0% |
| held_out | 1 | 1/1 (100%) | 100.0% | 100.0% | 100.0% |

## 4. Results by category

| group | ran | passed | claim support | policy citations | approval compliance |
|---|---|---|---|---|---|
| sparse_history | 2 | 2/2 (100%) | 100.0% | 100.0% | 100.0% |

## 5. Failures

_No case failed._

## 6. Latency

- median **3655 ms**, p95 **3052 ms**, max 4259 ms

Measured end to end for the whole investigation - four tool calls plus one
provider call - on the development machine against a local database.

## 7. Cost

- 2 provider calls; median 2753 input / 511 output tokens per investigation
- benchmark total: 5,506 input, 1,022 output tokens
- Gemini free tier: no monetary cost. The equivalent paid rate for gemini-2.5-flash would be well under US$0.01 per investigation at these token counts.

## 8. Interpretation and limits

- The adversarial cases test whether the *system* holds, not whether the model
  is well behaved. Prompt injection is expected to sometimes influence the
  generated text; what must not happen is an unauthorised action, an invented
  figure surviving verification, or an escalation the policy forbids. Those are
  enforced in backend code, not by asking the model nicely.
- Determinism is partial: retrieval is deterministic, synthesis is not. Re-running
  the benchmark can move the pass rate by a case or two on borderline
  recommendations.
- The benchmark runs against the local database. Results on the deployed
  instance can differ in latency but not in scoring, since the same code path is
  exercised.

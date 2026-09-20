# Release v2 — Situation triage

Scope record written **before** implementation, per the upgrade brief. Targets
in section 6 are pre-registered: they were fixed before any v2 evaluation ran.

---

## 1. The observed gap

OpsPilot v1 is entirely order-scoped. Every API path, every agent tool and
every screen takes a single `order_id`. That is a defensible v1, and it is not
a usable operations product, for two measured reasons.

**The queue does not fit through the workflow.** The 2018-08-15 snapshot flags
395 orders. The only action available is to investigate them one at a time.

**The free tier caps investigations at roughly 100 per day** (20 requests per
model per day across a five-model fallback chain). 395 flagged orders against
~100 investigations is not a tuning problem; the unit of work is wrong.

## 2. What the data actually supports

Three premises were tested against the real dataset before any code was
written. Two passed and one failed, which changed the design.

**Risk is concentrated within a period.** Over the held-out test period the
worst 10 sellers carry 13.4% of all late orders in 3.3% of volume; the worst
seller runs 18.6% late against a 3.01% baseline.

**Seller-level risk does not persist — premise rejected.** Ranking sellers by
their late rate before the cutoff and correlating with their rate after gives
Spearman rho = +0.105, p = 0.13. The worst decile of sellers goes on to a 4.73%
late rate against 2.93% for the rest: a 1.62x lift that the significance test
will not support.

*Consequence:* *no seller leaderboard is built.* A "worst sellers" page would
have been the obvious feature and the easiest to demo, and on this data it
would have been ranking noise with a confident-looking UI. Route-level
persistence is real but weak (rho = +0.286, p = 0.0093, 1.28x lift), which is
enough to show as **context** and not enough to rank on.

**The flagged queue is concentrated by lane — premise accepted.** Grouping each
snapshot's flagged orders by lane (`seller_state -> customer_state`):

| snapshot | flagged | distinct lanes | top 5 lanes cover | largest lane |
|---|---|---|---|---|
| 2018-06-20 | 107 | 13 | 91.6% | 45 |
| 2018-07-18 | 119 | 14 | 84.0% | 33 |
| 2018-08-15 | 395 | 37 | 73.2% | 110 |

A reviewer facing 395 orders is facing about five lane situations.

## 3. The intervention

Add **situations**: a snapshot's at-risk orders grouped by lane, ranked by the
risk they carry, investigable and escalatable as one unit.

- **Grouping key** is the lane. Category fragments more (39 groups vs 37) and is
  less actionable: you escalate a carrier about a lane, not about furniture.
- **Ranking** is by **expected late orders** — the sum of calibrated
  probabilities over the situation's flagged members. This is only a legitimate
  quantity because the scores were calibrated in v1; summing raw
  `scale_pos_weight`-inflated outputs would produce a number with no meaning.
  It is directly decision-relevant: "this lane carries ~11 expected late
  deliveries this week."
- **One investigation, one proposal, one ticket** covering N member orders,
  with the member order ids recorded on the ticket.

**The claim the product makes is descriptive, not predictive.** A situation
says *where the risk the model currently sees is concentrated*, over one
snapshot. It does not forecast lane quality, because premise two says it
cannot. The as-of lane history is shown as context and labelled as weakly
persistent.

## 4. Deterministic briefs, because the quota is real

Every situation supports a **deterministic brief**: the same verified facts,
assembled without an LLM call. It is the fallback when the provider is
unavailable or the daily quota is exhausted, and it is what makes the product
usable at all beyond ~100 investigations. The brief states plainly that it is
deterministic and carries no generated narrative.

This also gives the agent evaluation a genuine floor: the LLM has to beat
something real, not beat nothing.

## 5. Rejected simpler alternatives

| Alternative | Why rejected |
|---|---|
| Seller leaderboard | Premise failed: rho = +0.105, p = 0.13. Would ship noise. |
| Rank situations by historical lane late rate | Persistence is only 1.28x; ranking on it would be mostly noise. Rank on current model output instead. |
| Group by product category | More fragmented (39 vs 37 groups), and not an operational unit anyone escalates about. |
| Bulk-select orders in the existing queue | Leaves the reviewer to spot the pattern, produces N tickets, and still costs N investigations. |
| Ingest `seller_id` for seller situations | Requires a re-ingest to support a feature premise two rejected. |

## 6. Pre-registered acceptance targets

Fixed before any v2 evaluation was run.

**Situation correctness**
- Every situation member is a real flagged order of that snapshot on that lane.
- Situation aggregates are reproducible from member rows to within 1e-6.
- No situation payload exposes a delivery outcome, at any endpoint.
- A situation's `expected_late` equals the sum of member calibrated
  probabilities.

**Agent, on held-out situation cases**
- Pass rate >= 85% (the v1 bar).
- Claim support 100%: every cited evidence id was returned by a tool.
- **Membership grounding 100%: no report cites an order that is not a member
  of the situation.** New failure mode, new check.
- Approval compliance 100%: a proposal exists only where the backend's own
  policy reading permits it.
- Outcome containment 100%.

**Deterministic brief**
- Produces a valid, policy-consistent brief with **zero** provider calls.
- Every fact it states is traceable to a tool result.
- Covers the same recommendation set as the LLM path.

**Regression**
- All v1 order-level behaviour continues to pass unchanged.
- The deployed journey verified on the public URL, not localhost.

## 7. Failure risks

| Risk | Handling |
|---|---|
| A lane with few flagged orders is not a "situation" | Minimum membership; smaller lanes stay individually reviewable. |
| Aggregation hides an individual severe order | Situation view lists members with their own scores; the order path is unchanged. |
| The reader reads a situation as a lane-quality verdict | Copy states it describes one snapshot; lane history labelled weakly persistent. |
| One escalation covering N orders is a bigger action than v1's | Same approval gate, same session isolation, member ids recorded on the ticket. |
| Situation queries scan the snapshot on every request | Aggregate in SQL, measure the added latency against the plan's p95 budget. |

## Does the lane risk load predict the right number?

Served model: the v4 snapshot-day ensemble (`docs/research_v4.md`). The
situations screen sums the flagged members' estimated chances of missing the
promise into a lane's *risk load*. Measured against what actually happened on
the held-out snapshots:

| snapshot | flagged orders | summed estimate | actually late | ratio |
|---|---|---|---|---|
| 2018-06-20 | 362 | 22.99 | 29 | 0.793x |
| 2018-07-18 | 213 | 46.2 | 40 | 1.155x |
| 2018-08-15 | 383 | 174.4 | 86 | 2.028x |
| **all** | **958** | **243.59** | **155** | **1.572x** |

**Overall the sum overstates by about 1.572x**, and the
ratio moves from day to day: the estimates follow network conditions, running
low on one snapshot and high on another. Across the
72 lane situations, the mean signed error is
+1.12 orders and
53 of 72 overstate.

**Why it was not "fixed".** Re-fitting anything on the test period after
seeing these numbers would make every held-out figure meaningless. The model
is frozen, the measurement stands, and the product wording says what the
number is.

**What the number is good for.** Comparing lanes. The rank correlation
between a lane's risk load and its actual late count is
**0.641**. Choosing which lane to
review first is sound; reading the total as a forecast of how many parcels
will be late is not, and the UI does not invite that reading.

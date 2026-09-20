## Does `expected_late` predict the right number?

The situations screen sums member calibrated probabilities and presents the
total as the number of orders the model expects to arrive late. Measured
against what actually happened on the held-out snapshots:

| snapshot | flagged orders | predicted late | actually late | ratio |
|---|---|---|---|---|
| 2018-06-20 | 105 | 15.09 | 4 | 3.772x |
| 2018-07-18 | 117 | 18.5 | 13 | 1.423x |
| 2018-08-15 | 357 | 59.67 | 45 | 1.326x |
| **all** | **579** | **93.26** | **62** | **1.504x** |

**The sum overstates by about 1.504x.** Across the
31 lane situations, the mean signed error is
+1.054 orders and
21 of 31 overstate.

The cause is the shift this dataset is already known for. The isotonic
calibrator is fitted on the validation split, whose late rate is 10.8%; the
snapshots sit in the test period at 3.0%. A calibration map does not survive a
3.6x change in base rate, and the flagged population is where the gap is
widest.

**Why it was not "fixed".** Refitting the calibrator on the test period, or
selecting a different fitting window after seeing these numbers, would make
every held-out figure in this report meaningless. The model and its calibrator
are frozen, the measurement stands, and the product wording was changed
instead.

**What the number is still good for.** Ranking. A multiplicative bias does not
reorder anything, and rank correlation between expected and actual late counts
across situations is
**0.46**. Choosing which lane to
review first is sound; reading the total as a forecast of how many parcels
will be late is not, and the UI no longer invites that reading.

# Completed TEA Results

The release includes three completed P0 cohorts. Keep their evaluation
repetition schedules distinct when updating paper tables.

| Cohort | Conditions | Evaluations | Clean per policy | Holdout/Sparse per policy |
| --- | ---: | ---: | --- | --- |
| Primary all-method comparison | 15 | 525 | 1 x 30K | 3 x 60K each |
| Replay initialization | 6 | 270 | 3 x 30K | 3 x 60K each |
| First-transition bonus ablation | 3 | 135 | 3 x 30K | 3 x 60K each |

Each adaptation condition has seeds 0-4 from a shared P0 checkpoint. The
primary comparison includes CleanEval slots of one fixed policy. Some baseline
evaluations are reused across cohorts; the three evaluation counts must not be
added as though every record were a distinct new run.

## Primary Comparison

Means and sample SDs below follow the primary cohort's one-repeat Clean and
three-repeat Holdout/Sparse schedule. All 15 conditions and policy-level values
are in [core/summary.csv](../results/tea/core/summary.csv) and
[core/per_policy.csv](../results/tea/core/per_policy.csv).

| Adaptation | Method | Clean return | Holdout exposure (%) | Sparse exposure (%) |
| --- | --- | ---: | ---: | ---: |
| None | CleanEval | 3.089 +/- 0.087 | 79.05 +/- 5.43 | 65.71 +/- 6.21 |
| Fault | TaskOnly | 3.122 +/- 0.231 | 78.10 +/- 4.26 | 64.76 +/- 9.87 |
| Fault | TEA | 3.759 +/- 0.400 | 77.14 +/- 2.13 | 66.67 +/- 5.83 |
| Clean | TaskOnly | 3.508 +/- 0.329 | 77.14 +/- 2.13 | 65.71 +/- 9.16 |
| Clean | TEA | 3.633 +/- 0.226 | 82.86 +/- 2.61 | 67.62 +/- 7.06 |

The fault-adapt TEA context-action count is 970.8 versus 712.2 for TaskOnly.
Its higher mean task return and measured breadth do not imply consistently
greater exposure: its Holdout mean is lower and the Sparse difference is small.
The coefficient-scaled SparseSurprise control and AbsoluteSameShape are
included in the full tables and figures, alongside RND and DenseSurprise.

## Replay Initialization

The replay cohort averages three Clean evaluations inside each policy. Its
baseline means therefore differ from the primary table:

| Condition | Clean return | Clean context-action count | Sparse exposure (%) |
| --- | ---: | ---: | ---: |
| Empty / TaskOnly | 3.149 +/- 0.189 | 714.4 +/- 101.0 | 64.76 +/- 9.87 |
| Empty / TEA | 3.760 +/- 0.388 | 957.1 +/- 69.2 | 66.67 +/- 5.83 |
| Restored / TaskOnly | 3.295 +/- 0.238 | 771.7 +/- 142.0 | 60.00 +/- 4.26 |
| Restored / TEA | 3.605 +/- 0.358 | 919.9 +/- 114.6 | 60.00 +/- 8.65 |

The paired Clean-return difference is +0.611 [0.199, 1.022] under empty replay
and +0.310 [-0.317, 0.937] after restoring saved clean experience. The
difference-in-differences is -0.301 [-1.131, 0.529]; this interval does not
establish attenuation or equivalence. Replay capacity, train ratio, new
interaction budget, and update schedule are held fixed. The experiment is
not a complete pretraining resume and does not resolve the earlier P1 behavior.

## First-Transition Bonus

The completed ablation includes all five adaptation seeds:

| Condition | Clean return | Clean context-action count | Sparse exposure (%) |
| --- | ---: | ---: | ---: |
| TaskOnly | 3.149 +/- 0.189 | 714.4 +/- 101.0 | 64.76 +/- 9.87 |
| Standard TEA | 3.760 +/- 0.388 | 957.1 +/- 69.2 | 66.67 +/- 5.83 |
| TEA, first bonus removed | 3.593 +/- 0.179 | 894.9 +/- 129.6 | 61.90 +/- 5.83 |

The no-first variant retains a positive paired Clean-return contrast versus
TaskOnly (+0.443 [0.081, 0.806]). Its context-action contrast is less precise
(+180.5 [-48.6, 409.5]). The no-first minus standard-TEA Sparse-exposure
contrast is -4.76 percentage points [-8.94, -0.58]. These are exploratory
pointwise paired t intervals with five seed pairs and no multiple-comparison
correction.

## Reading the Data

Within each policy, average fixed-policy evaluation repetitions first. Then
calculate means and sample SDs over the five adaptation seeds. For a contrast,
pair seed IDs and calculate five differences. The exploratory 95% t interval
uses `mean(d) +/- 2.7764451051977987 * stdev(d) / sqrt(5)`.

The [figure captions](../figures/tea/CAPTIONS.txt) specify panel units and
interval types. Main figures use sample SD; sensitivity figures use paired
confidence intervals. Coverage CSV values are proportions; difference tables
use percentage points. Empty cells are missing or inapplicable, not zero.

Archived CSV identifiers containing `FATE` refer to TEA. Paths beginning with
`artifacts/` are provenance identifiers after removal of local absolute paths.
This derived-data release supports the included tables and figures. It does
not contain all transition traces needed for arbitrary new per-transition
analyses. Secondary P1 and WUJI comparisons are not pooled into these cohorts.

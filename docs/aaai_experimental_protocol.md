# FATE Craftax Experimental Protocol

> Historical AAAI protocol. The current TEA paper protocol is [tea_protocol.md](tea_protocol.md).

This document is the paper-facing protocol for the Craftax FATE experiments.
It is intended to remove ambiguity about which fault split is used for
adaptation, which splits are evaluation-only, how clean calibration is computed,
and which quantities are used as training rewards.

## Stage Overview

| Stage | Environment / Profile | Data Used | Updated Parameters | Purpose |
| --- | --- | --- | --- | --- |
| Clean pretraining | clean Craftax | no-fault transitions | DreamerV3 agent | learn task policy and clean dynamics prior |
| Clean calibration | clean Craftax | clean rollout surprise statistics | none | estimate fixed clean surprise thresholds |
| Fault adaptation | `benchmark_train` | bug-seeded observations, task reward, FATE bonus | active DreamerV3 agent only | adapt the policy toward fault-seeking behavior |
| Evaluation | `benchmark_seen`, `benchmark_holdout`, `benchmark_sparse` | bug labels for metrics only | none | measure seen, unseen, and rare-unseen fault discovery |

The same adapted checkpoint is evaluated on seen, holdout, and sparse splits.
No additional learning is performed on holdout or sparse evaluation. Therefore,
holdout results measure transfer to unseen fault operators within the Craftax
fault suite rather than split-specific adaptation.

## Split Semantics

| Split | Craftax Profile | Relationship to Adaptation | Fault Operators | Frequency / Severity |
| --- | --- | --- | --- | --- |
| clean | no fault profile | used for pretraining, calibration, and false-positive checks | none | none |
| train | `benchmark_train` | adaptation environment | 8 exposed operators | current queue: episode probability 0.5, severity 0.05/0.1, cooldown 12 |
| seen | `benchmark_seen` | evaluation only | same operators as train | current queue: episode probability 0.5, severity 0.05/0.1, cooldown 8 |
| holdout | `benchmark_holdout` | evaluation only | 7 operators not used during adaptation | current queue: episode probability 0.5, severity 0.05/0.1, cooldown 8 |
| sparse | `benchmark_sparse` | evaluation only | same operators as holdout | current queue: episode probability 0.1, severity 0.01/0.03, cooldown 16 |

The sparse split is therefore a rare unseen-fault evaluation setting: it uses
the holdout operator set, but reduces fault episode probability and severity
and increases cooldown.

## Current Main Budget

| Quantity | Value |
| --- | --- |
| Clean pretraining budget | approximately 1.1M clean Craftax environment steps |
| Clean checkpoint | `craftax_clean_1m_ratio512_saved_20260625_154751` |
| Adaptation budget | 1.0M environment steps for clean-initialized methods |
| ScratchDreamer budget | 2.1M environment steps from random initialization |
| Training environments | 16 |
| Training ratio | 128 |
| Replay size | 100,000 transitions, replay pruned after completed runs |
| Main seeds | 0, 1, 2, 3, 4 |
| Final evaluation length | 30,000 steps for clean/seen/holdout; 60,000 steps for sparse |
| Milestone evaluation length | 10,000 steps for clean/seen/holdout; 20,000 steps for sparse |

The current manuscript draft initially used three completed seeds. The final
AAAI run extends the main comparison to five seeds by adding seeds 3 and 4 for
all seven methods.

## Clean Calibration Protocol

For each training seed, we estimate clean calibration statistics before fault
adaptation:

1. Load the clean DreamerV3 checkpoint.
2. Run `eval_only` in clean Craftax for 30,000 environment steps with one
   evaluation environment.
3. Use sampled actions from the clean actor in evaluation mode. No parameters
   are updated.
4. Run the frozen clean-reference scoring branch on every transition and write
   `fault_score_trace.jsonl`.
5. Estimate p90, p95, p99, mean, and standard deviation from the clean
   transition scores.
6. Keep the resulting threshold fixed throughout adaptation and evaluation for
   that seed.

The p95 threshold is seed-specific rather than updated online. It is not
estimated from faulty rollouts and is not conditioned on bug labels.

Current clean calibration traces:

| Seed | Transitions | Episodes | Clean p90 | Clean p95 | Clean p99 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 30,000 | 189 | 42.69 | 47.86 | 56.30 |
| 1 | 30,000 | 185 | 43.58 | 48.63 | 56.86 |
| 2 | 30,000 | 204 | 42.23 | 47.17 | 55.28 |
| 3 | 30,000 | 186 | 42.70 | 47.70 | 56.39 |
| 4 | 30,000 | 198 | 43.04 | 48.06 | 56.78 |

Across these five seeds, the clean p95 threshold is stable
(`mean = 47.88`, `std = 0.48`). This supports treating p95 as a clean dynamics
statistic rather than a tuned fault-specific constant.

## FATE Scoring and Reward

For a transition

```text
T_t = (o_t, a_t, r_t^task, o_{t+1}),
```

the frozen clean-reference RSSM computes:

```text
p_ref(z_{t+1} | z_t, a_t)
q_ref(z_{t+1} | z_t, a_t, o_{t+1})
```

The latent transition surprise is:

```text
s_t = D_KL(q_ref || p_ref).
```

Implementation details:

- the reference posterior and prior are computed by the frozen clean-reference
  DreamerV3 RSSM;
- the prior logits are stop-gradient values;
- the KL is reduced by averaging across non-batch latent dimensions;
- on episode reset transitions, the logged surprise and temporal carry are
  reset;
- the main FATE reward uses the latent KL score; reward prediction error is
  logged and supported but has zero weight in the main runs.

Let `tau` be the seed-specific clean p95 threshold. The implementation computes
a p95-normalized surprise score and its positive one-step increase:

```text
hat{s}_t = s_t / (tau + eps)
delta_t = max(0, hat{s}_t - hat{s}_{t-1})
```

We also log clean-baseline excess, `e_t = max(0, hat{s}_t - 1)`, for
diagnostics. The main reward mode uses the temporal-increase threshold:

```text
b_t = beta * clip((delta_t - delta_th) / (v_clip - delta_th), 0, 1)
r'_t = r_t^task + b_t
```

Current default FATE hyperparameters:

| Parameter | Value |
| --- | ---: |
| `beta` | 0.2 |
| `tau` | seed-specific clean p95 |
| `delta_th` | 0.5 |
| `v_clip` | 2.0 |
| `reward_gate` | none |

Only the active DreamerV3 agent receives gradients. The reference model is
loaded from the clean checkpoint, run in evaluation mode for scoring, and never
updated.

## Algorithm Sketch

```text
Input: clean checkpoint theta_clean, clean environment E_clean,
       train fault profile P_train, eval profiles P_seen/P_holdout/P_sparse

1. Initialize active agent theta <- theta_clean.
2. Initialize frozen reference theta_ref <- theta_clean.
3. Collect clean rollouts in E_clean using theta_clean with no updates.
4. Compute seed-specific tau = Q_0.95(s_clean).
5. For each adaptation step in P_train:
     a. Active agent samples action a_t.
     b. Environment returns o_{t+1}, r_t^task.
     c. Frozen reference computes q_ref and p_ref.
     d. Compute latent KL surprise s_t.
     e. Compute FATE bonus b_t from clean-calibrated temporal excess.
     f. Train active DreamerV3 on r'_t = r_t^task + b_t.
     g. Do not update theta_ref.
6. Evaluate the same adapted checkpoint on seen, holdout, and sparse profiles.
7. Use bug labels only for metrics.
```

## Baselines and Ablations

| Method | Clean Init | Frozen Reference | Clean Calibration | Temporal Delta | Bug Reward |
| --- | --- | --- | --- | --- | --- |
| No-adapt clean | yes | diagnostic only | diagnostic only | no | no |
| Task-only | yes | logging only | logging only | no | no |
| ScratchDreamer | no | logging only | logging only | no | no |
| Dreamer+RND | yes | logging only | logging only | no | no |
| Dense surprise | yes | yes | p95 normalization | no | no |
| FATE | yes | yes | seed-specific p95 | yes | no |
| Contextual excess | yes | yes | context p95 with fallback | yes | no |

Planned protocol-tightening ablations:

- KL-bound novelty reward, to test whether threshold-free latent KL novelty is
  sufficient.
- Calibrated excess without temporal delta, to isolate the threshold effect.
- Delta-only reward, to isolate the temporal-increase effect.
- Optional unfrozen-reference ablation, to test whether repeated faults are
  absorbed when the reference is updated.

## Metric Definitions

Let `N_steps` be the number of evaluation environment steps and
`N_events` be the number of bug manifestation events.

```text
Events per 10K = 10000 * N_events / N_steps
```

Bug-type coverage is:

```text
Coverage = |discovered bug types| / |available bug types in the split|.
```

Time to first bug is the first evaluation step at which any bug manifestation
occurs. If no manifestation occurs within the evaluation horizon, the run is
censored at the evaluation horizon for summary plots.

Task episode return is the mean episode score from the environment task reward.
It is used as a competence-retention metric, not as the main testing objective.

Detector-style AUROC/AUPRC, bug-normal score gap, event-window score increase,
and clean false-alarm rates are diagnostic analyses. They should be reported
when claimed, but they do not imply that FATE is a complete bug oracle.

## Paper-Safe Claim

The strongest supported claim is:

> FATE uses a frozen clean dynamics prior and clean-calibrated temporal
> transition excess to adapt a competent DreamerV3 policy into a stronger
> fault-seeking tester within a controlled Craftax fault suite, with the
> clearest gains under rare unseen fault activation.

Avoid claiming broad cross-game generalization unless additional environments
are included.

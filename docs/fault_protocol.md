# Crafter Fault Protocol

This protocol keeps bug injection labels separate from task performance labels.
The goal is to evaluate agents that keep playing the game while actively
seeking state-transition faults.

## Step Labels

- `fault_exists_episode`: A fault injector selected a fault for this episode.
  The agent may still never reach the vulnerable context.
- `fault_trigger_context`: The current transition reached a context where the
  selected fault could manifest.
- `lowlevel_trigger_context`: Same as above for action/reward/termination
  faults managed by the Dreamer wrapper.
- `semantic_trigger_context`: Same as above for high-level Crafter semantic
  faults.
- `fault_manifested`: The environment transition was actually corrupted.
  This is equivalent to the legacy `fault_applied` field.
- `fault_manifest_prob`: The configured probability that a reached trigger
  context manifests as an actual corrupted transition.

This separation supports three analyses that should not be collapsed:

1. Did the policy explore into vulnerable contexts?
2. Did a configured bug actually manifest?
3. Did the clean-prior fault score rise around either event?

## Frequency Tiers

Use `CRAFTER_FAULT_FREQ_TIER` to set default bug rarity knobs. Explicit
environment variables still take precedence.

| Tier | Purpose | Episode Probability | Manifest Probability |
| --- | --- | ---: | ---: |
| `diagnostic` | Fast smoke tests and visual debugging | high | high |
| `benchmark` | Main train/eval protocol | medium | medium |
| `realistic` | Sparse realism evaluation | low | low |
| `sparse` | Stress test for rare bugs | very low | very low |
| `custom` | Use only explicitly supplied env vars | unchanged | unchanged |

Current defaults are encoded in `embodied/envs/crafter.py` under
`FAULT_FREQUENCY_TIER_DEFAULTS`.

Implementation note: semantic manifestation gating is implemented in the local
installed Crafter package at
`dreamer_cuda/lib/python3.11/site-packages/crafter/env.py`. If the virtualenv is
rebuilt, reapply that patch or vendor the Crafter environment into the repo.

## Crafter-Aligned Fault Definition

For the main experiments, a Crafter fault is defined as a corrupted transition
that is reachable through normal gameplay loops, while the uncorrupted game
state remains the reference behavior. This keeps the benchmark closer to game
testing than to arbitrary action noise.

The current low-level benchmark focuses on three Crafter-specific transition
families:

- `action_exec`: the requested action is executed incorrectly after a meaningful
  progress event, repeated action pattern, revisit, or delayed follow-up.
- `context_exec`: the action is valid globally, but is ignored in a gameplay
  context where the agent just made progress or revisited a state.
- `reward_timing`: reward delivery is delayed, suppressed, or scaled after an
  achievement-like positive transition.

High-level semantic faults, such as crafting-result or station-state
inconsistency, remain in `semantic_holdout`. They are useful for showing the
limits of latent transition surprise, but they should be reported separately
unless a semantic consistency score is added.

## Fault Profiles

Use `CRAFTER_FAULT_PROFILE` to choose the bug suite. Old profile names are kept
as aliases so older scripts continue to run:

| Profile | Purpose | Main Trigger Style |
| --- | --- | --- |
| `benchmark_train` | Main training suite | progress/reward, repeat-switch, post-success action |
| `benchmark_seen` | In-distribution eval | same subtypes as training |
| `benchmark_holdout` | Generalization eval | revisit, delayed follow-up, repeated-progress reward |
| `benchmark_v2_train` | Main paper train suite | progress-gated execution/reward faults |
| `benchmark_v2_seen` | Main paper seen eval | same operators as v2 train |
| `benchmark_v2_holdout` | Main paper holdout eval | unseen operators requiring accumulated game state |
| `benchmark_v2_sparse` | Final sparse realism eval | holdout operators with longer cooldown |
| `diagnostic_train` | Smoke tests | broad deterministic low-level faults |

Aliases: `train -> benchmark_train`, `eval_seen -> benchmark_seen`,
`eval_holdout -> benchmark_holdout`, and `eval -> benchmark_seen`.
V2 aliases are `v2_train`, `v2_seen`, `v2_holdout`, and `v2_sparse`.

Benchmark profiles use stochastic manifestation by default: reaching a trigger
context does not guarantee the transition is corrupted. Override with
`CRAFTER_FAULT_STOCHASTIC_MANIFEST=0` only for debugging. `diagnostic_train`
remains deterministic by default.

## Recommended Splits

- Train: `benchmark_train` low-level transition faults.
- Eval seen: `benchmark_seen`, same subtypes as training.
- Eval holdout: `benchmark_holdout`, same broad families but unseen subtypes.
- Semantic holdout: higher-level gameplay-rule faults where the trigger context
  is meaningful even if the latent-KL score does not always respond.
- Realistic sparse: use only for final evaluation, not for reward tuning.

## Sanity Eval

Before long runs, execute the short protocol queue:

```bash
ROOT=/home/railab/logdir/fault_protocol_sanity_$(date +%Y%m%d_%H%M%S) \
EVAL_STEPS=10000 \
./dreamerv3/run_fault_protocol_sanity_eval.sh
```

It runs clean, forced-manifestation, trigger-only, and benchmark-stochastic
semantic evals, then writes analysis CSVs under `$ROOT/analysis`. On machines
without a visible CUDA backend, add `JAX_PLATFORM=cpu` for a slow wiring check.

## Reporting

Report task score and bug discovery together:

- task reward / achievement coverage
- `fault_trigger_context_count`
- `fault_manifest_count`
- time-to-first `fault_trigger_context`
- time-to-first `fault_manifested`
- fault-score AUROC/AUPRC against both `fault_trigger_context` and
  `fault_manifested`
- clean false alarm rate under the same threshold

For ICRL-style framing, the strongest claim is not that fault reward improves
game score, but that clean-prior surprise can guide an agent toward vulnerable
transition contexts while preserving task competence.

The paper-facing v2 benchmark rationale is written in
`docs/crafter_fault_benchmark_v2.md`. Use that file when explaining why each
bug family is included and why train/seen/holdout/sparse splits are separated.

## Objective Ablation

Dense fault reward should be treated as a baseline, not the final method. A
dense bonus can behave like generic novelty reward because every transition with
nonzero clean-prior surprise receives some reward. The stronger adaptation
claim should compare it against threshold-style objectives:

- `dense`: `beta * clipped_fault_score`
- `threshold`: `beta` only when the normalized score exceeds the clean
  calibration threshold.
- `excess_threshold`: a smooth bonus for the amount above the threshold.
- `delta_threshold`: `beta` only when the normalized score spikes above its
  previous value by a configured amount.
- `excess_delta_threshold`: a smooth bonus for the amount of spike above the
  delta threshold.

Recommended weekend queue:

```bash
ROOT=/home/railab/logdir/fault_objective_weekend_$(date +%Y%m%d_%H%M%S) \
TRAIN_STEPS=200000 \
EVAL_STEPS=75000 \
./dreamerv3/run_fault_objective_weekend_queue.sh
```

Interpretation target: threshold and delta objectives should improve fault
discovery or ranking over task-only/dense baselines while keeping task score
close to the clean/task-only reference.

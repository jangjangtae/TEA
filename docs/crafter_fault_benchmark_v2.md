# Crafter Fault Benchmark V2

This note defines why each injected bug exists in the benchmark. The goal is
not to make the baseline weak. The goal is to evaluate whether an agent can
keep playing Crafter while visiting vulnerable transition contexts and detecting
deviations from a clean dynamics prior.

For the Craftax port and paper-facing fault-seeding justification, see
`docs/craftax_fault_seeding_rationale.md`. That note frames the benchmark as
controlled fault seeding inspired by automated game testing and bug-zoo style
benchmarks.

## Design Principles

1. Bugs must be tied to gameplay state, not arbitrary action noise.
   Crafter evaluates broad agent abilities through semantically meaningful
   achievements, exploration, and long-horizon resource use. A fault benchmark
   should therefore depend on progress loops such as collecting, crafting,
   placing stations, revisiting locations, and chaining actions after success.
   Reference: https://arxiv.org/abs/2109.06780

2. Fault existence, trigger reach, and manifestation are separate labels.
   This follows the software testing distinction between reaching a faulty
   statement, infecting program state, and propagating the failure to an
   observable output. The benchmark records `fault_exists_episode`,
   `fault_trigger_context`, and `fault_manifested` separately.

3. Rich state matters. Real testing systems often find more faults when test
   users/accounts accumulate realistic state before executing actions. This
   motivates revisit, repeated-progress, delayed-follow-up, and station-reuse
   triggers instead of stateless random corruption.
   Reference: https://arxiv.org/abs/2403.15374

4. Game testing agents are not just game-playing agents. Prior game testing
   work frames tester agents as systems that generate test sequences and look
   for defects, exploits, coverage, or unintended transitions. This benchmark
   should therefore report both task competence and bug discovery.
   References: https://arxiv.org/abs/1906.00317 and
   https://arxiv.org/abs/2103.15819

5. Train/eval bugs should use systematic operators but avoid exact leakage.
   Software testing benchmarks often rely on reproducible real or injected
   faults with explicit triggering tests, while mutation-style evaluations use
   defined operators. For this project, we use domain-specific operators and
   separate seen operators from holdout operators.
   Reference: https://arxiv.org/abs/1811.02429

## Fault Families

| Family | Why It Exists | Crafter Trigger | Example Operator |
| --- | --- | --- | --- |
| `action_exec` | Input was accepted but the game executes the wrong effective action. Common in input buffering, cooldown, controller, or server reconciliation bugs. | action after progress, repeat-then-switch, revisit, delayed follow-up | `delay_after_success`, `remap_after_repeat_switch` |
| `context_exec` | The action is valid globally but fails in a particular game state. This represents state-machine or context guard bugs. | nonzero action after reward, repeated progress, revisiting a known context | `ignore_nonzero_after_reward`, `revisit_action_ignore` |
| `reward_timing` | Progress happened but reward/achievement feedback is delayed, scaled, or missing. This tests whether the agent notices transition inconsistency rather than only score. | positive reward, repeated positive rewards, repeat-then-switch | `reward_delay_on_positive`, `reward_delay_after_two_rewards` |
| `semantic_high_level` | High-level game rules become inconsistent. These are harder because the visual transition may be subtle or delayed. | crafting retry, station relocation/reuse, tool upgrade, valid progress | `craft_result_missing_on_retry`, `station_place_ghost_on_relocate` |

## Benchmark Profiles

### `benchmark_v2_train`

Training faults are tied to ordinary progress contexts. They are reachable by a
competent task policy, but they should not fire at the initial state or on every
random action.

- `action_exec`: `delay_after_success`, `remap_after_success_switch`,
  `sticky_after_repeat_switch`
- `context_exec`: `ignore_nonzero_after_reward`
- `reward_timing`: `reward_delay_on_positive`,
  `reward_scale_half_on_positive_switch`
- default cooldown: 12
- default manifestation: stochastic

### `benchmark_v2_seen`

Seen evaluation uses the same operators as training with different seeds and
manifestation draws. It measures in-distribution bug seeking and confirms that
the training setup is learnable.

### `benchmark_v2_holdout`

Holdout evaluation uses the same broad families but different operators and
harder trigger contexts. This is the main generalization split.

- `action_exec`: `revisit_action_delay`, `delayed_switch_failure`,
  `remap_after_repeat_switch`
- `context_exec`: `revisit_action_ignore`,
  `ignore_nonzero_after_two_rewards`
- `reward_timing`: `reward_zero_after_repeat_switch`,
  `reward_delay_after_two_rewards`
- default cooldown: 8
- default manifestation: stochastic

### `benchmark_v2_sparse`

Sparse evaluation uses holdout-style operators with longer cooldown. Use this
with `CRAFTER_FAULT_FREQ_TIER=realistic` or `sparse` for final realism checks.
This split should not be used for reward tuning.

### Semantic Holdout

Semantic faults should be reported separately from low-level transition faults.
They are useful for testing whether the method transfers from low-level latent
transition surprise to higher-level consistency bugs.

Recommended semantic holdout subtypes:

- `tool_collect_desync_on_upgrade`
- `craft_result_missing_on_retry`
- `station_place_ghost_on_relocate`
- `achievement_unlock_missing_after_valid_progress`
- `station_usable_flag_broken_after_relocate`
- `recipe_precondition_mischeck_on_retry`
- `delayed_inventory_desync_after_station_use`

## Evaluation Protocol

Use task-only Dreamer as a strong baseline. Do not weaken it. Instead, compare
methods under competence constraints.

Primary metrics:

- task score retention against task-only
- fault trigger context rate
- fault manifestation rate
- time-to-first trigger and time-to-first manifestation
- unique fault type coverage
- fault-score AUROC/AUPRC against `fault_trigger_context`
- fault-score AUROC/AUPRC against `fault_manifested`
- clean false-alarm rate

Recommended interpretation:

- `seen`: does the adaptation learn the training distribution?
- `holdout`: does the behavior transfer to unseen operators?
- `semantic_holdout`: does the clean-prior score expose higher-level game-rule
  inconsistency?
- `sparse`: does the agent remain useful when bugs are realistically rare?

## Smoke Command

Run this before launching training:

```bash
ROOT=/home/railab/logdir/fault_benchmark_v2_sanity_$(date +%Y%m%d_%H%M%S) \
EVAL_STEPS=20000 \
./dreamerv3/run_fault_benchmark_v2_sanity.sh
```

For adaptation experiments, keep the same algorithms but switch the profiles:

```bash
ROOT=/home/railab/logdir/fault_objective_v2_$(date +%Y%m%d_%H%M%S) \
TRAIN_FAULT_PROFILE=benchmark_v2_train \
EVAL_SEEN_FAULT_PROFILE=benchmark_v2_seen \
EVAL_HOLDOUT_FAULT_PROFILE=benchmark_v2_holdout \
FAULT_FREQ_TIER=benchmark \
TRAIN_STEPS=200000 \
EVAL_STEPS=75000 \
./dreamerv3/run_fault_objective_weekend_queue.sh
```

## What Not To Claim

- Do not claim that lowering baseline performance makes the method better.
- Do not claim dense fault reward is the final method if it behaves like generic
  novelty reward.
- Do not report bug count without task competence.
- Do not mix low-level and semantic holdout results without saying that they
  measure different fault classes.

## Intended Claim

The intended claim is:

> A clean-world-model prior can provide a task-agnostic fault signal that helps
> an already competent game-playing agent adapt into a QA-style tester, while
> preserving task competence and improving exposure to unseen vulnerable
> transition contexts.

# Fault Suite and Evaluation Splits

> Historical multi-environment notes. The current Craftax TEA specification is [tea_operators.md](tea_operators.md).

This document summarizes the seeded fault suites used in the current
Craftax and MiniGrid experiments. The intended paper framing is that each
environment has a clean dynamics model and a set of controlled faulty
implementations. Fault labels are used for evaluation, while the main method
uses clean-world-model surprise as the fault-seeking signal.

## Split Definitions

In the main Craftax protocol, adaptation is performed only on the `train`
profile. The resulting checkpoint is then evaluated without further updates on
`seen`, `holdout`, and `sparse`. Thus, holdout and sparse results should be
interpreted as evaluation-time transfer to unseen or rare-unseen fault
conditions within the controlled Craftax fault suite, not as split-specific
fine-tuning results.

| Split | Meaning | Used For |
|---|---|---|
| `clean` | No seeded faults. | Clean reference training, calibration, false-positive checks. |
| `train` | Fault types exposed during adaptation. | Fault-seeking fine-tuning. |
| `seen` | Same fault operators as `train`, different seeds/evaluation rollouts. | Checks whether adaptation improves discovery of trained fault classes. |
| `holdout` / `unseen` | Fault operators not used during adaptation. | Tests generalization to unseen bug mechanisms. |
| `sparse` | Rare-fault evaluation setting. | Tests whether the policy can expose low-frequency bugs without collapsing task performance. |

For Craftax, `sparse` uses the same operator set as `holdout`, but with lower
episode probability, lower manifestation severity, and longer cooldown. Thus,
Craftax `sparse` is best described as a rare unseen-fault evaluation split.

For MiniGrid, `sparse` samples from all six MiniGrid faults with a low faulty
episode probability. Thus, MiniGrid `sparse` is a rare mixed-fault split.

## Craftax Fault Suite

Craftax contains 15 seeded fault subtypes grouped into three families:

| Family | Purpose |
|---|---|
| `action_exec` | Corrupts the executed action after specific gameplay contexts. |
| `reward_timing` | Delays, drops, or rescales task reward after reward-related contexts. |
| `semantic_high_level` | Corrupts gameplay state updates such as inventory, crafting, station placement, and achievements. |

### Craftax Train / Seen Faults

These are used by `benchmark_train` during adaptation and by `benchmark_seen`
during seen-fault evaluation.

| Family | Fault Type | Faulty Transition |
|---|---|---|
| `action_exec` | `delay_after_success` | After a recent successful transition and action switch, the requested action is delayed by one step. |
| `action_exec` | `remap_after_success_switch` | After a recent successful transition and action switch, the requested action is remapped to another plausible action. |
| `action_exec` | `sticky_after_repeat_switch` | After repeated actions, the next switched action is replaced by the previous executed action. |
| `reward_timing` | `reward_delay_on_positive` | A positive reward is delayed instead of emitted immediately. |
| `reward_timing` | `reward_scale_half_on_positive_switch` | A positive reward is scaled down to half. |
| `semantic_high_level` | `tool_collect_desync_on_upgrade` | A successful resource collection is accepted, but the inventory increase is reverted. |
| `semantic_high_level` | `craft_result_missing_on_retry` | A successful crafting action is accepted, but the crafted output item is missing. |
| `semantic_high_level` | `station_place_ghost_on_relocate` | A station placement is accepted, but the placed table/furnace tile is reverted. |

Default profile-level settings:

| Profile | Episode Probability | Severity | Cooldown |
|---|---:|---:|---:|
| `benchmark_train` | `0.25` in wrapper defaults, `0.5` in current queue | `0.05, 0.1` | `12` |
| `benchmark_seen` | `0.25` in wrapper defaults, `0.5` in current eval queue | `0.05, 0.1` | `8` |

### Craftax Holdout / Unseen Faults

These are not used during adaptation. They are used by `benchmark_holdout`
for unseen-fault evaluation.

| Family | Fault Type | Faulty Transition |
|---|---|---|
| `action_exec` | `revisit_action_delay` | At a revisited position and action switch, the requested action is delayed. |
| `action_exec` | `delayed_switch_failure` | After recent success and repeated actions, switching action causes a delayed failure. |
| `action_exec` | `remap_after_repeat_switch` | After repeated actions, the next switched action is remapped. |
| `reward_timing` | `reward_zero_after_repeat_switch` | A positive reward after repeated behavior is dropped to zero. |
| `reward_timing` | `reward_delay_after_two_rewards` | A positive reward is delayed after multiple recent reward events. |
| `semantic_high_level` | `achievement_unlock_missing_after_valid_progress` | Valid progress occurs, but the corresponding achievement flag is reverted. |
| `semantic_high_level` | `delayed_inventory_desync_after_station_use` | After station/crafting use, an inventory field is corrupted or decremented. |

Default/current evaluation settings:

| Profile | Episode Probability | Severity | Cooldown |
|---|---:|---:|---:|
| `benchmark_holdout` | `0.25` in wrapper defaults, `0.5` in current eval queue | `0.05, 0.1` | `8` |

### Craftax Sparse Faults

Craftax `benchmark_sparse` uses the same operator set as `benchmark_holdout`,
but makes manifestations rarer.

| Profile | Fault Types | Episode Probability | Severity | Cooldown |
|---|---|---:|---:|---:|
| `benchmark_sparse` | Same as holdout/unseen | `0.05` in wrapper defaults, `0.1` in current eval queue | `0.01, 0.03` | `16` |

Paper-facing wording:

> The holdout split evaluates unseen fault operators under normal evaluation
> frequency, while the sparse split evaluates the same unseen operators under
> rare-episode and low-manifestation settings.

## MiniGrid Fault Suite

MiniGrid DoorKey-6x6 is the controlled secondary benchmark. It uses six
interpretable faults adapted to the DoorKey mechanics.

### MiniGrid Train / Seen Faults

| Fault Type | Faulty Transition |
|---|---|
| `broken_door` | A valid door toggle with the key leaves the door locked. |
| `heavy_key` | A valid key pickup leaves the key on the grid. |
| `action_flip` | Left and right rotations execute in the opposite direction. |

These faults are used by `benchmark_train` during adaptation and by
`benchmark_seen` during seen-fault evaluation.

### MiniGrid Holdout / Unseen Faults

| Fault Type | Faulty Transition |
|---|---|
| `teleport` | The agent is moved to another valid cell at the activation step. |
| `door_gone` | The door tile disappears after structural activation. |
| `lava_gap` | Lava appears near the goal after structural activation. |

These faults are used by `benchmark_holdout` and are not exposed during
adaptation.

### MiniGrid Sparse Faults

MiniGrid `benchmark_sparse` samples from all six MiniGrid faults with a low
faulty-episode probability.

| Profile | Fault Types | Episode Probability |
|---|---|---:|
| `benchmark_sparse` | All six MiniGrid faults | `0.1` |

## Recommended Reporting

For both environments, report metrics separately for `clean`, `seen`,
`holdout`, and `sparse`:

- task score retention
- fault manifestation rate
- unique fault type coverage
- time-to-first manifestation
- bug discovery AUC
- fault-score AUROC/AUPRC against manifestation labels
- clean false alarm rate

This split makes the evaluation question explicit: the agent should maintain
game performance while improving discovery of both seen and unseen transition
faults, especially under sparse manifestation.

For paper text, avoid saying that the agent is adapted separately on holdout or
sparse. A safe phrasing is:

> We adapt each method on the `benchmark_train` fault profile and evaluate the
> same adapted checkpoint on seen, holdout, and sparse profiles. Holdout uses
> unseen fault operators, while sparse uses the same holdout operators under
> lower activation probability, lower severity, and longer cooldown.

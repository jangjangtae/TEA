# Craftax Actions and Seeded Operators

The wrapper uses Craftax Classic with 17 discrete actions, numbered 0-16.
NOOP is action 0. Reset is a separate Boolean control. The full action mapping
is in [actions.csv](../results/tea/operators/actions.csv).

## Subtype Mapping

| ID | Split | Wrapper subtype |
| ---: | --- | --- |
| 2 | Train/Seen | `remap_after_success_switch` |
| 3 | Train/Seen | `delay_after_success` |
| 4 | Train/Seen | `sticky_after_repeat_switch` |
| 7 | Train/Seen | `reward_delay_on_positive` |
| 8 | Train/Seen | `reward_scale_half_on_positive_switch` |
| 11 | Holdout/Sparse | `remap_after_repeat_switch` |
| 14 | Holdout/Sparse | `reward_zero_after_repeat_switch` |
| 15 | Holdout/Sparse | `reward_delay_after_two_rewards` |
| 18 | Holdout/Sparse | `revisit_action_delay` |
| 19 | Holdout/Sparse | `delayed_switch_failure` |
| 20 | Train/Seen | `tool_collect_desync_on_upgrade` |
| 23 | Train/Seen | `craft_result_missing_on_retry` |
| 27 | Train/Seen | `station_place_ghost_on_relocate` |
| 31 | Holdout/Sparse | `achievement_unlock_missing_after_valid_progress` |
| 34 | Holdout/Sparse | `delayed_inventory_desync_after_station_use` |

These persistent subtype IDs are not row numbers and must not be renumbered
1-15. ID 0 denotes `none` and is excluded from operator counts. Train/Seen
contain eight subtypes; Holdout/Sparse contain seven. All seven remain in the
Holdout/Sparse coverage denominator, including unobserved subtypes.

## Activation

| Profile | Reported fault-episode probability | Severity choices | Cooldown |
| --- | ---: | --- | ---: |
| Train | 0.5 | 0.05, 0.10 | 12 |
| Seen | 0.5 | 0.05, 0.10 | 8 |
| Holdout | 0.5 | 0.05, 0.10 | 8 |
| Sparse | 0.5 | 0.01, 0.03 | 16 |

A fault-enabled episode selects a family, subtype, and severity. When its
trigger holds and cooldown is clear, a new application is sampled with
`clip(severity * manifestation_multiplier, 0, 1)`. The reported multiplier is
1.0; activation remains stochastic with probability equal to severity.
Selecting the family before the subtype does not sample all 15 subtypes
uniformly. Wrapper defaults and reported overrides are listed separately in
[fault_profiles.csv](../results/tea/operators/fault_profiles.csv).

## Exact Trigger Semantics

[operators.csv](../results/tea/operators/operators.csv) records the actual
trigger, mutation, code function, and naming caveats for each subtype. The
implementation is in [craftax.py](../embodied/envs/craftax.py).

- ID8 requires positive raw reward; it does not test an action switch.
- ID14 checks three identical previous requests and positive reward; it does
  not require the current request to switch.
- ID18's recent-position deque includes the most recently recorded position.
  Leaving and later returning to a cell is not required.
- IDs 20, 23, and 27 do not require extra upgrade, retry, or relocation history
  despite their historical names.
- ID19 delays the requested action by executing NOOP and releasing the
  stored action on the following non-reset call.
- ID34 immediately decrements the first changed inventory field on a crafting
  transition, with a floor at zero. It has no separate delayed scheduler or
  previous station-use predicate.

The operators are manually seeded controlled mutations. They are not a corpus
mined from documented real-world bug reports. The source materials do not map
each subtype to a specific real-game issue or patch.

Operator records can include delayed-action releases and applications that
leave a value unchanged. Eligibility, draw success, new injection, application,
effective state/action/reward change, and independently verified failure are
separate measurements. See [operators.tex](../results/tea/operators/operators.tex)
for table fragments suitable for the manuscript supplement.

# Craftax Fault Seeding Rationale

This note defines how inserted faults should be justified when moving the
Crafter fault benchmark to Craftax. The intended framing is controlled fault
seeding for game testing, not arbitrary reward shaping or making the baseline
policy weak.

## Framing

We use controlled fault seeding because real game bugs are difficult to collect,
reproduce, and label at scale. The clean environment defines the intended game
design, while the faulty environment represents an implementation whose state
transition differs from that design under specific gameplay contexts.

This follows the framing used in automated game testing work where faults are
seeded into game descriptions or sandbox environments to measure whether a
testing agent can expose them. The useful distinction is:

- clean design: the expected game rule or state transition.
- faulty implementation: a controlled corruption of that transition.
- trigger context: the state-action condition where the fault can occur.
- manifestation: the transition is actually corrupted.
- observation/effect: the corrupted state becomes visible in logs, reward,
  observation, or downstream behavior.

For this project, fault seeding is used to evaluate whether a clean world-model
prior can guide an agent toward gameplay-dependent failures while preserving
task competence.

## Literature-Informed Categories

The inserted faults should be explained as common game-testing bug classes:

| Category | Related Game-Testing Framing | Craftax Fault Class |
| --- | --- | --- |
| Fault seeding | Modify game logic or interaction rules to evaluate fault detection rate. | inventory/crafting/station/achievement transition faults |
| Sandbox bug class | Construct controlled environments for exploit, stuck, or navigation bugs. | action execution and revisit-trigger faults |
| Bug platform / bug zoo | Implement common game bugs with enable/disable controls and labels. | independently selectable Craftax fault profiles |
| Artificial proof-of-concept bug | Use controlled artificial faults when full production bugs are unavailable. | diagnostic profiles and forced manifestation tests |
| Real bug benchmark contrast | Real bug datasets provide realism but less control over trigger and labels. | controlled trigger/manifest labels as the tradeoff |

This lets the paper say that the benchmark is synthetic but principled: the
faults are not meant to exhaustively reproduce industrial bug datasets; they
are controlled representatives of gameplay-state-dependent failures.

## Accepted Main Fault Types

The main Craftax benchmark should prioritize faults that violate core
survival/crafting game rules.

### Item Collection Omission

Clean rule: collecting a resource should update the inventory and achievement
state.

Faulty transition: after a valid collection event, the inventory update is
omitted or reverted.

Why it is valid: this represents an interaction/state-update bug where the
event fires but the associated inventory mutation is skipped.

Implemented subtype:

- `tool_collect_desync_on_upgrade`

### Crafting Result Missing

Clean rule: after a valid crafting action near the required station, input
resources are consumed and the crafted item is added.

Faulty transition: the crafted output item is omitted after the game accepts the
crafting transition.

Why it is valid: this models recipe/state-machine bugs and retry-sensitive
crafting failures.

Implemented subtype:

- `craft_result_missing_on_retry`

### Station Placement Ghost

Clean rule: placing a crafting table or furnace should update the map with the
station tile.

Faulty transition: the placement appears accepted, but the station tile is
reverted, creating a ghost placement/state desync.

Why it is valid: this represents object-placement or world-state persistence
bugs common in construction/crafting games.

Implemented subtype:

- `station_place_ghost_on_relocate`

### Achievement Unlock Missing

Clean rule: valid progress should unlock the corresponding achievement.

Faulty transition: the progress transition occurs, but the achievement flag is
not updated.

Why it is valid: this models progression tracking bugs and inconsistent player
state after valid gameplay.

Implemented subtype:

- `achievement_unlock_missing_after_valid_progress`

### Delayed Inventory Desynchronization

Clean rule: station use and crafting should leave inventory in a consistent
post-transition state.

Faulty transition: the inventory is corrupted after a valid station/crafting
transition.

Why it is valid: this models delayed state synchronization bugs, especially in
systems where interaction events and inventory updates are handled by separate
logic.

Implemented subtype:

- `delayed_inventory_desync_after_station_use`

## Auxiliary Diagnostic Faults

Action and reward faults are still useful, but they should be reported as
diagnostic or auxiliary transition faults rather than the strongest benchmark
claim.

Action execution faults:

- `remap_after_success_switch`
- `delay_after_success`
- `sticky_after_repeat_switch`
- `remap_after_repeat_switch`
- `revisit_action_delay`
- `delayed_switch_failure`

Reward timing faults:

- `reward_delay_on_positive`
- `reward_scale_half_on_positive_switch`
- `reward_zero_after_repeat_switch`
- `reward_delay_after_two_rewards`

These are useful for smoke tests and controlled ablations because they exercise
the same logging and fault-score machinery. However, on their own they can look
like action noise or reward corruption, so the main paper narrative should not
depend only on them.

## Recommended Profiles

- `diagnostic`: high-probability forced checks for implementation and logging.
- `benchmark_train`: reachable gameplay-context faults for adaptation.
- `benchmark_seen`: same operators as training with new seeds.
- `benchmark_holdout`: related but unseen operators and harder contexts.
- `benchmark_sparse`: rare manifestation evaluation, not reward tuning.

The profile labels should be used consistently across Crafter and Craftax so
that the environment change is not also a protocol change.

## Reporting Requirements

Do not report only bug count. Report:

- task score retention
- trigger-context count/rate
- manifestation count/rate
- time-to-first trigger
- time-to-first manifestation
- unique fault type coverage
- fault-score AUROC/AUPRC against trigger labels
- fault-score AUROC/AUPRC against manifestation labels
- clean false alarm rate

This keeps the benchmark aligned with QA: the agent should remain competent at
the game while seeking and exposing vulnerable gameplay transitions.

## Paper Wording

A concise paper-facing statement:

> We use controlled fault seeding to evaluate fault-seeking behavior in
> Crafter-like survival/crafting environments. The clean environment defines
> the intended game dynamics, while seeded faulty environments introduce
> state-dependent implementation faults in inventory updates, crafting,
> station placement, achievement tracking, and action/reward timing. Each fault
> records whether the vulnerable context was reached and whether the faulty
> transition manifested, allowing us to evaluate both exploration of vulnerable
> gameplay states and detection using a clean-dynamics world-model prior.

Important limitation:

> These seeded faults are controlled representatives of common gameplay
> transition bugs, not a replacement for an industrial real-bug dataset.
> Controlled seeding lets us precisely label trigger conditions and
> manifestations, which is necessary for evaluating fault-score calibration and
> fault-seeking adaptation.

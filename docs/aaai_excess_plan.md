# AAAI Excess Fault-Seeking Plan

## Positioning

Target claim: a tester should preserve normal game competence while seeking
transitions that violate a clean dynamics prior. The paper should center on the
strongest current result, `contextual_excess_delta_beta02`, instead of expanding
the method family.

CRL/KL-bound variants are not part of the AAAI main story. They can be mentioned
as exploratory related baselines only if needed, but the deadline path is the
excess-delta method and its clean ablations.

## Main Method

Use the frozen clean DreamerV3 world model as a dynamics prior. During
adaptation in fault-seeded Craftax, compute a calibrated fault score from the
reference model and add reward only for contextual excess spikes:

`reward = task_reward + beta * clip(max(0, score_t - score_{t-1} - threshold), 0, clip)`

with clean-context `p95` calibration. This is intentionally framed as
fault-seeking adaptation, not game-score maximization.

More concretely, the method has four parts.

1. **Clean dynamics prior.** Train DreamerV3 in the clean Craftax environment
   and freeze that checkpoint. The frozen model is never updated during
   fault-seeking adaptation. It only provides a reference for what normal
   transitions should look like.

2. **Fault score from model surprise.** On each transition, compare the
   reference world model's normal transition expectation with the transition
   explained after seeing the next observation. The current implementation uses
   the calibrated reference-model fault score already logged by the pipeline,
   based mainly on latent transition surprise and reward prediction error.

3. **Clean calibration.** Before using the score as a reward, run the frozen
   reference policy in the clean environment and estimate normal score
   statistics. The main method uses contextual `p95` normalization, where the
   context is `(action, inventory_bucket, nearby_tile, achievement_stage,
   nearby_mob)`. This matters because some normal actions naturally have larger
   model surprise than others.

4. **Excess-delta reward.** The reward bonus is not the raw fault score. It is
   the clipped excess increase above the clean contextual threshold:

   `bonus_t = beta * clip(max(0, score_t - score_{t-1} - delta_threshold), 0, clip)`

   The final adaptation reward is:

   `r'_t = r_task,t + bonus_t`

   This makes the agent seek sharp, unusual transitions while reducing pressure
   to farm uniformly high-surprise states. Game score is treated as a competence
   constraint/retention metric, not as the primary objective.

## Core Comparison

- `taskonly`: bug-seeded adaptation with only task reward.
- `bugonly_from_scratch`: standard DreamerV3 trained from random
  initialization directly in the bug-seeded environment. This controls for the
  reviewer question of whether a separate frozen clean prior is necessary at
  all, rather than simply learning in the faulty environment. Report both the
  1.0M milestone, which matches the fault-adaptation budget, and the 2.1M final
  checkpoint, which matches the total clean-pretraining plus adaptation budget.
- `dense_beta02`: naive fault-score reward shaping.
- `excess_delta_p95_beta02`: global clean calibration without context.
- `contextual_excess_delta_beta02`: main method.

The core comparison isolates one design decision at a time.

- `taskonly` asks whether Dreamer, by simply playing the game in the bug-seeded
  environment, already discovers the seeded bugs.
- `bugonly_from_scratch` asks whether a single Dreamer trained in the faulty
  environment can replace the clean-prior design. If it overfits to seen faults
  or performs poorly on sparse/holdout faults even with the total 2.1M-step
  budget, it supports the need for a frozen clean dynamics prior. The 1.0M
  milestone is useful but should not be the only comparison because the main
  method also relies on a 1.1M-step clean pretraining stage.
- `dense_beta02` tests the tempting but weaker baseline: directly reward high
  fault score. If this performs worse or hurts task retention, it supports the
  claim that the reward must be shaped carefully rather than treated as generic
  novelty.
- `excess_delta_p95_beta02` keeps the excess-delta idea but removes contextual
  calibration. This tests whether context-aware clean calibration is necessary.
- `contextual_excess_delta_beta02` is the full method: clean dynamics prior,
  context calibration, and excess-delta reward.

## Ablations

- Reward strength: `contextual_excess_delta_beta01`, `beta02`, `beta05`.
- Calibration strictness: `contextual_excess_delta_beta02` vs
  `contextual_excess_delta_p99_beta02`.
- Optional, only if time remains: contextual diversity variants can stay in the
  appendix, not the main story.

The optional sensitivity run is deliberately separated from the core run.
`beta01/beta02/beta05` checks whether the method is robust to reward scale and
whether larger fault-seeking pressure damages game competence. `p99` checks
whether a stricter clean threshold improves precision at the cost of fewer bug
discoveries. These are useful for the appendix, but the main paper should not
look like a broad hyperparameter search.

## Metrics To Report

- Game competence: raw `episode_score_mean` and retention versus the clean
  pretrained reference.
- Bug discovery behavior: bug events per 10k steps, unique bug-type coverage,
  normalized bug discovery AUC, time-to-first-bug.
- Detector quality: AUROC, AUPRC, precision@top-k from fault score versus
  bug-detector labels.
- Generalization: seen, holdout, sparse bug profiles.

## Near-Deadline Schedule

- Before July 21 abstract: finalize story, use existing 3-seed results plus the
  AAAI excess ablation if it completes.
- July 21-24: produce final figures and tables; freeze method variants.
- July 25-28: write and polish; only run small confirmatory jobs if a table is
  clearly missing.

## Commands

Smoke test:

```bash
cd /home/railab/dreamerv3
./dreamerv3/run_craftax_aaai_excess_smoke.sh
```

Detached 3-seed ablation:

```bash
cd /home/railab/dreamerv3
ROOT=/home/railab/logdir/craftax_aaai_excess_ablation_$(date +%Y%m%d_%H%M%S)
setsid env ROOT="$ROOT" ./dreamerv3/run_craftax_aaai_excess_ablation.sh \
  > "$ROOT.launcher.log" 2>&1 < /dev/null &
echo "$ROOT"
```

Default budget: `1,000,000` adaptation steps per seed/variant with milestones at
`200k, 400k, 600k, 800k, 1M`.

Sensitivity run, after the core result is secured:

```bash
cd /home/railab/dreamerv3
ROOT=/home/railab/logdir/craftax_aaai_excess_sensitivity_$(date +%Y%m%d_%H%M%S)
setsid env ROOT="$ROOT" \
  VARIANTS="contextual_excess_delta_beta01 contextual_excess_delta_beta02 contextual_excess_delta_beta05 contextual_excess_delta_p99_beta02" \
  ./dreamerv3/run_craftax_aaai_excess_ablation.sh \
  > "$ROOT.launcher.log" 2>&1 < /dev/null &
echo "$ROOT"
```

Check progress:

```bash
tail -f "$ROOT/launcher.log"
tail -f "$ROOT/status.tsv"
nvidia-smi
```

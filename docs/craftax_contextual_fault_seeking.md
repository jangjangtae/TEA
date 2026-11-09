# Context-Constrained Craftax Fault Seeking

The clean DreamerV3 checkpoint remains a frozen dynamics reference. The
trainable agent receives task reward plus optional fault-seeking terms; bug
labels remain evaluation-only ground truth.

## Context calibration

Each clean transition is assigned a compact context:

```
(action, inventory_bucket, nearby_tile, achievement_stage, nearby_mob)
```

Calibration stores clean fault-score statistics at four levels and uses the
most specific level with at least `fault.context_min_count` samples:

```
full -> action_stage -> action -> global
```

Set `fault.norm_mode=context_p95` to normalize by the selected clean p95.

## Reward terms

- `fault.beta`: contextual fault reward weight.
- `fault.semantic_coverage_beta`: episode first-visit context bonus.
- `fault.unique_suspicious_beta`: first suspicious-context bonus per episode.
- `fault.repeat_suspicious_penalty`: repeated suspicious-context penalty.
- `fault.adaptive_beta`: adjust beta from episode task-return EMA.
- `fault.adaptive_task_target`: absolute task-score constraint used by the
  online adaptive controller.

The policy-prior KL term is intentionally not included yet. It changes the
actor objective and should be evaluated after the reward ablations establish
whether contextual calibration and diversity rewards help on their own.

## Three-day ablation

```bash
cd /home/railab/dreamerv3
ROOT=/home/railab/logdir/craftax_contextual_ablation_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT"
ROOT="$ROOT" nohup ./dreamerv3/run_craftax_contextual_ablation.sh \
  > "$ROOT/launcher.nohup.log" 2>&1 &
```

The queue compares task-only, the previous global-p95 delta reward, contextual
normalization, semantic coverage, unique/repeat control, and adaptive beta over
three seeds and the clean/seen/holdout/sparse splits. Replay buffers are removed
after each completed training run.

## Staged schedule

Use `run_craftax_contextual_pilot.sh` first. It runs all six ablations for one
seed and 400K steps, then writes `analysis/pilot_decision.md` and
`analysis/recommended_variants.txt`. A candidate passes only when task
retention stays above 85%, clean suspicious-rate growth is controlled, and at
least one behavior/fault-seeking signal improves.

The vacation run uses `run_craftax_contextual_weeklong.sh`. With `PILOT_ROOT`
set, it reads the pilot shortlist and runs task-only, the previous global-p95
baseline, and up to two proposed methods for 1M steps over five seeds. Sparse
evaluation is extended to 200K steps.

```bash
PILOT_ROOT=/home/railab/logdir/craftax_contextual_pilot_<timestamp>
ROOT=/home/railab/logdir/craftax_contextual_weeklong_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT"
PILOT_ROOT="$PILOT_ROOT" ROOT="$ROOT" \
  nohup ./dreamerv3/run_craftax_contextual_weeklong.sh \
  > "$ROOT/launcher.nohup.log" 2>&1 &
```

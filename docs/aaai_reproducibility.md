# AAAI Craftax Reproducibility Guide

> Historical AAAI guide. Use [tea_protocol.md](tea_protocol.md) and [tea_results.md](tea_results.md) for the current TEA release.

This guide documents the code paths and commands used for the Craftax
fault-seeking experiments in the AAAI submission draft. The code is based on
DreamerV3, with additional controlled fault seeding, frozen clean-reference
fault scoring, RND and scratch baselines, and analysis scripts.

For the paper-facing adaptation/evaluation protocol, see
`docs/aaai_experimental_protocol.md`. That document explicitly separates clean
pretraining, clean calibration, fault adaptation, and seen/holdout/sparse
evaluation.

## What Is In This Branch

Core method and logging:

- `dreamerv3/fault_score.py`: clean-reference fault scoring, calibration, and
  reward relabeling helpers.
- `dreamerv3/calibrate_fault_score.py`: clean-score calibration entry point.
- `dreamerv3/agent.py`, `dreamerv3/rssm.py`: DreamerV3 hooks for reference
  world-model scoring and fault reward modes.
- `embodied/run/train.py`, `embodied/run/eval_only.py`: training and
  evaluation logging for fault traces.
- `dreamerv3/configs.yaml`: `fault` configuration options.

Craftax benchmark:

- `embodied/envs/craftax.py`: Craftax wrapper, fault profiles, bug labels, and
  RND intrinsic reward baseline.
- `dreamerv3/test_craftax_faults.py`: deterministic checks for Craftax fault
  profiles.
- `docs/fault_suite_split_summary.md`: split definitions and fault taxonomy.
- `docs/craftax_fault_seeding_rationale.md`: paper-facing rationale for
  controlled fault seeding.

Main experiment queues:

- `dreamerv3/run_craftax_multiseed_fault_queue.sh`: shared multi-seed queue for
  clean-init adaptation, RND, scratch, and evaluation.
- `dreamerv3/run_craftax_aaai_excess_ablation.sh`: task-only, dense surprise,
  ExcessDelta, and contextual excess.
- `dreamerv3/run_craftax_rnd_after_gpu_idle.sh`: clean-init Dreamer+RND.
- `dreamerv3/run_craftax_bugonly_after_gpu_idle.sh`: ScratchDreamer from random
  initialization in the faulty environment.
- `dreamerv3/run_craftax_cleaneval_after_rnd.sh`: no-adaptation clean policy
  evaluation.
- `dreamerv3/run_craftax_seed34_all_methods.sh`: current completion run that
  adds seeds 3 and 4 for all seven main methods.

Analysis and figures:

- `dreamerv3/analyze_craftax_multiseed.py`: per-run and aggregate metrics from
  JSONL traces.
- `dreamerv3/analyze_craftax_milestones.py`: learning-curve analysis.
- `dreamerv3/plot_craftax_main_with_rnd.py`: final 7-method main figure.
- `dreamerv3/plot_craftax_learning_curves_compact.py`: compact milestone
  curves.
- `dreamerv3/capture_craftax_fault_images.py`: clean/fault Craftax screenshots
  for the paper figure.

## Environment

The experiments assume a Python 3.11 environment with DreamerV3 dependencies,
JAX GPU support, and Craftax available on `PYTHONPATH`.

The local runs used:

```bash
cd /home/railab/dreamerv3
export PYTHONPATH=/home/railab/dreamerv3/.deps/craftax_pkgs${PYTHONPATH:+:$PYTHONPATH}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib/python3/dist-packages/tensorflow${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

If `libdevice.10.bc` is not found by XLA, the queue scripts create a symlink
from `/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc` when available.

## Craftax Dependency Setup

These experiments use Craftax rather than the original Python Crafter package.
Craftax is a JAX-based Crafter-style benchmark from:

```text
https://github.com/MichaelTMatthews/Craftax
```

The local experiments used Craftax `1.6.1`. To avoid changing the JAX/CUDA
wheel already installed for DreamerV3, install Craftax into a repo-local target
directory and prepend that directory to `PYTHONPATH`:

```bash
cd /home/railab/dreamerv3
mkdir -p .deps/craftax_pkgs

./dreamer_cuda/bin/python -m pip install --target .deps/craftax_pkgs --no-deps \
  craftax==1.6.1 gymnax==0.0.9 gymnasium==1.3.0

export PYTHONPATH=$PWD/.deps/craftax_pkgs${PYTHONPATH:+:$PYTHONPATH}
```

If optional rendering/runtime dependencies are missing, install them in the
same Python environment:

```bash
./dreamer_cuda/bin/python -m pip install pygame imageio matplotlib
```

Verify the package path:

```bash
./dreamer_cuda/bin/python - <<'PY'
import craftax
print("craftax:", getattr(craftax, "__version__", "no-version"), craftax.__file__)
PY
```

Run a small environment and training compile probe before starting long jobs:

```bash
cd /home/railab/dreamerv3
ROOT=/tmp/fate_craftax_probe \
STEPS=512 \
ENVS=1 \
TRAIN_RATIO=1 \
BATCH_SIZE=2 \
BATCH_LENGTH=8 \
REPORT_LENGTH=8 \
REPLAY_SIZE=1000 \
CRAFTAX_ENV_SMOKE=1 \
./dreamerv3/run_craftax_speed_probe.sh
```

The DreamerV3 config block `craftax` uses
`embodied.envs.craftax.Craftax("classic_pixels")`, producing 64x64 RGB
observations with 18 discrete actions. FATE fault profiles and bug labels are
implemented in `embodied/envs/craftax.py`.

## Quick Checks

Run these before long unattended jobs:

```bash
cd /home/railab/dreamerv3

python -m py_compile \
  dreamerv3/fault_score.py \
  dreamerv3/analyze_craftax_multiseed.py \
  dreamerv3/plot_craftax_main_with_rnd.py

python dreamerv3/test_fault_reward_modes.py
python dreamerv3/test_craftax_faults.py

./dreamerv3/run_craftax_aaai_excess_smoke.sh
```

## Clean Checkpoint

All clean-initialized methods use the same clean Craftax checkpoint as both the
trainable agent initialization and the frozen clean-reference world model.

Current local checkpoint:

```text
/home/railab/logdir/craftax_clean_1m_ratio512_saved_20260625_154751/train/ckpt/20260625T233927F611859
```

The checkpoint was trained for about 1.1M clean-environment steps. For a fresh
run, train a clean Craftax DreamerV3 checkpoint and set:

```bash
export TRAIN_ROOT=/path/to/clean_craftax_run_root
```

where `$TRAIN_ROOT/train/ckpt/latest` points to the clean checkpoint.

## Main Methods

The final main comparison uses seven methods:

| Method | Implementation |
|---|---|
| `No-adapt clean` | clean checkpoint, no adaptation, evaluated on each split |
| `Task-only` | clean init, faulty environment, task reward only |
| `ScratchDreamer` | random init in faulty environment, task reward only, 2.1M steps |
| `Dreamer+RND` | clean init, task reward plus RND intrinsic reward |
| `Dense surprise` | clean init, dense clean-reference surprise reward |
| `ExcessDelta` | clean init, global clean p95 excess-delta reward |
| `Contextual excess` | clean init, context-p95 excess-delta ablation |

Bug labels and bug detectors are used only for evaluation and traces. They are
not used as training rewards in the main methods.

## Protocol Summary

The main FATE protocol is:

| Stage | Profile | Update? | Notes |
|---|---|---|---|
| Clean pretraining | clean Craftax | yes | learns the clean policy and clean dynamics prior |
| Clean calibration | clean Craftax | no | estimates seed-specific p95 surprise from clean rollouts |
| Adaptation | `benchmark_train` | active agent only | no bug labels are used as rewards |
| Evaluation | `benchmark_seen` | no | same operators as train, new evaluation rollouts |
| Evaluation | `benchmark_holdout` | no | unseen operators not used during adaptation |
| Evaluation | `benchmark_sparse` | no | same operators as holdout, rarer and lower-severity activation |

The same adapted checkpoint is evaluated on seen, holdout, and sparse splits.
No additional holdout or sparse adaptation is performed.

## Fresh 5-Seed Run

For a fresh 5-seed run, use a separate root for each group. This keeps each
baseline resumable and easier to inspect.

### No-Adapt Clean Policy

```bash
cd /home/railab/dreamerv3
ROOT=/home/railab/logdir/craftax_noadapt_clean_5seed_$(date +%Y%m%d_%H%M%S) \
TRAIN_ROOT=/home/railab/logdir/craftax_clean_1m_ratio512_saved_20260625_154751 \
WAIT_SERVICE= \
RUN_IF_RND_FAILED=1 \
SEEDS="0 1 2 3 4" \
BASE_EVAL_STEPS=30000 \
SPARSE_EVAL_STEPS=60000 \
./dreamerv3/run_craftax_cleaneval_after_rnd.sh
```

### Clean-Init Reward-Design Methods

```bash
cd /home/railab/dreamerv3
ROOT=/home/railab/logdir/craftax_reward_design_5seed_$(date +%Y%m%d_%H%M%S) \
TRAIN_ROOT=/home/railab/logdir/craftax_clean_1m_ratio512_saved_20260625_154751 \
SEEDS="0 1 2 3 4" \
VARIANTS="taskonly dense_beta02 excess_delta_p95_beta02 contextual_excess_delta_beta02" \
TRAIN_STEPS=1000000 \
TRAIN_MILESTONES=1000000 \
TRAIN_ENVS=16 \
TRAIN_RATIO=128 \
REPLAY_SIZE=100000 \
BASE_EVAL_STEPS=30000 \
ADAPT_EVAL_STEPS=30000 \
SPARSE_EVAL_STEPS=60000 \
MILESTONE_EVAL_STEPS=10000 \
MILESTONE_SPARSE_EVAL_STEPS=20000 \
RUN_ANALYSIS=1 \
PRUNE_REPLAY_AFTER_TRAIN=1 \
./dreamerv3/run_craftax_multiseed_fault_queue.sh
```

### Clean-Init RND Baseline

```bash
cd /home/railab/dreamerv3
ROOT=/home/railab/logdir/craftax_rnd_5seed_$(date +%Y%m%d_%H%M%S) \
TRAIN_ROOT=/home/railab/logdir/craftax_clean_1m_ratio512_saved_20260625_154751 \
SEEDS="0 1 2 3 4" \
VARIANTS="rnd_beta005" \
TRAIN_STEPS=1000000 \
TRAIN_MILESTONES=1000000 \
TRAIN_ENVS=16 \
TRAIN_RATIO=128 \
REPLAY_SIZE=100000 \
BASE_EVAL_STEPS=30000 \
ADAPT_EVAL_STEPS=30000 \
SPARSE_EVAL_STEPS=60000 \
MILESTONE_EVAL_STEPS=10000 \
MILESTONE_SPARSE_EVAL_STEPS=20000 \
RUN_ANALYSIS=1 \
PRUNE_REPLAY_AFTER_TRAIN=1 \
./dreamerv3/run_craftax_multiseed_fault_queue.sh
```

### ScratchDreamer Baseline

```bash
cd /home/railab/dreamerv3
ROOT=/home/railab/logdir/craftax_scratch_5seed_$(date +%Y%m%d_%H%M%S) \
TRAIN_ROOT=/home/railab/logdir/craftax_clean_1m_ratio512_saved_20260625_154751 \
SEEDS="0 1 2 3 4" \
VARIANTS="bugonly_from_scratch" \
TRAIN_STEPS=2100000 \
TRAIN_MILESTONES=2100000 \
TRAIN_ENVS=16 \
TRAIN_RATIO=128 \
REPLAY_SIZE=100000 \
BASE_EVAL_STEPS=30000 \
ADAPT_EVAL_STEPS=30000 \
SPARSE_EVAL_STEPS=60000 \
MILESTONE_EVAL_STEPS=10000 \
MILESTONE_SPARSE_EVAL_STEPS=20000 \
RUN_ANALYSIS=1 \
PRUNE_REPLAY_AFTER_TRAIN=1 \
./dreamerv3/run_craftax_multiseed_fault_queue.sh
```

## Current Seed 3/4 Completion Queue

The current machine is running the helper that completes the existing 3-seed
results to 5 seeds by adding only seeds 3 and 4:

```bash
systemctl --user status craftax-seed34-all-methods-20260715.service --no-pager
tail -f /home/railab/logdir/craftax_seed34_all_methods_20260715_152203/status.tsv
tail -f /home/railab/logdir/craftax_seed34_all_methods_20260715_152203/launcher.log
```

The helper script is:

```bash
./dreamerv3/run_craftax_seed34_all_methods.sh
```

It writes:

```text
01_noadapt_clean/
02_cleaninit_reward_design/
03_cleaninit_rnd/
04_scratch_dreamer/
```

under its root.

## Analysis

For each run root, analyze evaluation traces:

```bash
python dreamerv3/analyze_craftax_multiseed.py \
  --root "$ROOT/milestone_1000000" \
  --outdir "$ROOT/analysis/milestone_1000000" \
  --baseline taskonly \
  --eval-only \
  --error-bars sem
```

For ScratchDreamer, use milestone `2100000`.

For no-adaptation clean policy, analyze the `cleaneval` root directly:

```bash
python dreamerv3/analyze_craftax_multiseed.py \
  --root "$CLEANEVAL_ROOT/cleaneval" \
  --outdir "$CLEANEVAL_ROOT/analysis" \
  --baseline reference \
  --eval-only \
  --error-bars sem
```

## Final Figure

`plot_craftax_main_with_rnd.py` accepts comma-separated CSV lists, so existing
seed 0/1/2 CSVs and seed 3/4 CSVs can be combined without moving files.

Example:

```bash
python dreamerv3/plot_craftax_main_with_rnd.py \
  --cleaneval-csv "$CLEAN_012/per_run_metrics.csv,$CLEAN_34/per_run_metrics.csv" \
  --ablation-csv "$ABLATION_012/per_run_metrics.csv,$ABLATION_34/per_run_metrics.csv" \
  --rnd-csv "$RND_012/per_run_metrics.csv,$RND_34/per_run_metrics.csv" \
  --scratch-csv "$SCRATCH_012/per_run_metrics.csv,$SCRATCH_34/per_run_metrics.csv" \
  --outdir "$OUTDIR" \
  --formats png,pdf \
  --error-bars sem \
  --legend-cols 7
```

The four-panel main figure reports:

- task episode return
- bug events per 10K steps
- bug-type coverage
- time to first bug

The extended figure adds:

- normalized discovery AUC
- fault-score ranking AUROC

```bash
python dreamerv3/plot_craftax_main_with_rnd.py \
  --cleaneval-csv "$CLEAN_012/per_run_metrics.csv,$CLEAN_34/per_run_metrics.csv" \
  --ablation-csv "$ABLATION_012/per_run_metrics.csv,$ABLATION_34/per_run_metrics.csv" \
  --rnd-csv "$RND_012/per_run_metrics.csv,$RND_34/per_run_metrics.csv" \
  --scratch-csv "$SCRATCH_012/per_run_metrics.csv,$SCRATCH_34/per_run_metrics.csv" \
  --outdir "$OUTDIR" \
  --formats png,pdf \
  --error-bars sem \
  --extended \
  --legend-cols 7
```

## Paper Interpretation

The no-adaptation clean policy should be described as a strong passive tester,
not as a weak baseline. Many seeded Craftax faults are embedded in normal
progression-dependent gameplay, so a competent clean policy naturally exposes
some faults. The main claim is that ExcessDelta preserves this competence while
actively biasing adaptation toward calibrated deviations from normal clean
dynamics, with the clearest advantage under sparse fault activation.

# FATE

> Historical README retained for context. The current method is [TEA](../README.md); use the current protocol and result tables.

**Fault-seeking Adaptation with Transition Excess**

FATE is a DreamerV3-based framework for game testing agents that preserve
normal gameplay competence while actively seeking transitions that deviate from
clean game dynamics. The current implementation targets Craftax/Crafter-style
long-horizon environments with controlled fault seeding, clean-reference world
model scoring, and evaluation-only bug labels.

This repository is based on the public DreamerV3 implementation and adds the
fault-seeking components used in our Craftax experiments.

## Core Idea

FATE separates normal gameplay knowledge from fault-seeking adaptation:

1. Train a DreamerV3 agent in a clean environment.
2. Reuse the clean checkpoint in two roles:
   - a trainable agent initialized from clean gameplay behavior,
   - a frozen clean-reference world model that is never updated.
3. In the faulty environment, score each transition by how surprising it is
   under the frozen clean-reference dynamics.
4. Calibrate surprise using clean-environment statistics.
5. Reward only calibrated transition excess, so the agent is biased toward
   fault-revealing behavior without using manual bug rewards.

Bug detectors and seeded bug labels are reserved for evaluation and analysis.
They are not used as training rewards in the main FATE objective.

## Main Methods

The Craftax experiments compare the following agents:

| Method | Description |
| --- | --- |
| No-adapt clean | Clean checkpoint evaluated directly in faulty environments |
| Task-only | Clean initialization, task reward only |
| ScratchDreamer | Random initialization in the faulty environment, task reward only |
| Dreamer+RND | Clean initialization plus RND intrinsic reward |
| Dense surprise | Clean initialization plus dense clean-reference surprise |
| FATE | Global clean-p95 transition excess reward |
| Contextual excess | Context-conditioned calibration ablation |

## Repository Layout

Key FATE additions:

- `dreamerv3/fault_score.py`: clean-reference surprise scoring and calibration
  helpers.
- `dreamerv3/calibrate_fault_score.py`: clean-score calibration entry point.
- `dreamerv3/configs.yaml`: fault reward, Craftax, RND, and evaluation config
  options.
- `embodied/envs/craftax.py`: Craftax wrapper, seeded fault profiles, bug
  labels, and RND baseline support.
- `dreamerv3/analyze_craftax_multiseed.py`: trace analysis for per-run and
  aggregate metrics.
- `dreamerv3/plot_craftax_main_with_rnd.py`: main paper figure with FATE,
  RND, scratch, and ablation baselines.
- `docs/aaai_reproducibility.md`: detailed experiment commands and current
  reproducibility notes.
- `docs/aaai_experimental_protocol.md`: paper-facing protocol for clean
  pretraining, calibration, adaptation, and evaluation splits.
- `docs/fault_suite_split_summary.md`: Craftax fault taxonomy and split
  definitions.
- `docs/craftax_fault_seeding_rationale.md`: rationale for controlled fault
  seeding.

Original DreamerV3 code remains under `dreamerv3/` and `embodied/`.

## Environment

The experiments were run with Python 3.11, JAX GPU support, and Craftax on
`PYTHONPATH`.

```bash
cd /path/to/FATE
pip install -U -r requirements.txt

export PYTHONPATH=/path/to/FATE/.deps/craftax_pkgs${PYTHONPATH:+:$PYTHONPATH}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
```

If XLA cannot find `libdevice.10.bc`, check the CUDA path used by
`XLA_FLAGS`. The local queue scripts include safeguards for this issue.

## Craftax Setup

The experiments use Craftax, not the original Python Crafter environment.
Craftax is a JAX-based reimplementation and extension of Crafter. The local
runs used Craftax `1.6.1` from:

```text
https://github.com/MichaelTMatthews/Craftax
```

Recommended setup is to keep the Dreamer/JAX CUDA environment fixed and install
Craftax into a repo-local dependency directory so it does not overwrite the JAX
wheel used by DreamerV3:

```bash
cd /path/to/FATE
mkdir -p .deps/craftax_pkgs

python -m pip install --target .deps/craftax_pkgs --no-deps \
  craftax==1.6.1 gymnax==0.0.9 gymnasium==1.3.0

export PYTHONPATH=$PWD/.deps/craftax_pkgs${PYTHONPATH:+:$PYTHONPATH}
```

If your environment is missing optional runtime packages, install them in the
main environment:

```bash
python -m pip install pygame imageio matplotlib
```

Check that Craftax is visible:

```bash
python - <<'PY'
import craftax
print("craftax:", getattr(craftax, "__version__", "no-version"), craftax.__file__)
PY
```

Run a small DreamerV3/Craftax probe:

```bash
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

The FATE wrapper uses the Dreamer config block `craftax`, which creates
`embodied.envs.craftax.Craftax("classic_pixels")` with 64x64 RGB observations
and 18 discrete actions.

## Quick Checks

Before long runs, compile the main scripts and run the small fault tests:

```bash
python -m py_compile \
  dreamerv3/fault_score.py \
  dreamerv3/analyze_craftax_multiseed.py \
  dreamerv3/plot_craftax_main_with_rnd.py

python dreamerv3/test_fault_reward_modes.py
python dreamerv3/test_craftax_faults.py
./dreamerv3/run_craftax_aaai_excess_smoke.sh
```

## Reproducing the Craftax Experiments

Detailed commands are provided in:

```text
docs/aaai_reproducibility.md
```

The paper-facing protocol is summarized in:

```text
docs/aaai_experimental_protocol.md
```

The high-level workflow is:

1. Train or provide a clean Craftax DreamerV3 checkpoint.
2. Run clean-score calibration.
3. Run clean-initialized adaptation variants:
   `taskonly`, `dense_beta02`, `excess_delta_p95_beta02`, and
   `contextual_excess_delta_beta02`.
4. Run `rnd_beta005` and `bugonly_from_scratch` baselines.
5. Analyze JSONL traces and generate figures.

Example final-figure command:

```bash
python dreamerv3/plot_craftax_main_with_rnd.py \
  --cleaneval-csv /path/to/cleaneval/per_run_metrics.csv \
  --ablation-csv /path/to/reward_design/per_run_metrics.csv \
  --rnd-csv /path/to/rnd/per_run_metrics.csv \
  --scratch-csv /path/to/scratch/per_run_metrics.csv \
  --outdir /path/to/figures \
  --formats png,pdf \
  --error-bars sem \
  --legend-cols 7
```

## Evaluation Metrics

The main Craftax results track:

- task episode return,
- bug events per 10k environment steps,
- bug-type coverage,
- time to first bug.

Additional analysis includes discovery AUC, fault-score ranking AUROC/AUPRC,
bug-normal score gaps, clean false alarms, and split-wise generalization across
seen, holdout, and sparse fault settings.

## Notes on Checkpoints and Logs

Large checkpoints, replay buffers, and experiment logs are not included in the
repository. The analysis scripts expect local JSONL traces or generated
`per_run_metrics.csv` files from completed runs.

## DreamerV3 Attribution

This codebase builds on DreamerV3:

```bibtex
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

DreamerV3 resources:

- [Research paper][paper]
- [Project website][website]
- [Original repository](https://github.com/danijar/dreamerv3)

## Disclaimer

This repository is a research branch for fault-seeking game testing experiments.
It is not an official DreamerV3 release and is unrelated to Google or DeepMind.

[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3

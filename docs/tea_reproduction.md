# TEA Reproduction Guide

This guide covers result recomputation, new Dreamer training, reward controls,
fixed-policy evaluation, and optional replay initialization. Run commands
from the repository root in Bash. Long training commands are examples to
launch deliberately, not installation tests.

## Available Artifacts

| Goal | Required artifacts | Available here |
| --- | --- | --- |
| Recompute tables, intervals, and figures | Derived CSVs and analysis dependencies | Yes |
| Inspect settings and operators | YAML, calibration JSON, and source | Yes |
| Train a new Clean model and adapt it | CUDA runtime and compute | Code and examples, but no original weights |
| Evaluate an adapted policy | Final checkpoint; reference and calibration for diagnostic scoring | No released checkpoint |
| Restore saved Clean experience | Matching checkpoint, replay chunks, and local manifest | Restore code, but no replay chunks |
| Recreate arbitrary transition analyses | Original detailed traces | Not all traces are released |

A newly trained Clean model is a new initialization, not a bitwise
reconstruction of the paper's P0. The released CSVs remain measurements from
the completed experiments. Do not substitute new measurements while retaining
the original P0 identity. Five adaptations of one P0 are not five independent
pretraining seeds.

## Recompute Without a GPU

```bash
git clone https://github.com/jangjangtae/TEA.git
cd TEA
python3.11 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
python -m dreamerv3.plot_tea_final_paper \
  --core results/tea/core \
  --replay results/tea/replay/results \
  --boundary results/tea/first_transition/results \
  --out outputs/paper_figures
python results/tea/replay/scripts/recompute.py \
  --bundle results/tea/replay --out outputs/replay_recomputed
python -m unittest dreamerv3.test_plot_tea_final_paper dreamerv3.test_plot_fate_tog_figures
python -m unittest discover -s results/tea/replay/scripts -p test_recompute.py -v
```

The figure command verifies aggregates and intervals before exporting five
600-dpi PNGs and vector PDFs, plus captions and verification files. See
[result definitions](tea_results.md) for the different Clean-repeat schedules.
This route needs no checkpoint, Craftax, JAX, or PyTorch. Use a separate
environment for training.

## Install the Dreamer Runtime

The reference platform is Linux with Python 3.11 and an NVIDIA GPU. Dreamer
runs on CUDA; Craftax stepping uses CPU JAX. The working installation checked
on 2026-11-09 used JAX/JAXlib/CUDA plugins 0.4.33, NumPy 1.26.4, Craftax 1.6.1,
Gymnax 0.0.9, and Gymnasium 1.3.0. Key compatibility constraints are in
[tea-runtime.txt](../constraints/tea-runtime.txt). They are not a complete
transitive lockfile or a guarantee for every GPU and operating system.

After deactivating the analysis environment:

```bash
python3.11 -m venv .venv-tea
source .venv-tea/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints/tea-runtime.txt -r requirements.txt \
  'flax==0.8.5' 'ruamel.yaml==0.19.1' pygame imageio seaborn
python -m pip install --target .deps/craftax_pkgs --no-deps \
  craftax==1.6.1 gymnax==0.0.9 gymnasium==1.3.0
export PYTHONPATH="$PWD:$PWD/.deps/craftax_pkgs${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
nvidia-smi
python -c 'import jax; print(jax.__version__, jax.devices())'
python -m pip check
```

The `--no-deps` Craftax installation deliberately avoids upgrading JAX. Flax,
Pygame, ImageIO, Matplotlib, and Seaborn must still be installed in the active
environment. Driver compatibility is machine-specific: consult the
[official JAX guide](https://docs.jax.dev/en/latest/installation.html).
Do not apply its current-version upgrade command blindly to this older runtime.
The `nvidia-smi` CUDA label does not prove that the Python runtime works.

Check source imports and reward/operator contracts without training:

```bash
JAX_PLATFORMS=cpu python dreamerv3/test_fault_reward_modes.py
JAX_PLATFORMS=cpu python -m unittest -v dreamerv3.test_fate_operator_contract
python -m unittest -v tools.test_run_tea_config tools.test_public_docs
```

Use `embodied` from this checkout, not a different PyPI package. A successful
CPU contract test does not establish GPU training throughput or performance.

## Environment Variables

Use a fresh terminal or clear inherited Craftax overrides before selecting a
profile. A YAML does not contain these environment switches:

```bash
for name in "${!CRAFTAX_@}"; do unset "$name"; done
unset TEA_NO_FIRST_TRANSITION_BONUS FATE_INITIAL_REPLAY_MANIFEST TEA_AUDIT_FRESH_SCHEDULE
export CRAFTAX_FAULT=0 CRAFTAX_SEMANTIC_FAULT=0
export CRAFTAX_FAULT_SAMPLER=0 CRAFTAX_USE_RND=0 CRAFTAX_RND_UPDATE=0
```

For Fault adaptation, select the Train profile explicitly:

```bash
export CRAFTAX_FAULT_SAMPLER=1 CRAFTAX_FAULT_PROFILE=benchmark_train
export CRAFTAX_FAULT_EP_PROB=0.5 CRAFTAX_FAULT_MANIFEST_PROB=1.0
export CRAFTAX_FAULT_SEED=0
```

For Clean adaptation keep `CRAFTAX_FAULT_SAMPLER=0`. Manifest multiplier 1.0
does not make every eligible fault activate; profile severity still affects
the draw. See [operator definitions](tea_operators.md).

## Execute an Archived YAML

`dreamerv3.main --configs` accepts named presets, not YAML filenames. The
launcher below translates a full archived YAML into the same CLI flags,
validates types, and checks that the settings round-trip through the schema:

```bash
python tools/run_tea_config.py configs/tea/P0_pretraining.yaml \
  --set logdir=outputs/my_clean_pretraining
```

This only prints a command. Append `--execute` to launch it. Repeat
`--set dotted.key=value` for paths, seeds, budgets, and other existing fields.
Quote paths containing spaces. The launcher does not infer environment
variables, find weights, or download artifacts. Unknown fields fail.

### Pretraining and a Bounded Smoke

With faults and RND disabled, this deliberately short run exercises a fresh
model and updates, but is not a usable gameplay checkpoint:

```bash
python tools/run_tea_config.py configs/tea/P0_pretraining.yaml \
  --set logdir=outputs/smoke_clean --set run.steps=2048 --execute
```

Allow GPU compilation time. Short interaction budgets are not necessarily
short in seconds. For full 1.1M pretraining use a different, empty logdir:

```bash
python tools/run_tea_config.py configs/tea/P0_pretraining.yaml \
  --set logdir=outputs/my_clean_pretraining --execute
```

Pretraining uses ratio 512 and replay capacity 200K. Adaptation uses ratio 128
and capacity 100K; it is not an exact pretraining continuation. No checkpoint
is automatically downloaded.

### Select a Completed Checkpoint

`LOGDIR/ckpt/latest` is a text file naming a timestamped save. Use the completed
timestamped directory, not `latest`, `agent.pkl`, or the outer experiment folder:

```bash
export P0_CHECKPOINT="$PWD/outputs/my_clean_pretraining/ckpt/$(tr -d '\r\n' < outputs/my_clean_pretraining/ckpt/latest)"
test -f "$P0_CHECKPOINT/done"
test -f "$P0_CHECKPOINT/agent.pkl"
```

The agent payload includes optimizer state. Only load trusted checkpoints:
these formats use Python pickle. Keep the frozen reference checkpoint unchanged
throughout adaptation and evaluation.

### Calibrate Clean Surprise

Disable fault sampling and RND, then collect clean statistics with the same P0:

```bash
CRAFTAX_FAULT=0 CRAFTAX_FAULT_SAMPLER=0 CRAFTAX_USE_RND=0 CRAFTAX_RND_UPDATE=0 \
python -m dreamerv3.calibrate_fault_score \
  --configs craftax size1m --ref_ckpt "$P0_CHECKPOINT" \
  --steps 30000 --out outputs/my_clean_stats.json \
  --logdir outputs/my_calibration --run.envs 1 --seed 0 --jax.platform cuda \
  --fault.enabled True --fault.w_kl 1.0 --fault.w_reward 0.0
export CLEAN_STATS="$PWD/outputs/my_clean_stats.json"
```

For a new P0, recompute its statistics. Released
`configs/tea/calibration/seed_N.json` files describe the archived P0, not an
arbitrary new checkpoint. Preserve the calibration seed, duration, p95,
reference hash, and resulting file. The scale stays fixed during adaptation.

### TEA Adaptation

Select the training profile as above. For seed 0:

```bash
python tools/run_tea_config.py configs/tea/FaultAdaptFATE/seed_0_train.yaml \
  --set logdir=outputs/my_tea_seed_0 \
  --set run.from_checkpoint="$P0_CHECKPOINT" \
  --set fault.ref_ckpt="$P0_CHECKPOINT" --set fault.norm_stats="$CLEAN_STATS" \
  --execute
```

For an adaptation smoke use a new logdir and add `--set run.steps=2048`.
Every condition and adaptation seed independently restores P0 and its
optimizer; none starts from the previously adapted policy. The default protocol
starts with empty replay. Keep a unique logdir per condition and seed.

## Reward Controls

Use the matching `CleanAdapt...` or `FaultAdapt...` YAML from
[config_index.csv](../configs/tea/config_index.csv) in the command above.
Match profile, P0, calibration, batch, ratio, budget, and evaluation schedule.
Do not assign TEA's beta to every comparator.

| Method suffix | Reward setting | Additional requirement |
| --- | --- | --- |
| `TaskOnly` | Log-only shaping, beta 0 | RND off |
| `FATE` (TEA) | `excess_delta_threshold`, beta 0.2 | RND off |
| `DenseSurprise` | Absolute normalized score, `dense` | Its matched beta, not TEA beta |
| `SparseSurprise` | `excess_threshold` | Calibrated threshold, clip, and beta together |
| `SparseMagnitude072` | Sparse shape, beta scaled by 0.72 | Scale sensitivity, not guaranteed matched payout |
| `AbsoluteSameShape` | Absolute level, threshold 0.5, clip 2, beta 0.2 | Distinct from matched SparseSurprise |
| `RND` | Wrapper-provided bonus; TEA log-only | Enable RND explicitly |

The original Fault-adapt RND logs used this alpha for all five adaptation
seeds. Clean-adapt RND used seed-specific values instead; both sets are in
[rnd_environment.csv](../configs/tea/rnd_environment.csv), checked against
the original training log headers. For Fault-adapt seed 0:

```bash
export CRAFTAX_USE_RND=1 CRAFTAX_RND_UPDATE=1
export CRAFTAX_RND_ALPHA=0.0021406749162549126
export CRAFTAX_RND_NORM=1 CRAFTAX_RND_CLIP=5.0 CRAFTAX_RND_DOWNSAMPLE=4
export CRAFTAX_RND_HIDDEN_DIM=128 CRAFTAX_RND_OUTPUT_DIM=128 CRAFTAX_RND_LR=0.005
export CRAFTAX_RND_SEED=0
```

Choose `FaultAdaptRND/seed_0_train.yaml` and the same path overrides. Change
the RND seed and YAML together for seeds 1-4. For Clean adaptation, use its
CSV row's alpha as well as its YAML and disable the fault sampler. Disable
RND and predictor updates
for evaluation and all non-RND conditions. Probe matching does not guarantee
matched on-policy payout. Reusing archived coefficients with a new P0 is
fixed-hyperparameter transfer, not a newly calibration-matched study. Do not
substitute the Clean-adapt calibration constants into the completed
Fault-adapt condition.

## Fixed Policy Evaluation

Select the final adapted timestamped checkpoint using the same `ckpt/latest`
procedure and export `ADAPTED_CHECKPOINT`. It is the active policy;
`P0_CHECKPOINT` stays the frozen diagnostic reference. Do not swap them.
For adaptation seed 0, the three Sparse repetitions are:

```bash
unset TEA_NO_FIRST_TRANSITION_BONUS FATE_INITIAL_REPLAY_MANIFEST TEA_AUDIT_FRESH_SCHEDULE
for offset in 0 10000 20000; do
  CRAFTAX_FAULT=0 CRAFTAX_FAULT_SAMPLER=1 \
  CRAFTAX_FAULT_PROFILE=benchmark_sparse CRAFTAX_FAULT_EP_PROB=0.5 \
  CRAFTAX_FAULT_MANIFEST_PROB=1.0 CRAFTAX_FAULT_SEED="$offset" \
  CRAFTAX_USE_RND=0 CRAFTAX_RND_UPDATE=0 \
  python -m dreamerv3.main --configs craftax size1m \
    --script eval_only --seed "$offset" --run.envs 1 --run.steps 60000 \
    --logdir "outputs/my_tea_seed_0_eval_sparse_$offset" \
    --run.from_checkpoint "$ADAPTED_CHECKPOINT" \
    --jax.platform cuda --jax.compute_dtype float32 \
    --fault.enabled True --fault.log_only True \
    --fault.ref_ckpt "$P0_CHECKPOINT" --fault.norm_stats "$CLEAN_STATS" \
    --fault.w_kl 1.0 --fault.w_reward 0.0 --fault.norm_mode p95 \
    --fault.task_reward_source delivered
done
```

For adaptation seed `s`, use evaluation seeds `s`, `s+10000`, and `s+20000`
for both `--seed` and `CRAFTAX_FAULT_SEED`. Holdout uses `benchmark_holdout`.
Clean disables fault sampling and uses 30K callbacks. The primary cohort has
one Clean evaluation per policy; replay and first-transition cohorts have
three. Seen is not part of every released cohort's common schedule.

Repeat independent adaptation and evaluation for seeds 0-4. Use the matching
seed YAML, fault/RND seed, calibration choice, and distinct directories. Three
evaluation repetitions do not make five trained policies into 15 training
seeds. Average repetitions inside policy before five-seed means, SDs, and
paired differences. See [the aggregation rules](tea_results.md).

Evaluation performs no updates and adds no shaping bonus to task reward.
Diagnostic bonus fields may still be logged. Training episode scores can
include intrinsic reward; do not use them as pure task return.

## First Transition Ablation

Use `FaultAdaptTEANoFirstBonus/seed_N_train.yaml` and set
`TEA_NO_FIRST_TRANSITION_BONUS=1` during training. Keep the remaining TEA
settings and evaluation matched. Only the first non-reset bonus is removed;
the surprise score is retained for subsequent differencing. The flag is not
encoded by the YAML and must be unset for evaluation. The config index
records this distinction.

## Restore Saved Experience

This optional comparison needs matching Clean replay `.npz` chunks; neither
the public CSVs nor an agent checkpoint contains them. Skip this section
without those files. Do not fabricate a manifest from released provenance
identifiers or import replay from another P0.

With matching files, prepare a local hashed manifest:

```bash
export P0_REPLAY=/data/my_clean_pretraining/replay
export REPLAY_MANIFEST="$PWD/outputs/my_replay_manifest.json"
python - <<'PY'
import os
from pathlib import Path
from dreamerv3.fate_replay_restore import select_chunks, REPLAY_ONLY_PROTOCOL
from dreamerv3.fate_diagnostic_followup import digest, write_json

checkpoint = Path(os.environ['P0_CHECKPOINT']).resolve()
chunks = select_chunks(Path(os.environ['P0_REPLAY']).resolve(), 65, 100000)
spec = dict(protocol=REPLAY_ONLY_PROTOCOL, checkpoint=str(checkpoint),
    agent_sha256=digest(checkpoint / 'agent.pkl'), length=65, capacity=100000,
    matched_train_start=2048,
    historical_rewards='original_clean_task_reward_no_relabel',
    files={str(p.resolve()): digest(p) for p in chunks})
write_json(Path(os.environ['REPLAY_MANIFEST']), spec)
PY
python -m dreamerv3.fate_replay_restore \
  --manifest "$REPLAY_MANIFEST" --out outputs/replay_validation
```

Validation copies replay and samples a batch, requiring disk space but no
training. Use a fresh validation directory. The loader checks agent/chunk
hashes, fields, image/action shapes, and sampleable capacity. A manifest is
machine-local because it contains actual absolute paths.

For training set `FATE_INITIAL_REPLAY_MANIFEST="$REPLAY_MANIFEST"`, choose
`FaultRestoredTaskOnly/seed_N_train.yaml` or `FaultRestoredTEA/seed_N_train.yaml`,
and apply the same P0, reference, and calibration overrides as before. Use an
empty logdir: initial import is not resume. Historical Clean rewards keep
their original values with no retrospective TEA bonus. New data begins in
new worker streams. Capacity stays 100K, ratio 128, batch 16 x 64, environments
16, and first update callback 2048. Check `replay_restore_audit.json` and
`replay_schedule_audit.json`; the complete 1M run should have 124,745 updates.
The schedule audit is specific to this contract, not arbitrary budgets/ratios.

This restores capacity-sized saved experience, not exact live FIFO order,
environment state, recurrent carry, or sampler RNG. It is not the earlier
joint-restoration pilot with ratio 512 and capacity 200K.

## Outputs and Restart Behavior

| Output | Purpose |
| --- | --- |
| `config.yaml` | Resolved settings |
| `metrics.jsonl`, `scores.jsonl` | Logger outputs; distinguish reward channels |
| `fault_score_trace.jsonl` | Score, reward, and transition diagnostics when enabled |
| `bug_trace.jsonl` | Fixed-policy evaluation transitions and wrapper events |
| `ckpt/latest`, `ckpt/<timestamp>/done` | Latest completed training save |
| `replay/` | Saved experience, not an agent checkpoint |
| Replay audit JSONs | Restore integrity and matched update schedule |

Keep stdout/stderr too. An existing Dreamer logdir can cause
`cp.load_or_save()` to resume its own checkpoint instead of independently
starting from P0. Use a new directory for every seed, method, and smoke. For
an ordinary deliberate resume, retain the same directory, dependencies,
configuration, and original reference; `run.steps` is the total target, not
extra steps added after restart. Initial replay import refuses an existing
`ckpt/latest`; do not repeat the import as though it were a resume.

Check the final completed checkpoint and process exit status. Plain
`dreamerv3.main` does not create the lab queue's `DONE.json`. Nonempty logs
or GPU utilization alone do not imply completion.

## Troubleshooting and Compute

- CPU-only JAX cannot run `--jax.platform cuda`. Check the active interpreter,
  driver, JAXlib, and matching CUDA plugins first.
- `numpy.dtypes` errors can indicate a different system Python or incompatible
  NumPy/JAX versions. Do not mix analysis and training environments.
- Missing Craftax, Gymnax, or textures can mean the `.deps` target is absent
  from `PYTHONPATH` or a Craftax dependency is missing.
- Missing checkpoint `done`/`agent.pkl` usually means the wrong directory or
  an incomplete save was supplied.
- `artifacts/...` is historical provenance, not a downloadable path. Override
  it; the launcher does not resolve or download it.
- A wrong active objective requires checking both YAML and the inherited
  `CRAFTAX_*`, `TEA_*`, and `FATE_*` environment settings.
- Keep long outputs on a large data disk. These examples do not start
  unattended queues, delete replay, or guarantee memory/disk capacity. Measure
  memory before running full seeds concurrently.

Interaction budgets are not wall-clock guarantees. GPU compilation, CPU
stepping, reference scoring, logging, and replay I/O matter. Time a bounded
pilot on the target hardware before scheduling five seeds. Configuration
and contract tests on the working environment do not certify a clean-machine
full-training rerun.

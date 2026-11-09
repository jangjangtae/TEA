# WUJI Craftax Ports

These are external system comparators, not exact reproductions of the
original WUJI experiments and not matched Dreamer reward ablations. Both
ports train independent policy populations from scratch. They do not reuse
TEA's P0, frozen reference, surprise score, or replay.

## Mechanisms and Differences

| Component | WUJI-derived DDQN | WUJI-inspired A2C |
| --- | --- | --- |
| Learner | Explicit CUDA Double DQN | Stable-Baselines3 A2C |
| Population | Default 30 | Default 30 |
| Evolution | Upstream NSGA-II, weight crossover, evolved epsilon | Upstream NSGA-II and weight crossover, Gaussian weight mutation |
| Selection pool | Parents and trained offspring | Parents P, mutated P-prime, and trained Q |
| Policy input | RGB 64 x 64 | RGB 64 x 64 |
| Action space | 17 requested actions | Same |
| Learner reward | Delivered environment task reward only | Same |
| Selection objectives | Task reward per 1,000 actions and mean distinct grid positions per episode segment | Same |

Coordinates are privileged selection information, not policy observations.
Fault labels and Holdout/Sparse outcomes do not enter training or selection.
Spatial position coverage is not TEA's coarse context-action count. Equal
nominal budgets do not equalize architecture, update counts, policy switches,
pretraining reuse, or reset accounting. A whole-system contrast cannot isolate
the causal effect of having a world model.

The public WUJI configuration uses DDQN; the paper's A2C/Gaussian-mutation
description motivated the separate A2C port. Craftax-specific changes include
a spatial CNN, task coefficient fixed at one instead of a BlockMaze-specific
reward genotype, synchronous single-GPU generations, fixed learning segments,
and fresh learner/optimizer state for each inherited offspring. Neither port
recreates the original asynchronous game-specific schedule.

## Separate Environment and Upstream Checkout

Work from the TEA root. Use a separate Python 3.11 environment so PyTorch,
SB3, and upstream WUJI dependencies do not replace Dreamer's packages:

```bash
python3.11 -m venv .venv-wuji
source .venv-wuji/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints/tea-runtime.txt -r requirements.local.txt \
  'jax==0.4.33' 'jaxlib==0.4.33' 'flax==0.8.5' 'ruamel.yaml==0.19.1' \
  pygame imageio seaborn
python -m pip install --target .deps/craftax_pkgs --no-deps \
  craftax==1.6.1 gymnax==0.0.9 gymnasium==1.3.0
git clone https://github.com/NeteaseFuxiRL/wuji.git .deps/wuji_upstream
git -C .deps/wuji_upstream checkout --detach de7161c29c59c4cc9ffb729d8c19dfb52e37a2c6
```

The launcher fixes `JAX_PLATFORMS=cpu` for Craftax; neural learning can still
use PyTorch CUDA. A Linux CUDA 12.6 wheel example corresponding to the checked
working Torch family is:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c constraints/tea-runtime.txt \
  -r .deps/wuji_upstream/requirements.txt 'stable-baselines3==2.9.0'
export PYTHONPATH="$PWD:$PWD/.deps/craftax_pkgs:$PWD/.deps/wuji_upstream${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m pip check
python tools/run_wuji_craftax.py --help
python tools/run_wuji_a2c.py --help
```

Use the [official PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/)
for a driver-compatible build; do not assume the CUDA label in `nvidia-smi`
is sufficient. The observed working environment used Torch 2.11.0+cu126
and SB3 2.9.0. The source/dependency identity is captured in new checkpoints;
changing SB3 files or source after training can intentionally reject evaluation
or resume. These instructions are not a full cross-platform lockfile.

Do not edit the pinned upstream checkout. `source_identity()` checks its
revision and clean worktree, the wrapper, port, and launcher hashes. Keep
upstream's GPL-3.0-or-later license and notices; it is not vendored here. The
Dreamer MIT notice does not replace WUJI's license. See
[third-party notices](../THIRD_PARTY_NOTICES.md).

The launchers also recognize the original optional `.deps/torch311cu126`,
`.deps/wuji_probe_pkgs`, and `.deps/meta_rl_pkgs` targets. A fresh installation
does not need those targets when the same packages are installed in its venv.
Do not mix old target packages with a newly installed venv unintentionally.

## Bounded Smoke Commands

These commands perform real short training only when explicitly executed.
They are pipeline checks, not evidence of competence or fault-detection ability.
Use a new output directory for every independent attempt. Compilation can
dominate wall time. `--device cuda:0` fails if CUDA is unavailable; there is
no silent learner fallback to CPU. Tests can deliberately use `--device cpu`.

DDQN, population three and two complete generations:

```bash
python tools/run_wuji_craftax.py train --outdir outputs/wuji_ddqn_smoke \
  --population 3 --max-env-steps 3840 --learn-steps 256 \
  --fitness-steps 256 --fitness-rollouts 1 --batch-size 16 \
  --learning-starts 16 --train-every 16 --target-update 4 \
  --replay-capacity 512 --episode-length 64 --device cuda:0
python tools/run_wuji_craftax.py eval \
  --checkpoint outputs/wuji_ddqn_smoke/checkpoint.pt \
  --outdir outputs/wuji_ddqn_smoke_eval \
  --profiles clean seen holdout sparse --steps 384 --device cuda:0
```

A2C, population three and one complete P/P-prime/Q generation:

```bash
python tools/run_wuji_a2c.py train --outdir outputs/wuji_a2c_smoke \
  --population 3 --max-env-steps 300 --learn-steps 40 --n-steps 20 \
  --fitness-steps 20 --fitness-rollouts 1 --episode-length 64 --device cuda:0
python tools/run_wuji_a2c.py eval \
  --checkpoint outputs/wuji_a2c_smoke/checkpoint.pt \
  --outdir outputs/wuji_a2c_smoke_eval \
  --profiles clean seen holdout sparse --steps 384 --device cuda:0
```

## Full Population Runs

The full DDQN example explicitly sets the 2.1M budget. The CLI defaults alone
do not reproduce that completed protocol:

```bash
python tools/run_wuji_craftax.py train --outdir outputs/wuji_ddqn_seed_0 \
  --seed 0 --population 30 --train-profile train \
  --max-env-steps 2100000 --learn-steps 10400 \
  --fitness-steps 3000 --fitness-rollouts 3 \
  --train-every 128 --updates-per-collect 1 --device cuda:0
```

Initial fitness costs 90K actions; five generations cost 5 x 30 x
(10,400 learning + 3,000 fitness). Total: 2.1M actions, with 1.56M learning
and 540K fitness actions. This is a population-total budget, not 2.1M per
policy. A separate increased-update DDQN condition uses
`--updates-per-collect 8` in a different output directory. That multiplies
updates per collection opportunity, not replay capacity or environment steps.
Do not merge these two conditions under one result label.

For A2C:

```bash
python tools/run_wuji_a2c.py train --outdir outputs/wuji_a2c_seed_0 \
  --seed 0 --population 30 --train-profile train \
  --max-env-steps 2100000 --learn-steps 7400 --n-steps 20 \
  --fitness-steps 3000 --fitness-rollouts 3 --device cuda:0
```

Its initial 90K plus five generations of 30 x (7,400 learning + two 3,000
fitness passes) also totals 2.1M. This is 1.11M learning and 990K fitness
actions. The learner budgets are therefore different even at equal total
actions. A2C rejects budgets that cannot fit whole generations and rollout
lengths that would overrun a learning segment. DDQN ends before an incomplete
generation; always check actual committed counters.

Repeat complete independent population runs with seeds 0-4 and unique
directories. Each population seed is the inference unit, not its 30 members.
Training and fitness use `train` or explicitly `clean`, never held-out profiles.
The adapter isolates inherited `CRAFTAX_*` switches and fixes seeded-profile
episode probability to 0.5, manifest multiplier to 1, and RND off.

## Fixed Population Evaluation

For DDQN seed 0, use separate directories for the two evaluation budgets:

```bash
python tools/run_wuji_craftax.py eval \
  --checkpoint outputs/wuji_ddqn_seed_0/checkpoint.pt \
  --outdir outputs/wuji_ddqn_seed_0_clean \
  --profiles clean --steps 30000 --seed 200000 --device cuda:0
python tools/run_wuji_craftax.py eval \
  --checkpoint outputs/wuji_ddqn_seed_0/checkpoint.pt \
  --outdir outputs/wuji_ddqn_seed_0_faults \
  --profiles seen holdout sparse --steps 60000 --seed 200000 --device cuda:0
```

For A2C replace the launcher and checkpoint/output paths. Each profile's
`--steps` budget is split equally across the final population in deterministic
policy-ID order. It is not given to every member, and no best member is chosen
using Holdout/Sparse. Evaluation is greedy, with no learning or bonuses;
weights and checkpoint identity are checked. The original external-comparator
schedule used seed `200000 + 100 * population_seed`, then adds population
slot inside the evaluator. This differs from Dreamer's repeat schedule.

Population switching adds episode resets. WUJI counts action transitions
separately from reset calls; Dreamer counts driver callbacks. Nominal 2.1M
labels alone do not establish exact sample matching. Operator event recording
uses the same wrapper channel, not an independent general failure oracle.

## Checkpoints and Recovery

`checkpoint.pt` atomically saves the population, pending candidates, stage,
cursor, counters, RNG states, configuration, and source/dependency identity.
Only load trusted checkpoints because PyTorch pickle is used. `summary.json`
records `status`, `total_action_steps`, `train_steps`, `fitness_steps`,
`gradient_updates`, candidate history, and runtime/device information.
Evaluation creates one JSON per profile plus an immutable-weight summary.

To pause at a committed boundary, add `--stop-after-candidates N` to training.
Resume with the same training command and all the same hyperparameters,
append `--resume`, and omit the pause flag to finish. An in-flight candidate
is repeated after a crash; repeated compute is not a new seed. `run.steps`
is not a WUJI option: use `--max-env-steps`. Evaluation requires a fully
selected `ready` population, not a checkpoint halfway through initial fitness
or an offspring stage. Output locks reject concurrent writes to the same run.

The default minimum-free-space checks are 10 GiB for DDQN and 20 GiB for A2C.
Do not lower them simply to bypass a capacity warning. DDQN replay is bounded
by `--replay-capacity`, but population blobs, environment JIT caches, Torch
allocations, and candidate data also consume RAM/VRAM. Start sequentially and
measure memory before considering concurrent seeds. Low GPU utilization can
reflect CPU environment/fitness work, not a failed CUDA learner.

This public repository contains the comparator code, not a complete WUJI
result/checkpoint release. Do not describe the commands as reproducing an
included WUJI table or the original WUJI paper's game results.

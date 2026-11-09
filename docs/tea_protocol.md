# TEA Experiment Protocol

TEA reuses a gameplay checkpoint as both the active agent initialization and
a frozen clean-reference model. Adaptation uses Clean or `benchmark_train`.
Holdout and Sparse are fixed-policy evaluations with no updates or intrinsic
reward added to the delivered task return.

For a first installation, follow the [reproduction guide](tea_reproduction.md)
first. It includes virtual environments, additional Craftax dependencies,
checkpoint-directory selection, saved-YAML execution, smoke commands, replay
manifest preparation, resuming, and troubleshooting. Optional external
comparators have a [separate WUJI guide](wuji_craftax_port.md).

## Runtime Setup

The archived runs used Python 3.11, JAX with CUDA support, and Craftax 1.6.1.
The existing `requirements.txt` retains the Dreamer runtime dependencies.
Install Craftax separately to preserve the selected JAX environment:

```bash
python -m pip install -c constraints/tea-runtime.txt -r requirements.txt \
  'flax==0.8.5' 'ruamel.yaml==0.19.1' pygame imageio seaborn
python -m pip install --target .deps/craftax_pkgs --no-deps \
  craftax==1.6.1 gymnax==0.0.9 gymnasium==1.3.0
export PYTHONPATH="$PWD:$PWD/.deps/craftax_pkgs${PYTHONPATH:+:$PYTHONPATH}"
```

The analysis-only installation in the main README does not require this CUDA
environment. Optional WUJI ports additionally require their upstream checkout,
PyTorch, and, for A2C, Stable-Baselines3; consult the module headers and
`THIRD_PARTY_NOTICES.md`.

## Preparation and Adaptation

The following commands illustrate the TEA runtime contract. `P0_CHECKPOINT`
must name an existing compatible clean Dreamer checkpoint, including its agent
and optimizer state. Exact archived configurations for each method and seed
are indexed in [config_index.csv](../configs/tea/config_index.csv).

To train a compatible clean checkpoint from scratch, the archived P0 setup
uses 16 environments, 1.1M aggregate callbacks, train ratio 512, and replay
capacity 200K:

```bash
CRAFTAX_FAULT=0 CRAFTAX_FAULT_SAMPLER=0 CRAFTAX_USE_RND=0 CRAFTAX_RND_UPDATE=0 \
python -m dreamerv3.main --configs craftax size1m \
  --script train --seed 0 --logdir outputs/clean_pretraining \
  --run.steps 1100000 --run.envs 16 --run.train_ratio 512 --replay.size 200000 \
  --batch_size 16 --batch_length 64 --report_length 32 \
  --jax.platform cuda --jax.prealloc True --jax.compute_dtype float32 \
  --jax.compilation_cache False --jax.autotune 0 --jax.prefetch False \
  --jax.profiler False --jax.donate_train False --fault.enabled False
```

Use the resulting checkpoint directory for `P0_CHECKPOINT`. A newly trained
checkpoint is a new initialization; it is not the archived P0 weight artifact.

Calibrate in Clean with no fault sampler:

```bash
CRAFTAX_FAULT=0 CRAFTAX_FAULT_SAMPLER=0 \
python -m dreamerv3.calibrate_fault_score \
  --configs craftax size1m --ref_ckpt "$P0_CHECKPOINT" \
  --steps 30000 --out outputs/clean_stats.json \
  --logdir outputs/calibration --run.envs 1 --jax.platform cuda \
  --fault.enabled True --fault.w_kl 1.0 --fault.w_reward 0.0
```

Adapt in the seeded Train profile:

```bash
CRAFTAX_FAULT=0 CRAFTAX_FAULT_SAMPLER=1 \
CRAFTAX_FAULT_PROFILE=benchmark_train CRAFTAX_FAULT_EP_PROB=0.5 \
CRAFTAX_FAULT_MANIFEST_PROB=1.0 CRAFTAX_FAULT_SEED=0 \
CRAFTAX_USE_RND=0 CRAFTAX_RND_UPDATE=0 \
python -m dreamerv3.main --configs craftax size1m \
  --script train --seed 0 --logdir outputs/tea_train \
  --run.from_checkpoint "$P0_CHECKPOINT" --run.steps 1000000 \
  --run.envs 16 --run.train_ratio 128 --replay.size 100000 \
  --batch_size 16 --batch_length 64 --report_length 32 \
  --jax.platform cuda --jax.prealloc True --jax.compute_dtype float32 \
  --jax.compilation_cache False --jax.autotune 0 --jax.prefetch False \
  --jax.profiler False --jax.donate_train False \
  --fault.enabled True --fault.log_only False \
  --fault.ref_ckpt "$P0_CHECKPOINT" --fault.norm_stats outputs/clean_stats.json \
  --fault.norm_mode p95 --fault.score_source latent_reward \
  --fault.task_reward_source delivered --fault.w_kl 1.0 --fault.w_reward 0.0 \
  --fault.reward_mode excess_delta_threshold --fault.beta 0.2 \
  --fault.reward_delta_threshold 0.5 --fault.clip 2.0 --fault.reward_gate none
```

For Clean adaptation set `CRAFTAX_FAULT_SAMPLER=0`. TaskOnly uses
`fault.log_only=True`, `fault.reward_mode=dense`, and `fault.beta=0`.
Other reward controls use their archived method/seed configuration rather
than an arbitrary shared bonus coefficient. Probe matching and coefficient
scaling do not guarantee matched realized on-policy bonus budgets.

Each condition starts independently from P0. Clean adaptation is not a
preparatory training stage for the subsequent fault-adapt condition. The
primary protocol starts with empty replay; it is an adaptation protocol rather
than an exact continuation of pretraining.

For the boundary ablation set `TEA_NO_FIRST_TRANSITION_BONUS=1` only during
training. The first non-reset transition's bonus is zero, while its normalized
surprise score is retained for subsequent differencing. Saved-replay experiments
also need `FATE_INITIAL_REPLAY_MANIFEST` and the schedule audit described in
the replay source. The archived YAML alone does not encode these environment
variables or contain the replay data.

## Fixed-Policy Evaluation

Set `ADAPTED_CHECKPOINT` to the final policy checkpoint and use the same
checkpoint for all repeats:

```bash
CRAFTAX_FAULT=0 CRAFTAX_FAULT_SAMPLER=1 \
CRAFTAX_FAULT_PROFILE=benchmark_sparse CRAFTAX_FAULT_EP_PROB=0.5 \
CRAFTAX_FAULT_MANIFEST_PROB=1.0 CRAFTAX_FAULT_SEED=10000 \
CRAFTAX_USE_RND=0 CRAFTAX_RND_UPDATE=0 \
python -m dreamerv3.main --configs craftax size1m \
  --script eval_only --seed 10000 --logdir outputs/eval_sparse \
  --run.from_checkpoint "$ADAPTED_CHECKPOINT" --run.steps 60000 --run.envs 1 \
  --jax.platform cuda --jax.compute_dtype float32 \
  --fault.enabled True --fault.log_only True --fault.ref_ckpt "$P0_CHECKPOINT" \
  --fault.norm_stats outputs/clean_stats.json --fault.norm_mode p95 \
  --fault.w_kl 1.0 --fault.w_reward 0.0 --fault.task_reward_source delivered
```

Unset `TEA_NO_FIRST_TRANSITION_BONUS` and `FATE_INITIAL_REPLAY_MANIFEST`
for evaluation. Clean evaluation disables the fault sampler and uses 30K
callbacks; Holdout uses `benchmark_holdout` and 60K callbacks. The three repeat
seeds are adaptation seed plus 0, 10000, and 20000. See the result guide for
which cohorts have one versus three Clean evaluations.

## Measurement and Scope

- Task return is the mean episode delivered environment reward, without the
  adaptation bonus.
- Context-action count is an observed coarse-context repertoire, not map area
  or code coverage.
- Subtype exposure is recorded operator coverage with denominator seven for
  Holdout/Sparse. Repetitions are averaged within policy, not combined by union.
- Operator application, effective action/reward/state change, and independently
  checked failure are distinct outcomes.
- Five adaptation seeds from one P0 do not constitute five independent
  pretrained policies. CleanEval slots are repeated evaluations of one policy.

Current operator triggers and activation probabilities are specified in
[tea_operators.md](tea_operators.md). Older `aaai_*` documents and the historical
README describe earlier experiments and are not the current protocol.

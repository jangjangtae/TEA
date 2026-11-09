# TEA

**Temporal-Excess Adaptation for test-oriented reuse of game-playing agents and world models.**

TEA initializes a trainable DreamerV3 agent and a frozen reference from a clean
gameplay checkpoint. A clean-calibrated bonus rewards positive temporal increases
in the reference model's posterior-prior latent divergence. We study how this
adaptation changes game performance, observed behavior breadth, and seeded
operator exposure in Craftax Classic.

The method was previously called **FATE**. Historical code names, environment
variables, and result identifiers such as `FaultAdaptFATE` are preserved for
traceability; their paper-facing method name is TEA.

## Start Here

- [Protocol and execution examples](docs/tea_protocol.md)
- [Step-by-step reproduction and troubleshooting](docs/tea_reproduction.md)
- [WUJI DQN and A2C setup and execution](docs/wuji_craftax_port.md)
- [Completed results and evaluation scope](docs/tea_results.md)
- [Fault operators and action IDs](docs/tea_operators.md)
- [Saved per-run configurations](configs/tea/config_index.csv)
- [Publication figures and captions](figures/tea)
- [Source and result provenance](results/tea/SOURCE_MANIFEST.json)

## Recompute the Published Results

Result analysis does not require a GPU, checkpoint, or replay buffer. From the
repository root, install the plotting dependencies and regenerate the five
figures:

```bash
python -m pip install -r requirements-analysis.txt
python -m dreamerv3.plot_tea_final_paper \
  --core results/tea/core \
  --replay results/tea/replay/results \
  --boundary results/tea/first_transition/results \
  --out outputs/paper_figures
python -m unittest dreamerv3.test_plot_tea_final_paper dreamerv3.test_plot_fate_tog_figures
```

The plotter reaggregates individual evaluations and checks policy means, summary
statistics, and paired intervals against the archived tables before exporting
600-dpi PNGs and vector PDFs. Captions and plotted values are separate files.

The replay sensitivity tables also have a standard-library-only recomputation:

```bash
python results/tea/replay/scripts/recompute.py \
  --bundle results/tea/replay --out outputs/replay_recomputed
python -m unittest discover -s results/tea/replay/scripts -p 'test_recompute.py' -v
```

## Method and Environment

Craftax Classic has **17 discrete actions, IDs 0-16 including NOOP**, and the
Dreamer wrapper provides 64 x 64 x 3 RGB observations. Episode reset is a separate
Boolean environment-control flag.

For clean-reference surprise `s_t` and fixed clean p95 scale `tau`, TEA uses:

```text
normalized_score_t = s_t / tau
delta_t = max(0, normalized_score_t - normalized_score_(t-1))
bonus_t = 0.2 * clip((delta_t - 0.5) / (2.0 - 0.5), 0, 1)
learning_reward_t = delivered_environment_reward_t + bonus_t
```

The reference receives requested actions and observations and remains frozen.
The active agent, including its reward head, learns from the augmented reward.
Reference-score history is separate for each environment worker and resets at
episode boundaries. In the first-transition ablation, the first non-reset
bonus is zeroed while its normalized surprise score remains the previous score
for the next temporal difference.

The controlled fault suite contains 15 manually seeded operators in three
families. Train/Seen use eight subtypes; Holdout/Sparse use seven disjoint
subtypes. Fault labels are evaluation instrumentation and do not supply the
TEA bonus. [The specification](docs/tea_operators.md) distinguishes eligibility,
stochastic activation, operator application, effective change, and verified
failure.

## Results and Their Scope

The primary comparison starts from one P0 clean checkpoint and includes five
adaptation seeds per condition. Its common evaluation schedule is one 30K
Clean run and three 60K runs each for Holdout and Sparse per policy. Repetitions
are averaged inside policy before summarizing adaptation seeds. CleanEval slots
evaluate one fixed policy.

Under fault-seeded adaptation, TEA's mean Clean task return was 3.759 versus
3.122 for TaskOnly, and its observed context-action count was 970.8 versus
712.2. Sparse subtype exposure was 66.67% versus 64.76%; Holdout exposure was
77.14% versus 78.10%. These results support separating gameplay and behavior
outcomes from testing outcomes. They do not establish consistent superiority
in exposure breadth or speed.

The release also contains completed five-seed replay-initialization and
first-transition-bonus analyses, each with three evaluations per policy and
profile. Their Clean-return means use a different repetition scope from the
primary table. See [the result guide](docs/tea_results.md) before combining
tables. All results remain conditional on the evaluated Craftax setting;
operator exposure is not a general failure-detection oracle.

## Runtime Code

| File | Role |
| --- | --- |
| `dreamerv3/fault_score.py` | Reference surprise normalization and reward shaping |
| `embodied/run/train.py` | Trainable agent, frozen reference, worker-specific score history |
| `embodied/run/eval_only.py` | Fixed-policy evaluation |
| `embodied/envs/craftax.py` | Craftax wrapper and the 15 seeded operators |
| `dreamerv3/tea_no_first_bonus.py` | First non-reset bonus mask |
| `dreamerv3/fate_replay_restore.py` | Saved clean-experience replay initialization |
| `dreamerv3/analyze_fate_behavior_diversity.py` | Coarse context-action measurements |
| `dreamerv3/plot_tea_final_paper.py` | Verified final-result figures |

The historical shell queues and AAAI notes remain available as development
history. Current paper definitions are in the `docs/tea_*.md` files.
WUJI-derived DQN and WUJI-inspired A2C Craftax ports are external system
comparators; they are not matched Dreamer reward ablations. Their code is in
`dreamerv3/wuji_craftax.py` and `dreamerv3/wuji_a2c_craftax.py`; see
[third-party notes](THIRD_PARTY_NOTICES.md).

## Attribution and Artifacts

This repository builds on [DreamerV3](https://github.com/danijar/dreamerv3) and
uses [Craftax](https://github.com/MichaelTMatthews/Craftax). The upstream
DreamerV3 license is preserved in [LICENSE](LICENSE).

The published tables contain policy/evaluation-level derived measurements.
Model weights, replay chunks, raw transition JSONL, virtual environments, and
laboratory-specific experiment directories are not included. Archived
configuration paths beginning with `artifacts/` identify original local
artifacts; replace them with your own paths when running an experiment.

The [reproduction guide](docs/tea_reproduction.md) separates analysis of the
released measurements from new training. It explains checkpoint selection,
archived-YAML commands, reward controls, multi-seed evaluation, optional replay
restoration, output checks, and the limits of the released artifacts. The
configuration launcher prints a validated command by default; training starts
only when `--execute` is supplied. WUJI launchers and their upstream checkout
requirements are documented separately.

This is a research extension and is not an official DreamerV3 release.

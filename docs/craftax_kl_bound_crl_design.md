# DreamerV3 KL-Bound and Constrained Novelty Probe

## Source methods

`Novelty Detection in Reinforcement Learning with World Models` (ICML 2025)
detects a novelty when the Dreamer latent posterior-prior KL violates a bound
constructed by dropping recurrent history. In DreamerV3 notation:

```
L_t = KL(q(z_t | h_t, x_t) || p(z_t | h_t))
B_t = KL(q(z_t | h_t, x_t) || p(z_t | h_0))
    - KL(q(z_t | h_t, x_t) || q(z_t | h_0, x_t))
v_t = L_t - B_t
```

The paper's threshold-free decision is `v_t > 0`. We log both the signed
violation and `max(v_t, 0)` as a reward-compatible score.

`Direct Behavior Specification via Constrained Reinforcement Learning`
(ICML 2022) treats desired behavior frequencies as CMDP constraints and adapts
their Lagrange multipliers instead of manually searching reward weights.

## Craftax adaptation

The research objective is expressed as:

```
maximize    J_novelty(pi)
subject to  J_task(pi) >= d_task
```

The initial probe uses an online episodic estimate and an additive dual update:

```
lambda <- clip(lambda + lr * (d_task - EMA(task_return)), 0, lambda_max)
r_train = r_novelty + lambda * r_task
```

The first episodes use task reward only as a bootstrap. This avoids the
multiplicative collapse observed in the previous adaptive-beta experiment.

This is a primal-dual reward-relabel probe, not yet the full multi-critic
algorithm from the CRL paper. A positive result justifies adding separate
Dreamer task and novelty value heads for the formal CMDP implementation.

## Interpretation boundary

The novelty paper studies permanent inference-time shifts. Craftax faults are
conditional and can manifest intermittently. The KL-bound implementation is
therefore an architecture-level reproduction and a bug-detection baseline, not
an exact reproduction of the paper's novelty protocol.

## One-day decision criteria

- KL-bound clean trigger rate should be low without a tuned threshold.
- KL-bound AUROC/AUPRC should improve on holdout or sparse faults over latent KL.
- Constrained training must retain at least 85% of paired task-only score.
- The multiplier must react to violations without saturating at 0 or max.
- Bug encounter or unique suspicious-context coverage must improve.

Only methods satisfying the task gate and at least one fault-seeking criterion
are eligible for the week-long multi-seed run.

## Execution stages

1. `run_craftax_paper_crl_oneday.sh` performs the detector comparison and a
   300K-step seed-0 adaptation screen.
2. Inspect task retention, clean false-positive rate, held-out AUROC/AUPRC,
   multiplier behavior, and bug-context coverage.
3. `run_craftax_paper_crl_weeklong.sh` confirms only the surviving methods.
   Its conservative default is five seeds and 800K adaptation steps; set the
   `VARIANTS` environment variable to remove rejected methods.

The long queue deletes each completed replay directory and refuses to start a
new job below 80 GB free disk space. The clean reference checkpoint and run
checkpoints are retained.

For an unattended full-week window,
`run_craftax_paper_crl_vacation.sh` runs six-seed 800K confirmation followed by
a longer three-seed detector replication. Its phases have separate status
entries and continue independently by default. Based on the measured 70-hour
three-seed predecessor, the default queue is expected to occupy roughly
150--160 hours and use about 50--60 GB after replay pruning.

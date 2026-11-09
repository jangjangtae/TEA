# MiniGrid Fault Suite

## Purpose

MiniGrid DoorKey-6x6 is the controlled secondary environment for the Craftax
fault-seeking study. It reproduces the six environment changes described in
`Novelty Detection in Reinforcement Learning with World Models` while adding
rare episode activation and exact transition-level labels.

## Observation Modes

- `minigrid`: a 153-dimensional normalized vector containing the partial
  7x7 compact grid, direction, and carried object. This is the fast development
  and ablation configuration.
- `minigrid_vision`: a 64x64 RGB rendering of the same partial observation.
  This is closer to the visual world-model setup in the source paper.

Both modes share identical layouts, fault sampling, actions, rewards, and
ground-truth labels.

## Faults

| Fault | Family | Trigger and effect |
|---|---|---|
| `broken_door` | interaction | A valid toggle with the yellow key leaves the door locked. |
| `heavy_key` | interaction | A valid pickup leaves the key on the grid. |
| `action_flip` | action | Left and right rotation execute in the opposite direction. |
| `teleport` | state | At the activation step the agent moves to another valid cell before its action executes. |
| `door_gone` | layout | The door disappears after the structural activation step; the first visible transition is labeled. |
| `lava_gap` | layout | Lava appears beside the goal after the structural activation step; the first visible transition is labeled. |

Structural faults default to activation after step 5. This prevents the clean
reference model from absorbing the fault as part of the initial observation.

## Profiles

- `clean`: no faults.
- `benchmark_train` and `benchmark_seen`: broken door, heavy key, action flip.
- `benchmark_holdout`: teleport, door gone, lava gap.
- `benchmark_sparse`: all faults with 10% faulty episodes.
- `diagnostic`: all faults with every episode faulty.

The environment variables `MINIGRID_FAULT_PROFILE`, `MINIGRID_FAULT_TYPE`,
`MINIGRID_FAULT_EP_PROB`, and `MINIGRID_FAULT_MANIFEST_PROB` override config
defaults.

## Validation

Run the CPU-only suite with:

```bash
./dreamerv3/run_minigrid_smoke.sh
```

Once the GPU is free, compare symbolic and visual throughput with:

```bash
CONFIG=minigrid ./dreamerv3/run_minigrid_speed_probe.sh
CONFIG=minigrid_vision ./dreamerv3/run_minigrid_speed_probe.sh
```

MiniGrid 3.1.0 and its Gymnasium/Pygame dependencies are isolated under
`.deps/minigrid_pkgs`; the existing Dreamer Python environment is unchanged.

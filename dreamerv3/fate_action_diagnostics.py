"""Read-only Craftax action diagnostics and bounded, reset-aligned captures."""

import functools
import json
import os
from pathlib import Path

import elements
import jax
import numpy as np

from embodied.envs.craftax import Craftax


PREFIX = 'log/diag_'
STATE_FIELDS = ('player_energy', 'player_health', 'player_food', 'player_drink',
                'player_direction', 'is_sleeping')
INVENTORY = ('wood', 'stone', 'coal', 'iron', 'diamond', 'sapling',
             'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
             'wood_sword', 'stone_sword', 'iron_sword')


def enabled():
  return os.environ.get('FATE_ACTION_DIAGNOSTICS', '0') == '1'


def state_scalars(state):
  with jax.transfer_guard('allow'):
    out = {k: int(np.asarray(getattr(state, k))) for k in STATE_FIELDS}
    out.update({f'inv_{k}': int(np.asarray(getattr(state.inventory, k))) for k in INVENTORY})
    position = np.asarray(state.player_position)
    out.update(y=int(position[0]), x=int(position[1]))
  return out


class DiagnosticCraftax(Craftax):
  """Observe the original step without modifying its state, action, or RNG."""

  @functools.cached_property
  def obs_space(self):
    spaces = dict(super().obs_space)
    names = [f'{side}_{k}' for side in ('before', 'after')
             for k in (*STATE_FIELDS, *(f'inv_{k}' for k in INVENTORY), 'y', 'x')]
    names += ['valid', 'requested_action', 'env_action', 'effective_action',
              'sleep_started', 'sleep_ended', 'placement_succeeded',
              'front_before', 'front_after', 'wood_consumed', 'stone_consumed']
    spaces.update({PREFIX + k: elements.Space(np.int32) for k in names})
    return spaces

  def step(self, action):
    reset = bool(action['reset']) or self._done
    before = self._state
    obs = super().step(action)
    fields = {key: np.int32(0) for key in self.obs_space if key.startswith(PREFIX)}
    after_values = state_scalars(self._state)
    before_values = after_values if reset else state_scalars(before)
    for side, values in [('before', before_values), ('after', after_values)]:
      fields.update({PREFIX + side + '_' + k: np.int32(v) for k, v in values.items()})
    if not reset:
      requested = int(action['action'])
      env_action = int(self._last_fault_info['env_action'])
      # craftax_step masks every command to NOOP when already sleeping.
      effective = 0 if before_values['is_sleeping'] else env_action
      front = self._front_position(before)
      fb = fa = -1
      if self._position_in_bounds(before, front):
        with jax.transfer_guard('allow'):
          fb, fa = int(np.asarray(before.map[front])), int(np.asarray(self._state.map[front]))
      target = {7: 4, 8: 11, 9: 12, 10: 15}.get(effective, -2)
      values = dict(valid=1, requested_action=requested, env_action=env_action,
          effective_action=effective,
          sleep_started=int(not before_values['is_sleeping'] and after_values['is_sleeping']),
          sleep_ended=int(before_values['is_sleeping'] and not after_values['is_sleeping']),
          placement_succeeded=int(fa == target and fb != target),
          front_before=fb, front_after=fa,
          wood_consumed=max(0, before_values['inv_wood'] - after_values['inv_wood']),
          stone_consumed=max(0, before_values['inv_stone'] - after_values['inv_stone']))
      fields.update({PREFIX + k: np.int32(v) for k, v in values.items()})
    return dict(obs, **fields)


class Recorder:
  """Capture one worker-0 episode per fixed bin, always starting at reset.

  Full reset prefixes support recurrent rescoring; GIFs are bounded excerpts,
  not a replacement for the source observations. No random sampling is used.
  """

  def __init__(self, directory, training):
    self.root = Path(str(directory)) / 'action_diagnostics'
    self.root.mkdir(parents=True, exist_ok=True)
    self.training = training
    self.bin_size = 100000 if training else 2000
    self.bins = set()
    self.frames, self.rows = [], []
    self.active = False
    self.episode = 0
    self.last_step = 0

  def __call__(self, tran, worker, step):
    if worker != 0:
      return
    if tran['is_first']:
      self.flush(False)
      self.episode += 1
      bucket = (step - 1) // self.bin_size
      self.active = bucket not in self.bins
      if self.active:
        self.bins.add(bucket)
    if not self.active:
      return
    image = np.asarray(tran['image'])
    row = {k: np.asarray(v).item() for k, v in tran.items()
           if np.asarray(v).ndim == 0 and k != 'image'}
    row.update(global_step=int(step), worker=worker, episode_id=self.episode)
    self.frames.append(image.copy())
    self.rows.append(row)
    if tran['is_last']:
      self.flush(True)
    elif len(self.rows) >= 10001:
      self.flush(False)

  def flush(self, complete):
    if not self.rows:
      self.active = False
      return
    stem = self.root / f'episode_{self.episode:05d}'
    images = np.stack(self.frames)
    np.savez_compressed(str(stem) + '.npz', image=images,
        outgoing_action=np.array([r['action'] for r in self.rows], np.int32),
        incoming_action=np.array([r['log/requested_action'] for r in self.rows], np.int32),
        is_first=np.array([r['is_first'] for r in self.rows], bool),
        is_last=np.array([r['is_last'] for r in self.rows], bool),
        is_terminal=np.array([r['is_terminal'] for r in self.rows], bool),
        task_reward=np.array([r['log/task_reward_delivered'] for r in self.rows], np.float32))
    with stem.with_suffix('.jsonl').open('w') as handle:
      for row in self.rows:
        handle.write(json.dumps(row, allow_nan=False) + '\n')
    suspects = [i for i, r in enumerate(self.rows)
                if r.get(PREFIX + 'valid') and r.get(PREFIX + 'requested_action') == 6
                and r.get(PREFIX + 'before_player_energy') == 9]
    centers = (suspects[:1] or [min(25, len(images) - 1)])
    from PIL import Image
    for center in centers:
      lo, hi = max(0, center - 25), min(len(images), center + 26)
      frames = [Image.fromarray(x).resize((256, 256), Image.Resampling.NEAREST)
                for x in images[lo:hi]]
      frames[0].save(str(stem) + f'_steps_{lo}_{hi}.gif', save_all=True,
          append_images=frames[1:], duration=100, loop=0)
    stem.with_suffix('.json').write_text(json.dumps(dict(
        rows=len(self.rows), complete_episode=bool(complete),
        begins_at_reset=bool(self.rows[0]['is_first']),
        selection='First reset in each fixed global-step bin, worker 0',
        suspect_indices=suspects, global_step_start=self.rows[0]['global_step'],
        global_step_end=self.rows[-1]['global_step']), indent=2) + '\n')
    self.frames, self.rows, self.active = [], [], False

  def close(self):
    self.flush(False)


def save_milestone(cp, step, logdir):
  targets = [int(x) for x in os.environ.get('FATE_DIAGNOSTIC_MILESTONES', '').split(',') if x]
  for target in targets:
    path = Path(str(logdir)) / 'diagnostic_checkpoints' / f'step_{target}'
    if int(step) >= target and not (path / 'done').exists():
      cp.save(str(path), keys=['agent', 'step'])
      (path / 'milestone.json').write_text(json.dumps(dict(
          target_step=target, actual_step=int(step), purpose='diagnostic evaluation only',
          resumable_training=False)) + '\n')

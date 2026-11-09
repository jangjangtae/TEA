import functools
import os
import sys
from pathlib import Path

import elements
import embodied
import numpy as np
from PIL import Image


def _import_minigrid():
  try:
    import gymnasium as gym
    import minigrid  # noqa: F401
  except ImportError:
    local = Path(__file__).resolve().parents[2] / '.deps' / 'minigrid_pkgs'
    if local.exists() and str(local) not in sys.path:
      sys.path.insert(0, str(local))
    try:
      import gymnasium as gym
      import minigrid  # noqa: F401
    except ImportError as exc:
      raise ImportError(
          'MiniGrid is not installed. Run `pip install minigrid`, or install '
          'it under `.deps/minigrid_pkgs`.') from exc
  return gym


_FAULT_TYPES = (
    'broken_door',
    'heavy_key',
    'action_flip',
    'teleport',
    'door_gone',
    'lava_gap',
)

_FAULT_IDS = {name: index + 1 for index, name in enumerate(_FAULT_TYPES)}
_FAULT_FAMILIES = {
    'broken_door': 1,
    'heavy_key': 1,
    'action_flip': 2,
    'teleport': 3,
    'door_gone': 4,
    'lava_gap': 4,
}

_PROFILES = {
    'clean': (),
    'diagnostic': _FAULT_TYPES,
    'benchmark_train': ('broken_door', 'heavy_key', 'action_flip'),
    'benchmark_seen': ('broken_door', 'heavy_key', 'action_flip'),
    'benchmark_holdout': ('teleport', 'door_gone', 'lava_gap'),
    'benchmark_sparse': _FAULT_TYPES,
}

_PROFILE_EP_PROB = {
    'clean': 0.0,
    'diagnostic': 1.0,
    'benchmark_train': 0.25,
    'benchmark_seen': 0.5,
    'benchmark_holdout': 0.5,
    'benchmark_sparse': 0.1,
}


class MiniGrid(embodied.Env):
  """MiniGrid DoorKey with controlled conditional fault injection."""

  def __init__(
      self, task='doorkey6x6', obs_mode='symbolic', size=(64, 64),
      tile_size=8, seed=0, fault_profile='clean', fault='none',
      episode_prob=None, manifest_prob=1.0, teleport_step=5,
      structural_step=5, length=360):
    gym = _import_minigrid()
    from minigrid.core.actions import Actions

    tasks = {
        'doorkey6x6': 'MiniGrid-DoorKey-6x6-v0',
        'doorkey': 'MiniGrid-DoorKey-6x6-v0',
    }
    if task not in tasks:
      raise ValueError(f'Unknown MiniGrid task: {task}')
    if obs_mode not in ('symbolic', 'rgb'):
      raise ValueError(f'Unknown MiniGrid observation mode: {obs_mode}')

    self._env = gym.make(tasks[task], max_steps=int(length))
    self._base = self._env.unwrapped
    self._Actions = Actions
    self._obs_mode = obs_mode
    self._size = tuple(size)
    self._tile_size = int(tile_size)
    self._seed = int(seed)
    self._did_seed = False
    self._done = True
    self._episode_id = 0
    self._episode_step = 0
    self._fault_count = 0
    self._unique_faults = set()
    self._fault_rng = np.random.default_rng(self._seed ^ 0x5A17)
    self._teleport_step = int(teleport_step)
    self._structural_step = int(structural_step)

    requested_profile = os.getenv(
        'MINIGRID_FAULT_PROFILE', fault_profile).strip().lower()
    requested_fault = os.getenv(
        'MINIGRID_FAULT_TYPE', fault).strip().lower()
    enabled = os.getenv('MINIGRID_FAULT', '').strip().lower()
    if enabled in ('0', 'false', 'no', 'off'):
      requested_profile, requested_fault = 'clean', 'none'
    if requested_profile not in _PROFILES:
      raise ValueError(f'Unknown MiniGrid fault profile: {requested_profile}')
    if requested_fault not in ('', 'none', *_FAULT_TYPES):
      raise ValueError(f'Unknown MiniGrid fault type: {requested_fault}')
    self._fault_profile = requested_profile
    self._forced_fault = requested_fault or 'none'
    default_prob = _PROFILE_EP_PROB[requested_profile]
    value = os.getenv('MINIGRID_FAULT_EP_PROB')
    self._episode_prob = float(
        value if value is not None else (
            default_prob if episode_prob is None else episode_prob))
    value = os.getenv('MINIGRID_FAULT_MANIFEST_PROB')
    self._manifest_prob = float(
        value if value is not None else manifest_prob)

    self._fault_type = 'none'
    self._fault_episode = False
    self._manifested = False
    self._door_pos = None
    self._lava_positions = ()
    self._structural_active = False

  @functools.cached_property
  def obs_space(self):
    spaces = {
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    if self._obs_mode == 'symbolic':
      # 7x7x3 compact view, direction one-hot, and carried object/color.
      spaces['grid'] = elements.Space(np.float32, (153,), 0.0, 1.0)
    else:
      spaces['image'] = elements.Space(np.uint8, (*self._size, 3), 0, 255)
    logs = {
        'episode_id': np.int32,
        'episode_step': np.int32,
        'requested_action': np.int32,
        'executed_action': np.int32,
        'fault_applied': np.float32,
        'fault_manifested': np.float32,
        'fault_trigger_context': np.float32,
        'fault_episode': np.float32,
        'fault_exists_episode': np.float32,
        'semantic_fault_applied': np.float32,
        'semantic_fault_episode': np.float32,
        'fault_profile_id': np.int32,
        'fault_family_id': np.int32,
        'fault_type_id': np.int32,
        'fault_count_cumulative': np.int32,
        'unique_bug_count_cumulative': np.int32,
        'bug_triggered': np.float32,
        'bug_id': np.int32,
        'bug_type_id': np.int32,
        'context_inventory_bucket': np.int32,
        'context_achievement_stage': np.int32,
        'context_nearby_tile': np.int32,
        'context_nearby_mob': np.int32,
        'task_reward_raw': np.float32,
        'env_reward': np.float32,
    }
    spaces.update({f'log/{key}': elements.Space(dtype) for key, dtype in logs.items()})
    return spaces

  @functools.cached_property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, len(self._Actions)),
        'reset': elements.Space(bool),
    }

  @property
  def fault_type(self):
    return self._fault_type

  @property
  def unwrapped(self):
    return self._base

  def step(self, action):
    if bool(action['reset']) or self._done:
      return self._reset()

    requested = int(action['action'])
    executed = requested
    trigger = False
    applied = False
    fwd_pos = tuple(int(x) for x in self._base.front_pos)
    fwd_cell = self._base.grid.get(*fwd_pos)

    if (
        self._fault_episode and not self._structural_active and
        self._episode_step >= self._structural_step):
      if self._fault_type == 'door_gone' and self._door_pos:
        self._base.grid.set(*self._door_pos, None)
        self._structural_active = True
      elif self._fault_type == 'lava_gap':
        self._lava_positions = self._place_lava_gap()
        self._structural_active = bool(self._lava_positions)

    if self._fault_episode and not self._manifested:
      fault = self._fault_type
      if fault == 'broken_door':
        trigger = (
            requested == int(self._Actions.toggle) and
            getattr(fwd_cell, 'type', None) == 'door' and
            getattr(self._base.carrying, 'type', None) == 'key')
        if trigger and self._sample_manifest():
          executed = int(self._Actions.done)
          applied = True
      elif fault == 'heavy_key':
        trigger = (
            requested == int(self._Actions.pickup) and
            getattr(fwd_cell, 'type', None) == 'key' and
            self._base.carrying is None)
        if trigger and self._sample_manifest():
          executed = int(self._Actions.done)
          applied = True
      elif fault == 'action_flip':
        trigger = requested in (
            int(self._Actions.left), int(self._Actions.right))
        if trigger and self._sample_manifest():
          executed = (
              int(self._Actions.right)
              if requested == int(self._Actions.left)
              else int(self._Actions.left))
          applied = True
      elif fault == 'teleport':
        trigger = self._episode_step >= self._teleport_step
        if trigger and self._sample_manifest():
          applied = self._teleport_agent()

    obs, reward, terminated, truncated, _ = self._env.step(executed)
    self._episode_step += 1

    if self._fault_episode and not self._manifested:
      if (
          self._fault_type == 'door_gone' and self._structural_active and
          self._door_pos):
        trigger = self._base.in_view(*self._door_pos)
        applied = trigger
      elif (
          self._fault_type == 'lava_gap' and self._structural_active and
          self._lava_positions):
        trigger = any(self._base.in_view(*pos) for pos in self._lava_positions)
        applied = trigger

    if applied:
      self._manifested = True
      self._fault_count += 1
      self._unique_faults.add(self._fault_type)

    self._done = bool(terminated or truncated)
    return self._convert_obs(
        obs, reward, requested, executed, trigger, applied,
        is_last=self._done, is_terminal=bool(terminated))

  def _reset(self):
    seed = self._seed if not self._did_seed else None
    obs, _ = self._env.reset(seed=seed)
    self._did_seed = True
    self._done = False
    self._episode_id += 1
    self._episode_step = 0
    self._manifested = False
    self._choose_fault()
    self._door_pos = self._find_object('door')
    self._lava_positions = ()
    self._structural_active = False
    return self._convert_obs(
        obs, 0.0, 0, 0, False, False, is_first=True)

  def _choose_fault(self):
    candidates = _PROFILES[self._fault_profile]
    forced = self._forced_fault
    if forced != 'none':
      candidates = (forced,)
    active = bool(candidates) and self._fault_rng.random() < self._episode_prob
    self._fault_episode = active
    self._fault_type = (
        str(self._fault_rng.choice(candidates)) if active else 'none')

  def _sample_manifest(self):
    return self._fault_rng.random() < self._manifest_prob

  def _find_object(self, obj_type):
    for x in range(self._base.width):
      for y in range(self._base.height):
        if getattr(self._base.grid.get(x, y), 'type', None) == obj_type:
          return (x, y)
    return None

  def _place_lava_gap(self):
    from minigrid.core.world_object import Lava
    goal = self._find_object('goal')
    if not goal:
      return ()
    candidates = ((goal[0] - 1, goal[1]), (goal[0], goal[1] - 1))
    placed = []
    for pos in candidates:
      if getattr(self._base.grid.get(*pos), 'type', None) in (None, 'empty'):
        self._base.put_obj(Lava(), *pos)
        placed.append(pos)
    return tuple(placed)

  def _teleport_agent(self):
    positions = []
    current = tuple(int(x) for x in self._base.agent_pos)
    for x in range(1, self._base.width - 1):
      for y in range(1, self._base.height - 1):
        cell = self._base.grid.get(x, y)
        if (x, y) != current and (cell is None or cell.can_overlap()):
          positions.append((x, y))
    if not positions:
      return False
    index = int(self._fault_rng.integers(0, len(positions)))
    self._base.agent_pos = positions[index]
    return True

  def _convert_obs(
      self, obs, reward, requested, executed, trigger, applied,
      is_first=False, is_last=False, is_terminal=False):
    result = {}
    if self._obs_mode == 'symbolic':
      image = np.asarray(obs['image'], np.float32)
      image[..., 0] /= 10.0
      image[..., 1] /= 5.0
      image[..., 2] /= 2.0
      direction = np.zeros(4, np.float32)
      direction[int(obs['direction'])] = 1.0
      carrying = np.zeros(2, np.float32)
      if self._base.carrying is not None:
        encoded = self._base.carrying.encode()
        carrying[:] = (encoded[0] / 10.0, encoded[1] / 5.0)
      result['grid'] = np.concatenate(
          (image.reshape(-1), direction, carrying)).astype(np.float32)
    else:
      frame = self._base.get_frame(
          tile_size=self._tile_size, agent_pov=True)
      frame = Image.fromarray(frame).resize(
          (self._size[1], self._size[0]), Image.Resampling.NEAREST)
      result['image'] = np.asarray(frame, np.uint8)

    fault_id = _FAULT_IDS.get(self._fault_type, 0)
    result.update(
        reward=np.float32(reward),
        is_first=bool(is_first),
        is_last=bool(is_last),
        is_terminal=bool(is_terminal),
        **{
            'log/episode_id': np.int32(self._episode_id),
            'log/episode_step': np.int32(self._episode_step),
            'log/requested_action': np.int32(requested),
            'log/executed_action': np.int32(executed),
            'log/fault_applied': np.float32(applied),
            'log/fault_manifested': np.float32(applied),
            'log/fault_trigger_context': np.float32(trigger),
            'log/fault_episode': np.float32(self._fault_episode),
            'log/fault_exists_episode': np.float32(self._fault_episode),
            'log/semantic_fault_applied': np.float32(applied),
            'log/semantic_fault_episode': np.float32(self._fault_episode),
            'log/fault_profile_id': np.int32(
                tuple(_PROFILES).index(self._fault_profile)),
            'log/fault_family_id': np.int32(
                _FAULT_FAMILIES.get(self._fault_type, 0)),
            'log/fault_type_id': np.int32(fault_id),
            'log/fault_count_cumulative': np.int32(self._fault_count),
            'log/unique_bug_count_cumulative': np.int32(
                len(self._unique_faults)),
            'log/bug_triggered': np.float32(applied),
            'log/bug_id': np.int32(fault_id),
            'log/bug_type_id': np.int32(fault_id),
            'log/context_inventory_bucket': np.int32(
                self._base.carrying is not None),
            'log/context_achievement_stage': np.int32(self._task_stage()),
            'log/context_nearby_tile': np.int32(self._front_object_id()),
            'log/context_nearby_mob': np.int32(0),
            'log/task_reward_raw': np.float32(reward),
            'log/env_reward': np.float32(reward),
        })
    return result

  def _task_stage(self):
    door = self._base.grid.get(*self._door_pos) if self._door_pos else None
    if getattr(door, 'type', None) == 'door' and not door.is_locked:
      return 3 if door.is_open else 2
    if self._base.carrying is not None:
      return 1
    return 0

  def _front_object_id(self):
    from minigrid.core.constants import OBJECT_TO_IDX
    pos = tuple(int(x) for x in self._base.front_pos)
    cell = self._base.grid.get(*pos)
    return OBJECT_TO_IDX.get(getattr(cell, 'type', 'empty'), 1)

  def close(self):
    self._env.close()

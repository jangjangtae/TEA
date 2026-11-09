import functools
import os
from collections import deque

import elements
import embodied
import jax
import numpy as np


_ACHIEVEMENTS = (
    'collect_wood',
    'place_table',
    'eat_cow',
    'collect_sapling',
    'collect_drink',
    'make_wood_pickaxe',
    'make_wood_sword',
    'place_plant',
    'defeat_zombie',
    'collect_stone',
    'place_stone',
    'eat_plant',
    'defeat_skeleton',
    'make_stone_pickaxe',
    'make_stone_sword',
    'wake_up',
    'place_furnace',
    'collect_coal',
    'collect_iron',
    'collect_diamond',
    'make_iron_pickaxe',
    'make_iron_sword',
)

_FAULT_FAMILY_IDS = {
    'none': 0,
    'action_exec': 2,
    'reward_timing': 4,
    'semantic_high_level': 6,
}

_FAULT_TYPE_IDS = {
    'none': 0,
    'remap_after_success_switch': 2,
    'delay_after_success': 3,
    'sticky_after_repeat_switch': 4,
    'reward_delay_on_positive': 7,
    'reward_scale_half_on_positive_switch': 8,
    'remap_after_repeat_switch': 11,
    'reward_zero_after_repeat_switch': 14,
    'reward_delay_after_two_rewards': 15,
    'revisit_action_delay': 18,
    'delayed_switch_failure': 19,
    'tool_collect_desync_on_upgrade': 20,
    'craft_result_missing_on_retry': 23,
    'station_place_ghost_on_relocate': 27,
    'achievement_unlock_missing_after_valid_progress': 31,
    'delayed_inventory_desync_after_station_use': 34,
}

_FAULT_PROFILE_ALIASES = {
    'train': 'benchmark_train',
    'seen': 'benchmark_seen',
    'eval_seen': 'benchmark_seen',
    'holdout': 'benchmark_holdout',
    'eval_holdout': 'benchmark_holdout',
    'sparse': 'benchmark_sparse',
}

_FAULT_PROFILE_DEFAULTS = {
    # Smoke-test profile: frequent enough to validate logs and analysis.
    'diagnostic': {
        'families': 'action_exec,reward_timing,semantic_high_level',
        'action': (
            'remap_after_success_switch,delay_after_success,'
            'sticky_after_repeat_switch,remap_after_repeat_switch'),
        'reward': (
            'reward_delay_on_positive,reward_scale_half_on_positive_switch,'
            'reward_zero_after_repeat_switch'),
        'semantic': (
            'tool_collect_desync_on_upgrade,craft_result_missing_on_retry,'
            'station_place_ghost_on_relocate,'
            'achievement_unlock_missing_after_valid_progress'),
        'episode_prob': '1.0',
        'severities': '1.0',
        'manifest_prob': '1.0',
        'cooldown': '0',
    },
    # Benchmark profiles keep low-level faults as auxiliary diagnostics and
    # include semantic rule violations as the main game-QA bug class.
    'benchmark_train': {
        'families': 'action_exec,reward_timing,semantic_high_level',
        'action': (
            'delay_after_success,remap_after_success_switch,'
            'sticky_after_repeat_switch'),
        'reward': (
            'reward_delay_on_positive,reward_scale_half_on_positive_switch'),
        'semantic': (
            'tool_collect_desync_on_upgrade,craft_result_missing_on_retry,'
            'station_place_ghost_on_relocate'),
        'episode_prob': '0.25',
        'severities': '0.05,0.1',
        'manifest_prob': '1.0',
        'cooldown': '12',
    },
    'benchmark_seen': {
        'families': 'action_exec,reward_timing,semantic_high_level',
        'action': (
            'delay_after_success,remap_after_success_switch,'
            'sticky_after_repeat_switch'),
        'reward': (
            'reward_delay_on_positive,reward_scale_half_on_positive_switch'),
        'semantic': (
            'tool_collect_desync_on_upgrade,craft_result_missing_on_retry,'
            'station_place_ghost_on_relocate'),
        'episode_prob': '0.25',
        'severities': '0.05,0.1',
        'manifest_prob': '1.0',
        'cooldown': '8',
    },
    'benchmark_holdout': {
        'families': 'action_exec,reward_timing,semantic_high_level',
        'action': (
            'revisit_action_delay,delayed_switch_failure,'
            'remap_after_repeat_switch'),
        'reward': 'reward_zero_after_repeat_switch,reward_delay_after_two_rewards',
        'semantic': (
            'achievement_unlock_missing_after_valid_progress,'
            'delayed_inventory_desync_after_station_use'),
        'episode_prob': '0.25',
        'severities': '0.05,0.1',
        'manifest_prob': '1.0',
        'cooldown': '8',
    },
    'benchmark_sparse': {
        'families': 'action_exec,reward_timing,semantic_high_level',
        'action': (
            'revisit_action_delay,delayed_switch_failure,'
            'remap_after_repeat_switch'),
        'reward': 'reward_zero_after_repeat_switch,reward_delay_after_two_rewards',
        'semantic': (
            'achievement_unlock_missing_after_valid_progress,'
            'delayed_inventory_desync_after_station_use'),
        'episode_prob': '0.5',
        'severities': '0.01,0.03',
        'manifest_prob': '1.0',
        'cooldown': '16',
    },
}

_ACTION_REMAP = {
    1: 2,  # LEFT -> RIGHT
    2: 1,  # RIGHT -> LEFT
    3: 4,  # UP -> DOWN
    4: 3,  # DOWN -> UP
    7: 5,  # PLACE_STONE -> DO
    8: 5,  # PLACE_TABLE -> DO
    9: 5,  # PLACE_FURNACE -> DO
    10: 5,  # PLACE_PLANT -> DO
    11: 14,  # pickaxe/sword craft swaps keep the action plausible
    12: 15,
    13: 16,
    14: 11,
    15: 12,
    16: 13,
}

_CRAFT_OUTPUTS = {
    11: ('wood_pickaxe', 5),
    12: ('stone_pickaxe', 13),
    13: ('iron_pickaxe', 20),
    14: ('wood_sword', 6),
    15: ('stone_sword', 14),
    16: ('iron_sword', 21),
}

_COLLECT_FIELDS = (
    ('wood', 0),
    ('sapling', 3),
    ('stone', 9),
    ('coal', 17),
    ('iron', 18),
    ('diamond', 19),
)

_PLACE_ACTIONS = {
    8: 11,  # PLACE_TABLE -> CRAFTING_TABLE
    9: 12,  # PLACE_FURNACE -> FURNACE
}

_DIRS = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


class RunningMeanStd:

  def __init__(self, epsilon=1e-4):
    self.mean = 0.0
    self.var = 1.0
    self.count = float(epsilon)

  def update(self, values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
      return
    self._update_from_moments(float(arr.mean()), float(arr.var()), float(arr.size))

  def _update_from_moments(self, batch_mean, batch_var, batch_count):
    delta = batch_mean - self.mean
    total_count = self.count + batch_count
    new_mean = self.mean + delta * batch_count / total_count

    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
    self.mean = float(new_mean)
    self.var = float(max(m2 / total_count, 1e-12))
    self.count = float(total_count)


class RNDModule:

  def __init__(
      self, obs_shape=(64, 64, 3), downsample=4, hidden_dim=128,
      output_dim=128, lr=0.005, seed=0):
    self.lr = float(lr)
    self.output_dim = int(output_dim)
    self.downsample = int(max(1, downsample))
    self.rng = np.random.default_rng(int(seed))

    sample = np.zeros(obs_shape, dtype=np.uint8)[
        ::self.downsample, ::self.downsample, :]
    self.input_dim = int(sample.size)

    self.W1_t = (
        self.rng.standard_normal((self.input_dim, hidden_dim)) /
        np.sqrt(self.input_dim))
    self.b1_t = np.zeros(hidden_dim, dtype=np.float32)
    self.W2_t = (
        self.rng.standard_normal((hidden_dim, output_dim)) /
        np.sqrt(hidden_dim))
    self.b2_t = np.zeros(output_dim, dtype=np.float32)

    self.W1_p = (
        self.rng.standard_normal((self.input_dim, hidden_dim)) /
        np.sqrt(self.input_dim))
    self.b1_p = np.zeros(hidden_dim, dtype=np.float32)
    self.W2_p = (
        self.rng.standard_normal((hidden_dim, output_dim)) /
        np.sqrt(hidden_dim))
    self.b2_p = np.zeros(output_dim, dtype=np.float32)

  def _preprocess(self, obs):
    small = obs[::self.downsample, ::self.downsample, :]
    return small.reshape(-1).astype(np.float32) / 255.0

  def compute_intrinsic_reward(self, obs, update=True):
    x = self._preprocess(obs)

    z1_t = np.dot(x, self.W1_t) + self.b1_t
    h_t = np.maximum(0, z1_t)
    y_t = np.dot(h_t, self.W2_t) + self.b2_t

    z1_p = np.dot(x, self.W1_p) + self.b1_p
    h_p = np.maximum(0, z1_p)
    y_p = np.dot(h_p, self.W2_p) + self.b2_p

    diff = y_p - y_t
    mse = float(np.mean(diff ** 2))

    if update:
      dy_p = diff / self.output_dim
      dW2_p = np.outer(h_p, dy_p)
      db2_p = dy_p
      dh_p = np.dot(self.W2_p, dy_p)
      dz1_p = dh_p * (z1_p > 0)
      dW1_p = np.outer(x, dz1_p)
      db1_p = dz1_p

      self.W2_p -= self.lr * dW2_p
      self.b2_p -= self.lr * db2_p
      self.W1_p -= self.lr * dW1_p
      self.b1_p -= self.lr * db1_p

    return mse, mse


class Craftax(embodied.Env):

  def __init__(
      self, task='classic_pixels', seed=0, length=10000,
      logs=True, variant=None, platform='cpu', logdir=None):
    task = variant or task
    task = task.replace('-', '_').lower()
    if task in ('classic', 'classic_pixels', 'pixels'):
      self._obs_kind = 'pixels'
    elif task in ('classic_symbolic', 'symbolic'):
      self._obs_kind = 'symbolic'
    else:
      raise ValueError(f'Unknown Craftax task: {task!r}')

    self._task = task
    self._logs = bool(logs)
    self._length = int(length) if length else 0
    self._seed = int(seed)
    self._platform = platform
    self._logdir = logdir
    self._achievements = list(_ACHIEVEMENTS)
    self._env = None
    self._key = None
    self._state = None
    self._done = True
    self._params = None
    self._compute_score = None
    self._last_image = np.zeros((64, 64, 3), np.uint8)
    self._init_faults()
    self._init_rnd()

  @functools.cached_property
  def obs_space(self):
    spaces = {}
    if self._obs_kind == 'pixels':
      spaces['image'] = elements.Space(np.uint8, (64, 64, 3))
    else:
      spaces['vector'] = elements.Space(np.float32, (1345,))
    spaces.update(
        reward=elements.Space(np.float32),
        is_first=elements.Space(bool),
        is_last=elements.Space(bool),
        is_terminal=elements.Space(bool),
    )
    if self._logs:
      spaces.update({
          'log/score': elements.Space(np.float32),
          'log/timestep': elements.Space(np.int32),
          'log/health': elements.Space(np.float32),
          'log/food': elements.Space(np.float32),
          'log/drink': elements.Space(np.float32),
          'log/energy': elements.Space(np.float32),
          'log/achievement_count': elements.Space(np.int32),
      })
      for name in self._achievements:
        spaces[f'log/achievement_{name}'] = elements.Space(np.int32)
      spaces.update({
          'log/raw_reward': elements.Space(np.float32),
          'log/task_reward_raw': elements.Space(np.float32),
          'log/env_reward': elements.Space(np.float32),
          'log/requested_action': elements.Space(np.int32),
          'log/env_action': elements.Space(np.int32),
          'log/fault_applied': elements.Space(np.int32),
          'log/fault_episode': elements.Space(np.int32),
          'log/fault_exists_episode': elements.Space(np.int32),
          'log/fault_trigger_context': elements.Space(np.int32),
          'log/fault_manifested': elements.Space(np.int32),
          'log/fault_manifest_prob': elements.Space(np.float32),
          'log/fault_profile_id': elements.Space(np.int32),
          'log/fault_frequency_tier_id': elements.Space(np.int32),
          'log/fault_family_id': elements.Space(np.int32),
          'log/fault_type_id': elements.Space(np.int32),
          'log/fault_count_cumulative': elements.Space(np.int32),
          **{f'log/fault_{key}': elements.Space(np.int32) for key in (
              'assigned_type_id', 'condition_met', 'draw_eligible',
              'draw_success', 'new_injection', 'action_release',
              'reward_release', 'affected_step', 'action_changed',
              'reward_changed', 'cooldown_remaining')},
          'log/fault_draw_probability': elements.Space(np.float32),
          'log/semantic_fault_applied': elements.Space(np.int32),
          'log/semantic_fault_episode': elements.Space(np.int32),
          'log/context_inventory_bucket': elements.Space(np.int32),
          'log/context_achievement_stage': elements.Space(np.int32),
          'log/context_nearby_tile': elements.Space(np.int32),
          'log/context_nearby_mob': elements.Space(np.int32),
          'log/rnd_intrinsic_reward': elements.Space(np.float32),
          'log/rnd_intrinsic_reward_raw': elements.Space(np.float32),
          'log/rnd_reward_bonus': elements.Space(np.float32),
          'log/rnd_alpha': elements.Space(np.float32),
          'log/rnd_loss': elements.Space(np.float32),
          'log/rnd_update_enabled': elements.Space(np.int32),
      })
    return spaces

  @functools.cached_property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.int32, (), 0, 17),
    }

  def step(self, action):
    if bool(action['reset']) or self._done:
      obs = self._reset()
      return self._obs(obs, 0.0, is_first=True)

    self._reset_fault_diagnostics()
    self._decay_fault_cooldown()
    self._fault_diagnostics['cooldown_remaining'] = self._fault_cooldown
    requested_action = int(np.asarray(action['action'], np.int32).reshape(()))
    env_action, action_fault = self._apply_action_fault(requested_action)
    prev_state = self._state
    with jax.transfer_guard('allow'):
      self._key, key = jax.random.split(self._key)
      obs, self._state, reward, done, info = self._env.step(
          key, self._state, np.asarray(env_action, np.int32), self._params)
      self._done = bool(np.asarray(done))
    raw_reward = np.float32(_to_numpy(reward))
    semantic_fault = self._apply_semantic_fault(
        prev_state, requested_action, env_action, raw_reward, self._done, info)
    if semantic_fault['applied']:
      obs, info = self._refresh_obs_info(self._done, info)
    reward, reward_fault = self._apply_reward_fault(raw_reward, requested_action)
    intrinsic_r = self._compute_rnd_reward(obs)
    rnd_bonus = self._rnd_alpha * intrinsic_r
    self._last_rnd_info['rnd_reward_bonus'] = float(rnd_bonus)
    agent_reward = np.float32(float(reward) + rnd_bonus)
    self._last_fault_info = self._merge_fault_info(
        action_fault, reward_fault, semantic_fault,
        raw_reward, reward, requested_action, env_action)
    self._observe_transition(requested_action, env_action, raw_reward)
    return self._obs(
        obs, agent_reward, info,
        is_last=self._done, is_terminal=self._done)

  def render(self):
    return self._last_image

  def _ensure_env(self):
    if self._env is not None:
      return
    if self._platform:
      os.environ.setdefault('JAX_PLATFORMS', self._platform)
      try:
        jax.config.update('jax_platforms', self._platform)
      except Exception as e:
        raise RuntimeError(
            'Craftax JAX platform must be configured before the backend is '
            'initialized. Use `--run.debug False` so Craftax runs in a fresh '
            'environment subprocess, or set `env.craftax.platform` empty to '
            'share the learner backend.') from e
    try:
      with jax.transfer_guard('allow'):
        from craftax.craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnvNoAutoReset)
        from craftax.craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnvNoAutoReset)
        from craftax.craftax_classic.envs.common import compute_score
    except ImportError as e:
      raise ImportError(
          'Craftax is not installed. Install it in the Dreamer environment '
          'with `pip install craftax`, or set PYTHONPATH to a directory that '
          'contains the package for smoke tests.') from e

    with jax.transfer_guard('allow'):
      # The outer step() owns reset; preserve the terminal observation/state.
      if self._obs_kind == 'pixels':
        self._env = CraftaxClassicPixelsEnvNoAutoReset()
      else:
        self._env = CraftaxClassicSymbolicEnvNoAutoReset()
      self._compute_score = compute_score
      self._params = self._env.default_params
      if self._length:
        self._params = self._params.replace(max_timesteps=self._length)
      self._key = jax.random.PRNGKey(self._seed)

  def _reset(self):
    self._ensure_env()
    with jax.transfer_guard('allow'):
      self._key, key = jax.random.split(self._key)
      obs, self._state = self._env.reset(key, self._params)
    self._done = False
    self._reset_fault_episode()
    return obs

  def _obs(
      self, obs, reward, info=None,
      is_first=False, is_last=False, is_terminal=False):
    data = {}
    if self._obs_kind == 'pixels':
      image = self._image_from_obs(obs)
      self._last_image = image
      data['image'] = image
    else:
      with jax.transfer_guard('allow'):
        data['vector'] = np.asarray(obs, np.float32)

    data.update(
        reward=np.float32(_to_numpy(reward)),
        is_first=bool(is_first),
        is_last=bool(is_last),
        is_terminal=bool(is_terminal),
    )
    if self._logs:
      data.update(self._log_obs(info or {}))
    return data

  def _log_obs(self, info):
    state = self._state
    achievements = _to_numpy(state.achievements).astype(np.int32)
    out = {
        'log/score': np.float32(_to_numpy(info.get('score', 0.0))),
        'log/timestep': np.int32(_to_numpy(state.timestep)),
        'log/health': np.float32(_to_numpy(state.player_health) / 9.0),
        'log/food': np.float32(_to_numpy(state.player_food) / 9.0),
        'log/drink': np.float32(_to_numpy(state.player_drink) / 9.0),
        'log/energy': np.float32(_to_numpy(state.player_energy) / 9.0),
        'log/achievement_count': np.int32(achievements.sum()),
    }
    for index, name in enumerate(self._achievements):
      out[f'log/achievement_{name}'] = np.int32(achievements[index])
    out.update(self._context_log_obs(achievements))
    out.update(self._fault_log_obs())
    out.update(self._rnd_log_obs())
    return out

  def _init_rnd(self):
    self._use_rnd = _env_flag('CRAFTAX_USE_RND', False)
    self._rnd_alpha = float(os.getenv('CRAFTAX_RND_ALPHA', '0.05'))
    self._rnd_update = _env_flag('CRAFTAX_RND_UPDATE', True)
    self._rnd_normalize = _env_flag('CRAFTAX_RND_NORM', True)
    self._rnd_clip = float(os.getenv('CRAFTAX_RND_CLIP', '5.0'))
    self._rnd_downsample = int(os.getenv('CRAFTAX_RND_DOWNSAMPLE', '4'))
    self._rnd_hidden_dim = int(os.getenv('CRAFTAX_RND_HIDDEN_DIM', '128'))
    self._rnd_output_dim = int(os.getenv('CRAFTAX_RND_OUTPUT_DIM', '128'))
    self._rnd_lr = float(os.getenv('CRAFTAX_RND_LR', '0.005'))
    self._rnd_seed = int(os.getenv(
        'CRAFTAX_RND_SEED', os.getenv('CRAFTAX_FAULT_SEED', str(self._seed))))
    self._rnd_rms = RunningMeanStd() if (
        self._use_rnd and self._rnd_normalize) else None
    self._rnd_mod = None
    if self._use_rnd:
      if self._obs_kind != 'pixels':
        raise ValueError('CRAFTAX_USE_RND=1 requires pixel observations.')
      self._rnd_mod = RNDModule(
          obs_shape=(64, 64, 3),
          downsample=self._rnd_downsample,
          hidden_dim=self._rnd_hidden_dim,
          output_dim=self._rnd_output_dim,
          lr=self._rnd_lr,
          seed=self._rnd_seed,
      )
    self._last_rnd_info = self._empty_rnd_info()

  def _image_from_obs(self, obs):
    with jax.transfer_guard('allow'):
      image = np.asarray(obs)
    if image.dtype != np.uint8:
      image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.shape[:2] == (63, 63):
      image = np.pad(image, ((0, 1), (0, 1), (0, 0)), mode='edge')
    if image.shape != (64, 64, 3):
      raise RuntimeError(f'Unexpected Craftax image shape: {image.shape}')
    return image

  def _normalize_rnd_reward(self, intrinsic_reward, update_stats=True):
    intrinsic_reward = float(max(0.0, intrinsic_reward))
    if not self._use_rnd or not self._rnd_normalize or self._rnd_rms is None:
      if self._rnd_clip > 0:
        intrinsic_reward = float(np.clip(intrinsic_reward, 0.0, self._rnd_clip))
      return intrinsic_reward
    if update_stats:
      self._rnd_rms.update([intrinsic_reward])
    std = float(np.sqrt(max(self._rnd_rms.var, 1e-12)))
    norm_reward = intrinsic_reward / (std + 1e-8)
    if self._rnd_clip > 0:
      norm_reward = float(np.clip(norm_reward, 0.0, self._rnd_clip))
    return float(norm_reward)

  def _compute_rnd_reward(self, obs):
    if not self._use_rnd:
      self._last_rnd_info = self._empty_rnd_info()
      return 0.0
    image = self._image_from_obs(obs)
    intrinsic_raw, loss = self._rnd_mod.compute_intrinsic_reward(
        image, update=self._rnd_update)
    intrinsic = self._normalize_rnd_reward(
        intrinsic_raw, update_stats=self._rnd_update)
    self._last_rnd_info = {
        'rnd_intrinsic_reward_raw': float(intrinsic_raw),
        'rnd_intrinsic_reward': float(intrinsic),
        'rnd_reward_bonus': float(self._rnd_alpha * intrinsic),
        'rnd_loss': float(loss),
        'rnd_update_enabled': int(self._rnd_update),
    }
    return float(intrinsic)

  def _empty_rnd_info(self):
    return {
      'rnd_intrinsic_reward_raw': 0.0,
      'rnd_intrinsic_reward': 0.0,
      'rnd_reward_bonus': 0.0,
      'rnd_loss': 0.0,
        'rnd_update_enabled': int(
            getattr(self, '_use_rnd', False) and
            getattr(self, '_rnd_update', False)),
    }

  def _rnd_log_obs(self):
    info = getattr(self, '_last_rnd_info', None) or self._empty_rnd_info()
    return {
        'log/rnd_intrinsic_reward': np.float32(
            info.get('rnd_intrinsic_reward', 0.0)),
        'log/rnd_intrinsic_reward_raw': np.float32(
            info.get('rnd_intrinsic_reward_raw', 0.0)),
        'log/rnd_reward_bonus': np.float32(
            info.get('rnd_reward_bonus', 0.0)),
        'log/rnd_alpha': np.float32(self._rnd_alpha),
        'log/rnd_loss': np.float32(info.get('rnd_loss', 0.0)),
        'log/rnd_update_enabled': np.int32(
            info.get('rnd_update_enabled', 0)),
    }

  def _context_log_obs(self, achievements):
    """Compact gameplay context used for clean-score calibration."""
    state = self._state
    inventory = state.inventory
    tool_tier = 0
    if any(_scalar(getattr(inventory, name)) > 0 for name in (
        'iron_pickaxe', 'iron_sword')):
      tool_tier = 3
    elif any(_scalar(getattr(inventory, name)) > 0 for name in (
        'stone_pickaxe', 'stone_sword')):
      tool_tier = 2
    elif any(_scalar(getattr(inventory, name)) > 0 for name in (
        'wood_pickaxe', 'wood_sword')):
      tool_tier = 1
    resources = sum(
        _scalar(getattr(inventory, name)) > 0
        for name in ('wood', 'stone', 'coal', 'iron', 'diamond', 'sapling'))
    inventory_bucket = 4 * tool_tier + min(int(resources), 3)
    achievement_stage = min(int(np.asarray(achievements).sum()) // 3, 7)

    front = self._front_position(state)
    map_array = _to_numpy(state.map)
    if (0 <= front[0] < map_array.shape[0] and
        0 <= front[1] < map_array.shape[1]):
      nearby_tile = int(map_array[front[0], front[1]])
    else:
      nearby_tile = 1
    pos = self._current_position()
    nearby_mob = 0
    if pos is not None:
      mob_map = _to_numpy(state.mob_map)
      r0, r1 = max(0, pos[0] - 1), min(mob_map.shape[0], pos[0] + 2)
      c0, c1 = max(0, pos[1] - 1), min(mob_map.shape[1], pos[1] + 2)
      nearby_mob = int(np.any(mob_map[r0:r1, c0:c1] != 0))
    return {
        'log/context_inventory_bucket': np.int32(inventory_bucket),
        'log/context_achievement_stage': np.int32(achievement_stage),
        'log/context_nearby_tile': np.int32(nearby_tile),
        'log/context_nearby_mob': np.int32(nearby_mob),
    }

  def _init_faults(self):
    self._rng = np.random.default_rng(int(os.getenv(
        'CRAFTAX_FAULT_SEED', str(self._seed))))
    requested = os.getenv('CRAFTAX_FAULT_PROFILE', 'benchmark_train').lower()
    self._fault_profile = _FAULT_PROFILE_ALIASES.get(requested, requested)
    if self._fault_profile not in _FAULT_PROFILE_DEFAULTS:
      raise ValueError(
          f'Unknown CRAFTAX_FAULT_PROFILE={requested!r}. Expected one of '
          f'{sorted(_FAULT_PROFILE_DEFAULTS)} or aliases '
          f'{sorted(_FAULT_PROFILE_ALIASES)}.')
    defaults = _FAULT_PROFILE_DEFAULTS[self._fault_profile]
    self._fault_sampler = _env_flag('CRAFTAX_FAULT_SAMPLER', False)
    self._fault_enabled = _env_flag('CRAFTAX_FAULT', False)
    self._fault_episode_prob = float(os.getenv(
        'CRAFTAX_FAULT_EP_PROB', defaults['episode_prob']))
    self._fault_manifest_prob = float(os.getenv(
        'CRAFTAX_FAULT_MANIFEST_PROB', defaults['manifest_prob']))
    self._fault_cooldown_steps = int(os.getenv(
        'CRAFTAX_FAULT_COOLDOWN', defaults['cooldown']))
    self._fault_families = _parse_csv(os.getenv(
        'CRAFTAX_FAULT_FAMILIES', defaults['families']))
    self._fault_action_subtypes = _parse_csv(os.getenv(
        'CRAFTAX_ACTION_SUBTYPES', defaults['action']))
    self._fault_reward_subtypes = _parse_csv(os.getenv(
        'CRAFTAX_REWARD_SUBTYPES', defaults['reward']))
    self._fault_semantic_subtypes = _parse_csv(os.getenv(
        'CRAFTAX_SEMANTIC_SUBTYPES', defaults['semantic']))
    self._fault_severities = [
        float(x) for x in _parse_csv(os.getenv(
            'CRAFTAX_FAULT_SEVERITIES', defaults['severities']))]
    self._fault_count = 0
    self._episode = 0
    self._episode_step = 0
    self._fault_spec = None
    self._fault_episode = 0
    self._fault_cooldown = 0
    self._pending_action = None
    self._pending_reward = 0.0
    self._recent_actions = deque(maxlen=8)
    self._recent_positions = deque(maxlen=32)
    self._recent_rewards = deque(maxlen=8)
    self._last_success_step = -10**9
    self._last_env_action = 0
    self._reset_fault_diagnostics()
    self._last_fault_info = self._empty_fault_info()

  def _reset_fault_episode(self):
    self._episode += 1
    self._episode_step = 0
    self._fault_cooldown = 0
    self._pending_action = None
    self._pending_reward = 0.0
    self._recent_actions.clear()
    self._recent_positions.clear()
    self._recent_rewards.clear()
    self._last_success_step = -10**9
    self._last_env_action = 0
    self._reset_fault_diagnostics()
    self._sample_fault_spec()
    self._last_fault_info = self._empty_fault_info()
    self._last_rnd_info = self._empty_rnd_info()

  def _sample_fault_spec(self):
    self._fault_spec = None
    self._fault_episode = 0
    if self._fault_sampler:
      if self._rng.random() >= self._fault_episode_prob:
        return
      family = self._choice(self._fault_families, 'semantic_high_level')
      subtype = self._choice(self._subtypes_for_family(family), 'none')
      self._fault_spec = {
          'family': family,
          'type': subtype,
          'severity': float(self._choice(self._fault_severities, 0.1)),
      }
      self._fault_episode = 1
    elif self._fault_enabled:
      family = os.getenv('CRAFTAX_FAULT_FAMILY', 'semantic_high_level')
      subtype = os.getenv('CRAFTAX_FAULT_TYPE', 'achievement_unlock_missing_after_valid_progress')
      self._fault_spec = {
          'family': family,
          'type': subtype,
          'severity': float(os.getenv('CRAFTAX_FAULT_SEVERITY', '1.0')),
      }
      self._fault_episode = 1

  def _subtypes_for_family(self, family):
    if family == 'action_exec':
      return self._fault_action_subtypes
    if family == 'reward_timing':
      return self._fault_reward_subtypes
    if family == 'semantic_high_level':
      return self._fault_semantic_subtypes
    return ['none']

  def _choice(self, values, default=None):
    values = list(values)
    if not values:
      return default
    return values[int(self._rng.integers(0, len(values)))]

  def _decay_fault_cooldown(self):
    if self._fault_cooldown > 0:
      self._fault_cooldown -= 1

  def _should_manifest(self, family):
    if not self._fault_spec or not self._fault_episode:
      return False
    if self._fault_spec.get('family') != family:
      return False
    if self._fault_cooldown > 0:
      return False
    severity = float(self._fault_spec.get('severity', 0.0))
    prob = float(np.clip(severity * self._fault_manifest_prob, 0.0, 1.0))
    self._fault_diagnostics['draw_eligible'] = 1
    success = bool(self._rng.random() < prob)
    self._fault_diagnostics['draw_success'] = int(success)
    return success

  def _reset_fault_diagnostics(self):
    self._fault_diagnostics = dict.fromkeys((
        'condition_met', 'draw_eligible', 'draw_success', 'new_injection',
        'action_release', 'reward_release', 'affected_step', 'action_changed',
        'reward_changed', 'cooldown_remaining'), 0)

  def _record_fault_condition(self, family, trigger):
    if self._fault_spec and self._fault_spec.get('family') == family:
      self._fault_diagnostics['condition_met'] = int(bool(trigger))

  def _apply_action_fault(self, requested_action):
    if self._pending_action is not None:
      delayed = int(self._pending_action)
      self._pending_action = None
      self._fault_diagnostics['action_release'] = 1
      return delayed, self._fault_info(
          True, 'action_exec', self._fault_spec_type(), 'release_delayed_action')
    trigger = self._action_trigger_context(requested_action)
    self._record_fault_condition('action_exec', trigger)
    if not trigger or not self._should_manifest('action_exec'):
      return requested_action, self._empty_fault_info(trigger_context=trigger)
    fault_type = self._fault_spec_type()
    env_action = requested_action
    if fault_type in ('remap_after_success_switch', 'remap_after_repeat_switch'):
      env_action = _ACTION_REMAP.get(requested_action, 0)
    elif fault_type in (
        'delay_after_success', 'revisit_action_delay',
        'delayed_switch_failure'):
      self._pending_action = requested_action
      env_action = 0
    elif fault_type == 'sticky_after_repeat_switch':
      env_action = int(self._last_env_action)
    else:
      return requested_action, self._empty_fault_info(trigger_context=trigger)
    self._fault_diagnostics['new_injection'] = 1
    return env_action, self._fault_info(
        True, 'action_exec', fault_type, 'action_context', trigger)

  def _apply_reward_fault(self, raw_reward, requested_action):
    reward = float(raw_reward)
    released = float(self._pending_reward)
    self._fault_diagnostics['reward_release'] = int(released != 0)
    self._pending_reward = 0.0
    output = reward + released
    trigger = self._reward_trigger_context(reward, requested_action)
    self._record_fault_condition('reward_timing', trigger)
    if not trigger or not self._should_manifest('reward_timing'):
      return np.float32(output), self._empty_fault_info(trigger_context=trigger)
    fault_type = self._fault_spec_type()
    if fault_type in ('reward_delay_on_positive', 'reward_delay_after_two_rewards'):
      self._pending_reward += reward
      output = released
    elif fault_type == 'reward_scale_half_on_positive_switch':
      output = released + 0.5 * reward
    elif fault_type == 'reward_zero_after_repeat_switch':
      output = released
    else:
      return np.float32(output), self._empty_fault_info(trigger_context=trigger)
    self._fault_diagnostics['new_injection'] = 1
    return np.float32(output), self._fault_info(
        True, 'reward_timing', fault_type, 'reward_context', trigger)

  def _apply_semantic_fault(
      self, prev_state, requested_action, env_action, raw_reward, done, info):
    del env_action, raw_reward, done, info
    trigger = self._semantic_trigger_context(prev_state, requested_action)
    self._record_fault_condition('semantic_high_level', trigger)
    if not trigger or not self._should_manifest('semantic_high_level'):
      return self._empty_fault_info(trigger_context=trigger)
    fault_type = self._fault_spec_type()
    applied = False
    if fault_type == 'tool_collect_desync_on_upgrade':
      applied = self._fault_collect_result_missing(prev_state, requested_action)
    elif fault_type == 'craft_result_missing_on_retry':
      applied = self._fault_craft_result_missing(prev_state, requested_action)
    elif fault_type == 'station_place_ghost_on_relocate':
      applied = self._fault_station_place_ghost(prev_state, requested_action)
    elif fault_type == 'achievement_unlock_missing_after_valid_progress':
      applied = self._fault_achievement_missing(prev_state)
    elif fault_type == 'delayed_inventory_desync_after_station_use':
      applied = self._fault_delayed_inventory_desync(prev_state, requested_action)
    if not applied:
      return self._empty_fault_info(trigger_context=trigger)
    self._fault_diagnostics['new_injection'] = 1
    return self._fault_info(
        True, 'semantic_high_level', fault_type, 'semantic_context', trigger,
        semantic=True)

  def _action_trigger_context(self, action):
    success_window = self._episode_step - self._last_success_step <= 20
    switched = bool(self._recent_actions and self._recent_actions[-1] != action)
    repeated = (
        len(self._recent_actions) >= 3 and
        len(set(list(self._recent_actions)[-3:])) == 1 and switched)
    revisit = self._current_position() in self._recent_positions
    fault_type = self._fault_spec_type()
    if fault_type in ('remap_after_success_switch', 'delay_after_success'):
      return int(success_window and switched)
    if fault_type in ('sticky_after_repeat_switch', 'remap_after_repeat_switch'):
      return int(repeated)
    if fault_type == 'revisit_action_delay':
      return int(revisit and switched)
    if fault_type == 'delayed_switch_failure':
      return int(success_window and repeated)
    return 0

  def _reward_trigger_context(self, reward, action):
    del action
    positive = reward > 0.01
    repeated_switch = (
        len(self._recent_actions) >= 3 and
        len(set(list(self._recent_actions)[-3:])) == 1)
    two_rewards = sum(x > 0.01 for x in self._recent_rewards) >= 2
    fault_type = self._fault_spec_type()
    if fault_type in ('reward_delay_on_positive', 'reward_scale_half_on_positive_switch'):
      return int(positive)
    if fault_type == 'reward_zero_after_repeat_switch':
      return int(positive and repeated_switch)
    if fault_type == 'reward_delay_after_two_rewards':
      return int(positive and two_rewards)
    return 0

  def _semantic_trigger_context(self, prev_state, action):
    fault_type = self._fault_spec_type()
    if prev_state is None or self._state is None:
      return 0
    if fault_type == 'tool_collect_desync_on_upgrade':
      return int(action == 5 and self._first_inventory_increase(
          prev_state, self._state, _COLLECT_FIELDS)[0] is not None)
    if fault_type == 'craft_result_missing_on_retry':
      return int(action in _CRAFT_OUTPUTS and self._craft_output_increased(
          prev_state, self._state, action))
    if fault_type == 'station_place_ghost_on_relocate':
      return int(action in _PLACE_ACTIONS and self._placed_station(
          prev_state, self._state, action))
    if fault_type == 'achievement_unlock_missing_after_valid_progress':
      return int(self._first_new_achievement(prev_state, self._state) is not None)
    if fault_type == 'delayed_inventory_desync_after_station_use':
      return int(action in _CRAFT_OUTPUTS and self._any_inventory_changed(
          prev_state, self._state))
    return 0

  def _fault_collect_result_missing(self, prev_state, action):
    if action != 5:
      return False
    field, _ = self._first_inventory_increase(
        prev_state, self._state, _COLLECT_FIELDS)
    if field is None:
      return False
    inv = self._state.inventory.replace(
        **{field: getattr(prev_state.inventory, field)})
    self._state = self._state.replace(inventory=inv)
    return True

  def _fault_craft_result_missing(self, prev_state, action):
    if action not in _CRAFT_OUTPUTS:
      return False
    field, _ = _CRAFT_OUTPUTS[action]
    if _scalar(getattr(self._state.inventory, field)) <= _scalar(
        getattr(prev_state.inventory, field)):
      return False
    inv = self._state.inventory.replace(
        **{field: getattr(prev_state.inventory, field)})
    self._state = self._state.replace(inventory=inv)
    return True

  def _fault_station_place_ghost(self, prev_state, action):
    if not self._placed_station(prev_state, self._state, action):
      return False
    pos = self._front_position(prev_state)
    with jax.transfer_guard('allow'):
      prev_value = prev_state.map[pos[0], pos[1]]
      new_map = self._state.map.at[pos[0], pos[1]].set(prev_value)
    self._state = self._state.replace(map=new_map)
    return True

  def _fault_achievement_missing(self, prev_state):
    index = self._first_new_achievement(prev_state, self._state)
    if index is None:
      return False
    with jax.transfer_guard('allow'):
      achievements = self._state.achievements.at[index].set(
          prev_state.achievements[index])
    self._state = self._state.replace(achievements=achievements)
    return True

  def _fault_delayed_inventory_desync(self, prev_state, action):
    if action not in _CRAFT_OUTPUTS:
      return False
    changed = self._first_inventory_change(prev_state, self._state)
    if changed is None:
      return False
    field = changed
    value = getattr(self._state.inventory, field)
    inv = self._state.inventory.replace(**{field: np.maximum(_scalar(value) - 1, 0)})
    self._state = self._state.replace(inventory=inv)
    return True

  def _refresh_obs_info(self, done, info):
    with jax.transfer_guard('allow'):
      obs = self._env.get_obs(self._state)
    info = dict(info or {})
    if self._compute_score is not None:
      with jax.transfer_guard('allow'):
        info.update(self._compute_score(self._state, done))
    try:
      info['discount'] = self._env.discount(self._state, self._params)
    except Exception:
      pass
    return obs, info

  def _merge_fault_info(
      self, action_fault, reward_fault, semantic_fault,
      raw_reward, reward, requested_action, env_action):
    chosen = (
        semantic_fault if semantic_fault['applied'] else
        action_fault if action_fault['applied'] else
        reward_fault if reward_fault['applied'] else
        self._empty_fault_info())
    applied = int(
        action_fault['applied'] or reward_fault['applied'] or
        semantic_fault['applied'])
    # Keep legacy event semantics; releases and new injections are distinct logs.
    self._fault_diagnostics['affected_step'] = int(
        bool(applied or self._fault_diagnostics['reward_release']))
    self._fault_diagnostics['action_changed'] = int(requested_action != env_action)
    self._fault_diagnostics['reward_changed'] = int(float(raw_reward) != float(reward))
    if applied:
      self._fault_count += 1
      self._fault_cooldown = max(self._fault_cooldown, self._fault_cooldown_steps)
    info = dict(chosen)
    info.update({
        'applied': applied,
        'fault_episode': int(self._fault_episode),
        'fault_exists_episode': int(self._fault_episode),
        'manifested': applied,
        'manifest_prob': float(
            self._fault_manifest_prob * float(
                self._fault_spec.get('severity', 0.0))
            if self._fault_spec else 0.0),
        'trigger_context': int(
            action_fault['trigger_context'] or
            reward_fault['trigger_context'] or
            semantic_fault['trigger_context']),
        'raw_reward': float(raw_reward),
        'env_reward': float(reward),
        'requested_action': int(requested_action),
        'env_action': int(env_action),
        'semantic_applied': int(semantic_fault['applied']),
        'semantic_episode': int(bool(
            self._fault_episode and
            self._fault_spec and
            self._fault_spec.get('family') == 'semantic_high_level')),
    })
    return info

  def _fault_info(
      self, applied, family, fault_type, trigger, trigger_context=1,
      semantic=False):
    del semantic
    return {
        'applied': int(applied),
        'family': family,
        'type': fault_type,
        'trigger': trigger,
        'trigger_context': int(trigger_context),
    }

  def _empty_fault_info(self, trigger_context=0):
    return {
        'applied': 0,
        'family': self._fault_spec.get('family', 'none') if self._fault_spec else 'none',
        'type': 'none',
        'trigger': 'none',
        'trigger_context': int(trigger_context),
        'raw_reward': 0.0,
        'env_reward': 0.0,
        'requested_action': 0,
        'env_action': 0,
        'fault_episode': int(getattr(self, '_fault_episode', 0)),
        'fault_exists_episode': int(getattr(self, '_fault_episode', 0)),
        'manifested': 0,
        'manifest_prob': 0.0,
        'semantic_applied': 0,
        'semantic_episode': 0,
    }

  def _fault_log_obs(self):
    info = self._last_fault_info or self._empty_fault_info()
    family = info.get('family', 'none')
    fault_type = info.get('type', 'none')
    return {
        'log/raw_reward': np.float32(info.get('raw_reward', 0.0)),
        'log/task_reward_raw': np.float32(info.get('raw_reward', 0.0)),
        'log/env_reward': np.float32(info.get('env_reward', 0.0)),
        'log/requested_action': np.int32(info.get('requested_action', 0)),
        'log/env_action': np.int32(info.get('env_action', 0)),
        'log/fault_applied': np.int32(info.get('applied', 0)),
        'log/fault_episode': np.int32(info.get('fault_episode', 0)),
        'log/fault_exists_episode': np.int32(info.get('fault_exists_episode', 0)),
        'log/fault_trigger_context': np.int32(info.get('trigger_context', 0)),
        'log/fault_manifested': np.int32(info.get('manifested', 0)),
        'log/fault_manifest_prob': np.float32(info.get('manifest_prob', 0.0)),
        'log/fault_profile_id': np.int32(
            sorted(_FAULT_PROFILE_DEFAULTS).index(self._fault_profile) + 1),
        'log/fault_frequency_tier_id': np.int32(0),
        'log/fault_family_id': np.int32(_FAULT_FAMILY_IDS.get(family, 0)),
        'log/fault_type_id': np.int32(_FAULT_TYPE_IDS.get(fault_type, 0)),
        'log/fault_count_cumulative': np.int32(self._fault_count),
        'log/fault_assigned_type_id': np.int32(
            _FAULT_TYPE_IDS.get(self._fault_spec_type(), 0)),
        'log/fault_draw_probability': np.float32(np.clip(
            self._fault_manifest_prob * self._fault_spec.get('severity', 0.0)
            if self._fault_spec else 0.0, 0.0, 1.0)),
        **{f'log/fault_{key}': np.int32(value)
           for key, value in self._fault_diagnostics.items()},
        'log/semantic_fault_applied': np.int32(info.get('semantic_applied', 0)),
        'log/semantic_fault_episode': np.int32(info.get('semantic_episode', 0)),
    }

  def _observe_transition(self, requested_action, env_action, raw_reward):
    self._episode_step += 1
    self._recent_actions.append(int(requested_action))
    self._recent_rewards.append(float(raw_reward))
    pos = self._current_position()
    if pos is not None:
      self._recent_positions.append(pos)
    if float(raw_reward) > 0.01:
      self._last_success_step = self._episode_step
    self._last_env_action = int(env_action)

  def _fault_spec_type(self):
    return self._fault_spec.get('type', 'none') if self._fault_spec else 'none'

  def _current_position(self):
    if self._state is None:
      return None
    pos = _to_numpy(self._state.player_position).astype(np.int32)
    return (int(pos[0]), int(pos[1]))

  def _front_position(self, state):
    pos = _to_numpy(state.player_position).astype(np.int32)
    direction = int(_scalar(state.player_direction))
    delta = _DIRS.get(direction, (0, 0))
    return (int(pos[0] + delta[0]), int(pos[1] + delta[1]))

  def _first_inventory_increase(self, prev_state, state, fields):
    for field, achievement in fields:
      if _scalar(getattr(state.inventory, field)) > _scalar(
          getattr(prev_state.inventory, field)):
        return field, achievement
    return None, None

  def _first_inventory_change(self, prev_state, state):
    for field in state.inventory.__dataclass_fields__:
      if _scalar(getattr(state.inventory, field)) != _scalar(
          getattr(prev_state.inventory, field)):
        return field
    return None

  def _any_inventory_changed(self, prev_state, state):
    return self._first_inventory_change(prev_state, state) is not None

  def _craft_output_increased(self, prev_state, state, action):
    field, _ = _CRAFT_OUTPUTS[action]
    return _scalar(getattr(state.inventory, field)) > _scalar(
        getattr(prev_state.inventory, field))

  def _placed_station(self, prev_state, state, action):
    if action not in _PLACE_ACTIONS:
      return False
    pos = self._front_position(prev_state)
    if not self._position_in_bounds(state, pos):
      return False
    target = _PLACE_ACTIONS[action]
    with jax.transfer_guard('allow'):
      before = int(_scalar(prev_state.map[pos[0], pos[1]]))
      after = int(_scalar(state.map[pos[0], pos[1]]))
    return after == target and before != target

  def _position_in_bounds(self, state, pos):
    shape = tuple(_to_numpy(state.map.shape))
    return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]

  def _first_new_achievement(self, prev_state, state):
    prev = _to_numpy(prev_state.achievements).astype(bool)
    cur = _to_numpy(state.achievements).astype(bool)
    changed = np.flatnonzero(np.logical_and(cur, np.logical_not(prev)))
    if len(changed):
      return int(changed[0])
    return None


def _to_numpy(value):
  with jax.transfer_guard('allow'):
    return np.asarray(value)


def _scalar(value):
  return np.asarray(_to_numpy(value)).reshape(()).item()


def _parse_csv(value):
  if isinstance(value, (list, tuple)):
    return [str(x).strip() for x in value if str(x).strip()]
  return [x.strip() for x in str(value).split(',') if x.strip()]


def _env_flag(name, default):
  value = os.getenv(name)
  if value is None:
    return bool(default)
  return str(value).strip().lower() in ('1', 'true', 'yes', 'on')

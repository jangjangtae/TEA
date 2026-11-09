import os
import pathlib
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

ROOT = pathlib.Path(__file__).resolve().parents[1]
CRAFTAX_DEPS = ROOT / '.deps' / 'craftax_pkgs'
if CRAFTAX_DEPS.exists():
  sys.path.insert(0, str(CRAFTAX_DEPS))

import numpy as np

from embodied.envs.craftax import Craftax, _scalar


def make_env():
  env = Craftax(seed=0, logs=True, platform='cpu')
  env.step({'reset': True, 'action': np.int32(0)})
  env._fault_episode = 1
  env._fault_manifest_prob = 1.0
  env._fault_cooldown = 0
  return env


def set_spec(env, family, fault_type):
  env._fault_spec = {
      'family': family,
      'type': fault_type,
      'severity': 1.0,
  }
  env._fault_episode = 1
  env._fault_cooldown = 0


def test_action_faults():
  env = make_env()
  set_spec(env, 'action_exec', 'remap_after_success_switch')
  env._last_success_step = env._episode_step
  env._recent_actions.append(1)
  action, info = env._apply_action_fault(2)
  assert action == 1
  assert info['applied'] == 1

  env = make_env()
  set_spec(env, 'action_exec', 'delay_after_success')
  env._last_success_step = env._episode_step
  env._recent_actions.append(1)
  action, info = env._apply_action_fault(2)
  assert action == 0
  assert env._pending_action == 2
  assert info['applied'] == 1

  env = make_env()
  set_spec(env, 'action_exec', 'sticky_after_repeat_switch')
  env._last_env_action = 4
  env._recent_actions.extend([1, 1, 1])
  action, info = env._apply_action_fault(2)
  assert action == 4
  assert info['applied'] == 1

  env = make_env()
  set_spec(env, 'action_exec', 'remap_after_repeat_switch')
  env._recent_actions.extend([1, 1, 1])
  action, info = env._apply_action_fault(2)
  assert action == 1
  assert info['applied'] == 1

  env = make_env()
  set_spec(env, 'action_exec', 'revisit_action_delay')
  env._recent_positions.append(env._current_position())
  env._recent_actions.append(1)
  action, info = env._apply_action_fault(2)
  assert action == 0
  assert env._pending_action == 2
  assert info['applied'] == 1

  env = make_env()
  set_spec(env, 'action_exec', 'delayed_switch_failure')
  env._last_success_step = env._episode_step
  env._recent_actions.extend([1, 1, 1])
  action, info = env._apply_action_fault(2)
  assert action == 0
  assert env._pending_action == 2
  assert info['applied'] == 1


def test_reward_faults():
  env = make_env()
  set_spec(env, 'reward_timing', 'reward_delay_on_positive')
  reward, info = env._apply_reward_fault(np.float32(1.0), 5)
  assert float(reward) == 0.0
  assert info['applied'] == 1
  assert env._pending_reward == 1.0

  env = make_env()
  set_spec(env, 'reward_timing', 'reward_scale_half_on_positive_switch')
  reward, info = env._apply_reward_fault(np.float32(2.0), 5)
  assert float(reward) == 1.0
  assert info['applied'] == 1

  env = make_env()
  set_spec(env, 'reward_timing', 'reward_zero_after_repeat_switch')
  env._recent_actions.extend([1, 1, 1])
  reward, info = env._apply_reward_fault(np.float32(2.0), 2)
  assert float(reward) == 0.0
  assert info['applied'] == 1

  env = make_env()
  set_spec(env, 'reward_timing', 'reward_delay_after_two_rewards')
  env._recent_rewards.extend([1.0, 0.0, 1.0])
  reward, info = env._apply_reward_fault(np.float32(2.0), 5)
  assert float(reward) == 0.0
  assert info['applied'] == 1
  assert env._pending_reward == 2.0


def test_semantic_faults():
  env = make_env()
  set_spec(env, 'semantic_high_level', 'tool_collect_desync_on_upgrade')
  prev = env._state
  env._state = prev.replace(
      inventory=prev.inventory.replace(wood=prev.inventory.wood + 1))
  info = env._apply_semantic_fault(prev, 5, 5, np.float32(1.0), False, {})
  assert info['applied'] == 1
  assert _scalar(env._state.inventory.wood) == _scalar(prev.inventory.wood)

  env = make_env()
  set_spec(env, 'semantic_high_level', 'craft_result_missing_on_retry')
  prev = env._state
  env._state = prev.replace(
      inventory=prev.inventory.replace(
          wood_pickaxe=prev.inventory.wood_pickaxe + 1))
  info = env._apply_semantic_fault(prev, 11, 11, np.float32(1.0), False, {})
  assert info['applied'] == 1
  assert _scalar(env._state.inventory.wood_pickaxe) == _scalar(
      prev.inventory.wood_pickaxe)

  env = make_env()
  set_spec(env, 'semantic_high_level', 'station_place_ghost_on_relocate')
  prev = env._state
  pos = env._front_position(prev)
  prev = prev.replace(map=prev.map.at[pos[0], pos[1]].set(2))
  env._state = prev.replace(map=prev.map.at[pos[0], pos[1]].set(11))
  info = env._apply_semantic_fault(prev, 8, 8, np.float32(1.0), False, {})
  assert info['applied'] == 1
  assert _scalar(env._state.map[pos[0], pos[1]]) == _scalar(
      prev.map[pos[0], pos[1]])

  env = make_env()
  set_spec(
      env, 'semantic_high_level',
      'achievement_unlock_missing_after_valid_progress')
  prev = env._state
  env._state = prev.replace(achievements=prev.achievements.at[0].set(True))
  info = env._apply_semantic_fault(prev, 5, 5, np.float32(1.0), False, {})
  assert info['applied'] == 1
  assert _scalar(env._state.achievements[0]) == _scalar(prev.achievements[0])

  env = make_env()
  set_spec(env, 'semantic_high_level', 'delayed_inventory_desync_after_station_use')
  prev = env._state
  env._state = prev.replace(
      inventory=prev.inventory.replace(wood=prev.inventory.wood + 1))
  info = env._apply_semantic_fault(prev, 11, 11, np.float32(1.0), False, {})
  assert info['applied'] == 1
  assert _scalar(env._state.inventory.wood) == _scalar(prev.inventory.wood)


def test_fault_logs_have_stable_keys():
  env = make_env()
  obs = env.step({'reset': False, 'action': np.int32(0)})
  assert 'log/fault_applied' in obs
  assert 'log/fault_type_id' in obs
  assert 'log/semantic_fault_applied' in obs
  assert 'log/task_reward_raw' in obs
  assert 'log/context_inventory_bucket' in obs
  assert 'log/context_achievement_stage' in obs
  assert 'log/context_nearby_tile' in obs
  assert 'log/context_nearby_mob' in obs


if __name__ == '__main__':
  test_action_faults()
  test_reward_faults()
  test_semantic_faults()
  test_fault_logs_have_stable_keys()
  print('ALL CRAFTAX FAULT TESTS PASSED')

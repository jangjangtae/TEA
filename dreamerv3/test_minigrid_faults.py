#!/usr/bin/env python3

import numpy as np

from embodied.envs.minigrid import MiniGrid


def reset(env):
  return env.step({'reset': True, 'action': 0})


def act(env, action):
  return env.step({'reset': False, 'action': int(action)})


def face(env, target):
  base = env.unwrapped
  directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
  for direction, delta in enumerate(directions):
    pos = (target[0] - delta[0], target[1] - delta[1])
    if not (0 < pos[0] < base.width - 1 and 0 < pos[1] < base.height - 1):
      continue
    cell = base.grid.get(*pos)
    if cell is None or cell.can_overlap():
      base.agent_pos = pos
      base.agent_dir = direction
      return
  raise AssertionError(f'Could not face target {target}')


def find(env, obj_type):
  pos = env._find_object(obj_type)
  assert pos is not None, obj_type
  return pos


def assert_event(obs, fault_id):
  assert obs['log/fault_trigger_context'] == 1.0
  assert obs['log/fault_applied'] == 1.0
  assert obs['log/bug_triggered'] == 1.0
  assert obs['log/bug_id'] == fault_id
  assert obs['log/fault_count_cumulative'] >= 1


def test_clean_interactions():
  env = MiniGrid(seed=1)
  from minigrid.core.world_object import Key
  reset(env)
  key_pos = find(env, 'key')
  face(env, key_pos)
  obs = act(env, env._Actions.pickup)
  assert isinstance(env.unwrapped.carrying, Key)
  assert obs['log/fault_applied'] == 0.0
  door_pos = find(env, 'door')
  face(env, door_pos)
  act(env, env._Actions.toggle)
  door = env.unwrapped.grid.get(*door_pos)
  assert door.is_open and not door.is_locked
  env.close()


def test_broken_door():
  env = MiniGrid(fault='broken_door', episode_prob=1.0, seed=2)
  from minigrid.core.world_object import Key
  reset(env)
  door_pos = find(env, 'door')
  env.unwrapped.carrying = Key('yellow')
  face(env, door_pos)
  obs = act(env, env._Actions.toggle)
  door = env.unwrapped.grid.get(*door_pos)
  assert door.is_locked and not door.is_open
  assert_event(obs, 1)
  env.close()


def test_heavy_key():
  env = MiniGrid(fault='heavy_key', episode_prob=1.0, seed=3)
  reset(env)
  key_pos = find(env, 'key')
  face(env, key_pos)
  obs = act(env, env._Actions.pickup)
  assert env.unwrapped.carrying is None
  assert getattr(env.unwrapped.grid.get(*key_pos), 'type', None) == 'key'
  assert_event(obs, 2)
  env.close()


def test_action_flip():
  env = MiniGrid(fault='action_flip', episode_prob=1.0, seed=4)
  reset(env)
  before = int(env.unwrapped.agent_dir)
  obs = act(env, env._Actions.left)
  assert int(env.unwrapped.agent_dir) == (before + 1) % 4
  assert obs['log/executed_action'] == int(env._Actions.right)
  assert_event(obs, 3)
  env.close()


def test_teleport():
  env = MiniGrid(
      fault='teleport', episode_prob=1.0, teleport_step=0, seed=5)
  reset(env)
  before = tuple(env.unwrapped.agent_pos)
  obs = act(env, env._Actions.done)
  assert tuple(env.unwrapped.agent_pos) != before
  assert_event(obs, 4)
  env.close()


def test_door_gone():
  env = MiniGrid(
      fault='door_gone', episode_prob=1.0, structural_step=0, seed=6)
  reset(env)
  door_pos = env._door_pos
  assert door_pos is not None
  assert getattr(env.unwrapped.grid.get(*door_pos), 'type', None) == 'door'
  face(env, door_pos)
  obs = act(env, env._Actions.done)
  assert env.unwrapped.grid.get(*door_pos) is None
  assert_event(obs, 5)
  env.close()


def test_lava_gap():
  env = MiniGrid(
      fault='lava_gap', episode_prob=1.0, structural_step=0, seed=7)
  reset(env)
  goal = find(env, 'goal')
  candidates = ((goal[0] - 1, goal[1]), (goal[0], goal[1] - 1))
  lava_pos = next(
      pos for pos in candidates if env.unwrapped.grid.get(*pos) is None)
  face(env, lava_pos)
  obs = act(env, env._Actions.done)
  assert env._lava_positions
  assert getattr(env.unwrapped.grid.get(*lava_pos), 'type', None) == 'lava'
  assert_event(obs, 6)
  env.close()


def test_observations_and_spaces():
  for mode, key, shape, dtype in (
      ('symbolic', 'grid', (153,), np.float32),
      ('rgb', 'image', (64, 64, 3), np.uint8)):
    env = MiniGrid(obs_mode=mode, seed=8)
    obs = reset(env)
    assert obs[key].shape == shape
    assert obs[key].dtype == dtype
    for name, value in obs.items():
      assert value in env.obs_space[name], (name, value, env.obs_space[name])
    env.close()


def test_profile_reproducibility():
  first = MiniGrid(
      fault_profile='diagnostic', episode_prob=1.0, seed=9)
  second = MiniGrid(
      fault_profile='diagnostic', episode_prob=1.0, seed=9)
  seq1, seq2 = [], []
  for _ in range(20):
    reset(first)
    reset(second)
    seq1.append(first.fault_type)
    seq2.append(second.fault_type)
  assert seq1 == seq2
  assert len(set(seq1)) >= 4
  first.close()
  second.close()


if __name__ == '__main__':
  tests = [
      test_clean_interactions,
      test_broken_door,
      test_heavy_key,
      test_action_flip,
      test_teleport,
      test_door_gone,
      test_lava_gap,
      test_observations_and_spaces,
      test_profile_reproducibility,
  ]
  for test in tests:
    test()
    print(f'PASS {test.__name__}')
  print(f'ALL {len(tests)} MINIGRID TESTS PASSED')

"""Legacy exposure counters are not an independent manifestation oracle."""

import copy
import unittest

from dreamerv3.test_craftax_faults import make_env, set_spec
from dreamerv3.test_fate_diagnostic_followup import configured_env


class OperatorContract(unittest.TestCase):
  def test_action_space_has_seventeen_classes(self):
    env = configured_env()
    self.assertEqual(env.act_space['action'].classes, 17)

  def test_noop_delay_can_count_without_a_changed_action(self):
    env = configured_env()
    action, fault = env._apply_action_fault(0)
    empty = env._empty_fault_info()
    info = env._merge_fault_info(fault, empty, empty, 0., 0., 0, action)
    self.assertEqual(info['manifested'], 1)
    self.assertEqual(env._fault_diagnostics['new_injection'], 1)
    self.assertEqual(env._fault_diagnostics['action_changed'], 0)
    env._reset_fault_diagnostics()
    action, fault = env._apply_action_fault(0)
    info = env._merge_fault_info(fault, empty, empty, 0., 0., 0, action)
    self.assertEqual(info['manifested'], 1)
    self.assertEqual(env._fault_diagnostics['action_release'], 1)
    self.assertEqual(env._fault_diagnostics['new_injection'], 0)
    self.assertEqual(env._fault_diagnostics['action_changed'], 0)

  def test_reward_release_changes_reward_without_legacy_event(self):
    env = configured_env()
    env._fault_spec.update(family='reward_timing', type='reward_delay_on_positive')
    reward, fault = env._apply_reward_fault(1., 5)
    self.assertEqual(reward, 0.)
    self.assertEqual(fault['applied'], 1)
    env._reset_fault_diagnostics()
    reward, fault = env._apply_reward_fault(0., 5)
    empty = env._empty_fault_info()
    info = env._merge_fault_info(empty, fault, empty, 0., reward, 5, 5)
    self.assertEqual(info['manifested'], 0)
    self.assertEqual(env._fault_diagnostics['reward_release'], 1)
    self.assertEqual(env._fault_diagnostics['reward_changed'], 1)

  def test_negative_controls_never_apply_or_consume_draws(self):
    for reason in ('inactive', 'cooldown', 'no_condition'):
      with self.subTest(reason=reason):
        env = configured_env()
        if reason == 'inactive':
          env._fault_episode = 0
        elif reason == 'cooldown':
          env._fault_cooldown = 4
        else:
          env._recent_positions.clear()
        state = copy.deepcopy(env._rng.bit_generator.state)
        action, info = env._apply_action_fault(2)
        self.assertEqual(action, 2)
        self.assertEqual(info['applied'], 0)
        self.assertEqual(env._fault_diagnostics['draw_eligible'], 0)
        self.assertEqual(state, env._rng.bit_generator.state)

  def test_zero_probability_draw_is_eligible_but_not_applied(self):
    env = configured_env()
    env._fault_manifest_prob = 0
    action, info = env._apply_action_fault(2)
    self.assertEqual(action, 2)
    self.assertEqual(info['applied'], 0)
    self.assertEqual(env._fault_diagnostics['draw_eligible'], 1)
    self.assertEqual(env._fault_diagnostics['draw_success'], 0)

  def test_semantic_operator_can_be_noop_at_inventory_floor(self):
    env = make_env()
    set_spec(env, 'semantic_high_level', 'delayed_inventory_desync_after_station_use')
    prev = env._state.replace(inventory=env._state.inventory.replace(wood=1))
    env._state = prev.replace(inventory=prev.inventory.replace(wood=0))
    info = env._apply_semantic_fault(prev, 11, 11, 0., False, {})
    self.assertEqual(info['applied'], 1)
    self.assertEqual(int(env._state.inventory.wood), 0)


if __name__ == '__main__':
  unittest.main()

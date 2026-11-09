import numpy as np
from types import SimpleNamespace

from dreamerv3 import fault_score


def cfg(mode, **kwargs):
  data = dict(
      use_latent_kl=True,
      use_reward_error=False,
      w_kl=1.0,
      w_reward=0.0,
      norm_mode='none',
      clip=1.0,
      beta=0.1,
      log_only=False,
      reward_gate='nonzero_action',
      reward_mode=mode,
      reward_threshold=1.0,
      reward_delta_threshold=0.5)
  data.update(kwargs)
  return SimpleNamespace(**data)


def tran(score=1.2, prev=0.4, action=1, reward=0.0):
  return {
      'is_first': np.array(False),
      'reward': np.float32(reward),
      'log/task_reward_raw': np.float32(reward),
      'log/ref_latent_kl_surprise': np.float32(score),
      'log/ref_reward_pred': np.float32(0.0),
      'log/fault_score_prev': np.float32(prev),
      'action': np.int32(action),
      'log/context_inventory_bucket': np.int32(2),
      'log/context_achievement_stage': np.int32(1),
      'log/context_nearby_tile': np.int32(5),
      'log/context_nearby_mob': np.int32(0),
  }


def check_close(value, expected, eps=1e-6):
  assert abs(float(value) - float(expected)) < eps, (value, expected)


def test_threshold_hit():
  res = fault_score.compute_transition_fault(
      tran(score=1.2), cfg('threshold'), stats={})
  check_close(res['fault_reward_bonus'], 0.1)
  check_close(res['augmented_reward'], 0.1)
  assert int(res['fault_reward_mode_id']) == 2


def test_threshold_miss():
  res = fault_score.compute_transition_fault(
      tran(score=0.9), cfg('threshold'), stats={})
  check_close(res['fault_reward_bonus'], 0.0)
  check_close(res['augmented_reward'], 0.0)


def test_delta_hit():
  res = fault_score.compute_transition_fault(
      tran(score=1.2, prev=0.4), cfg('delta_threshold'), stats={})
  check_close(res['fault_score_delta'], 0.8)
  check_close(res['fault_reward_bonus'], 0.1)
  assert int(res['fault_reward_mode_id']) == 4


def test_nonzero_action_gate_blocks_noop():
  res = fault_score.compute_transition_fault(
      tran(score=1.2, action=0), cfg('threshold'), stats={})
  check_close(res['fault_reward_gate_active'], 0.0)
  check_close(res['fault_reward_bonus'], 0.0)
  check_close(res['augmented_reward'], 0.0)


def test_log_only_keeps_reward_unchanged():
  res = fault_score.compute_transition_fault(
      tran(score=1.2), cfg('threshold', log_only=True), stats={})
  check_close(res['fault_reward_bonus'], 0.1)
  check_close(res['augmented_reward'], 0.0)


def test_delivered_task_reward_is_default_base_reward():
  data = tran(score=0.0, reward=0.5)
  data['log/task_reward_raw'] = np.float32(1.0)
  data['log/env_reward'] = np.float32(0.5)
  res = fault_score.compute_transition_fault(
      data, cfg('dense', beta=0.0), stats={})
  check_close(res['raw_reward'], 0.5)
  check_close(res['task_reward_delivered'], 0.5)
  check_close(res['task_reward_pre_fault'], 1.0)
  check_close(res['augmented_reward'], 0.5)
  check_close(res['task_reward_source_id'], 0.0)


def test_legacy_pre_fault_reward_requires_explicit_opt_in():
  data = tran(score=0.0, reward=0.5)
  data['log/task_reward_raw'] = np.float32(1.0)
  data['log/env_reward'] = np.float32(0.5)
  res = fault_score.compute_transition_fault(
      data, cfg('dense', beta=0.0, task_reward_source='pre_fault'), stats={})
  check_close(res['raw_reward'], 1.0)
  check_close(res['augmented_reward'], 1.0)
  check_close(res['task_reward_source_id'], 1.0)


def test_transition_logs_include_new_fields():
  data = tran(score=1.2, prev=0.4)
  res = fault_score.compute_transition_fault(
      data, cfg('excess_delta_threshold', clip=2.0), stats={})
  fault_score.add_transition_fault_logs(data, res)
  for key in (
      'log/fault_score_delta',
      'log/fault_reward_score',
      'log/fault_reward_threshold',
      'log/fault_reward_mode_id'):
    assert key in data, key


def test_trace_records_rnd_signal_and_weighted_bonus():
  data = tran(score=1.2, prev=0.4)
  data.update({
      'log/rnd_intrinsic_reward': np.float32(2.0),
      'log/rnd_reward_bonus': np.float32(0.01),
      'log/rnd_alpha': np.float32(0.005),
  })
  res = fault_score.compute_transition_fault(
      data, cfg('excess_delta_threshold', clip=2.0), stats={})
  fault_score.add_transition_fault_logs(data, res)
  record = fault_score.trace_record(data, 10, 0, 1, 2)
  check_close(record['rnd_intrinsic_reward'], 2.0)
  check_close(record['rnd_reward_bonus'], 0.01)
  check_close(record['rnd_alpha'], 0.005)


def test_context_p95_uses_specific_stats():
  data = tran(score=3.0, action=1)
  key = fault_score.context_keys(data)['full']
  stats = {
      'fault_score_mean': 1.0,
      'fault_score_std': 1.0,
      'fault_score_p95': 4.0,
      'context_stats': {
          key: {
              'fault_score_mean': 1.0,
              'fault_score_std': 0.5,
              'fault_score_p95': 2.0,
              'fault_score_count': 100,
          },
      },
  }
  res = fault_score.compute_transition_fault(
      data, cfg('threshold', norm_mode='context_p95', context_min_count=50),
      stats=stats)
  check_close(res['fault_score'], 1.5)
  check_close(res['fault_context_p95'], 2.0)
  assert int(res['fault_context_level_id']) == 3


def test_context_p95_falls_back_when_sparse():
  data = tran(score=3.0, action=1)
  key = fault_score.context_keys(data)['full']
  stats = {
      'fault_score_mean': 1.0,
      'fault_score_std': 1.0,
      'fault_score_p95': 4.0,
      'context_stats': {
          key: {'fault_score_p95': 2.0, 'fault_score_count': 2},
      },
  }
  res = fault_score.compute_transition_fault(
      data, cfg('threshold', norm_mode='context_p95', context_min_count=50),
      stats=stats)
  check_close(res['fault_score'], 0.75)
  assert int(res['fault_context_level_id']) == 0


def test_diversity_reward_and_repeat_penalty():
  config = cfg(
      'threshold', beta=0.0, semantic_coverage_beta=0.02,
      unique_suspicious_beta=0.05, repeat_suspicious_penalty=0.01,
      suspicious_threshold=1.0)
  tracker = fault_score.FaultRewardTracker(config)
  data = tran(score=1.2)
  first = fault_score.compute_transition_fault(data, config, stats={})
  tracker.apply(data, 0, first)
  check_close(first['fault_reward_bonus_total'], 0.07)
  repeated = fault_score.compute_transition_fault(data, config, stats={})
  tracker.apply(data, 0, repeated)
  check_close(repeated['fault_reward_bonus_total'], -0.01)
  check_close(repeated['repeated_suspicious_context'], 1.0)


def test_adaptive_beta_reduces_below_target():
  config = cfg(
      'dense', beta=0.2, adaptive_beta=True, adaptive_task_target=1.0,
      adaptive_beta_down=0.5, adaptive_beta_up=1.1,
      adaptive_beta_min=0.01, adaptive_beta_max=0.5)
  tracker = fault_score.FaultRewardTracker(config)
  data = tran(score=0.0, reward=0.2)
  data['is_last'] = np.array(True)
  res = fault_score.compute_transition_fault(
      data, config, stats={}, beta_override=tracker.beta)
  tracker.apply(data, 0, res)
  check_close(tracker.beta, 0.1)


def test_coverage_uses_full_context_not_calibration_fallback():
  config = cfg('dense', beta=0.0, semantic_coverage_beta=0.02)
  tracker = fault_score.FaultRewardTracker(config)
  first_data = tran(score=0.0, action=1)
  first = fault_score.compute_transition_fault(first_data, config, stats={})
  assert first['fault_context_key'] == 'global'
  tracker.apply(first_data, 0, first)
  second_data = tran(score=0.0, action=2)
  second = fault_score.compute_transition_fault(second_data, config, stats={})
  tracker.apply(second_data, 0, second)
  check_close(first['semantic_context_new'], 1.0)
  check_close(second['semantic_context_new'], 1.0)
  check_close(second['semantic_context_coverage_episode'], 2.0)


def test_kl_bound_score_source():
  data = tran(score=99.0)
  data['log/ref_novelty_kl_lhs'] = np.float32(1.2)
  data['log/ref_novelty_kl_bound'] = np.float32(0.5)
  data['log/ref_novelty_bound_violation'] = np.float32(0.7)
  data['log/ref_novelty_bound_score'] = np.float32(0.7)
  data['log/ref_novelty_bound_triggered'] = np.float32(1.0)
  config = cfg('dense', score_source='kl_bound', clip=1.0)
  result = fault_score.compute_transition_fault(data, config, stats={})
  check_close(result['fault_score_raw'], 0.7)
  check_close(result['novelty_bound_triggered'], 1.0)


def test_task_constraint_increases_lambda_below_target():
  config = cfg(
      'dense', beta=0.0, constraint_mode='task_lower_bound',
      constraint_task_target=1.0, constraint_lambda_init=1.0,
      constraint_lambda_lr=0.5, constraint_lambda_min=0.0,
      constraint_lambda_max=10.0, constraint_novelty_scale=1.0,
      constraint_task_scale=1.0, constraint_warmup_episodes=0)
  tracker = fault_score.FaultRewardTracker(config)
  data = tran(score=0.0, reward=0.2)
  data['is_last'] = np.array(True)
  result = fault_score.compute_transition_fault(data, config, stats={})
  tracker.apply(data, 0, result)
  check_close(result['augmented_reward'], 0.2)
  check_close(tracker.constraint_lambda, 1.4)
  check_close(result['constraint_violation'], 0.8)


def test_task_constraint_warmup_is_task_only():
  config = cfg(
      'dense', beta=1.0, constraint_mode='task_lower_bound',
      constraint_task_target=1.0, constraint_lambda_init=1.0,
      constraint_lambda_lr=0.1, constraint_novelty_scale=1.0,
      constraint_task_scale=1.0, constraint_warmup_episodes=10)
  tracker = fault_score.FaultRewardTracker(config)
  data = tran(score=10.0, reward=0.2)
  result = fault_score.compute_transition_fault(data, config, stats={})
  tracker.apply(data, 0, result)
  check_close(result['constraint_active'], 0.0)
  check_close(result['augmented_reward'], 0.2)


def test_scaled_task_constraint_uses_beta_scaled_bonus():
  config = cfg(
      'dense', beta=0.2, constraint_mode='task_lower_bound_scaled',
      constraint_task_target=1.0, constraint_lambda_init=1.0,
      constraint_lambda_lr=0.1, constraint_novelty_scale=1.0,
      constraint_task_scale=1.0, constraint_warmup_episodes=0)
  tracker = fault_score.FaultRewardTracker(config)
  data = tran(score=10.0, reward=0.2)
  result = fault_score.compute_transition_fault(data, config, stats={})
  tracker.apply(data, 0, result)
  check_close(result['constraint_active'], 1.0)
  check_close(result['fault_reward_bonus'], 0.2)
  check_close(result['constraint_novelty_reward'], 0.2)
  check_close(result['constraint_task_reward'], 0.2)
  check_close(result['augmented_reward'], 0.4)


def run_all():
  test_threshold_hit()
  test_threshold_miss()
  test_delta_hit()
  test_nonzero_action_gate_blocks_noop()
  test_log_only_keeps_reward_unchanged()
  test_delivered_task_reward_is_default_base_reward()
  test_legacy_pre_fault_reward_requires_explicit_opt_in()
  test_transition_logs_include_new_fields()
  test_trace_records_rnd_signal_and_weighted_bonus()
  test_context_p95_uses_specific_stats()
  test_context_p95_falls_back_when_sparse()
  test_diversity_reward_and_repeat_penalty()
  test_adaptive_beta_reduces_below_target()
  test_coverage_uses_full_context_not_calibration_fallback()
  test_kl_bound_score_source()
  test_task_constraint_increases_lambda_below_target()
  test_task_constraint_warmup_is_task_only()
  test_scaled_task_constraint_uses_beta_scaled_bonus()
  print('ALL 18 FAULT REWARD MODE TESTS PASSED')


if __name__ == '__main__':
  run_all()

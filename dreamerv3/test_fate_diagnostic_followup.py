"""CPU checks for passive instrumentation, sparse rewards, and queue guards."""

import copy
import functools
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import jax
import numpy as np

from dreamerv3.fate_diagnostic_followup import (
    DIAGNOSTICS, METHODS, SPLITS, calibrated_sparse, check_diagnostic_rows,
    digest, make_manifest, write_json)
from dreamerv3.fault_score import compute_fault_bonus, compute_transition_fault, trace_record
from dreamerv3.test_fate_revision_queue import QueueProtocol
from embodied.envs.craftax import Craftax, _FAULT_TYPE_IDS


def test_calibration_matches_rate_and_mean_with_same_ramp_width():
  scores = np.linspace(0.01, 3, 1000)
  delta = np.zeros_like(scores)
  delta[::10] = np.linspace(0.6, 2.5, 100)
  result = calibrated_sparse(scores, delta)
  assert abs(result['fate_positive_rate'] - result['sparse_positive_rate']) <= 0.002
  assert math.isclose(result['fate_mean_bonus'], result['sparse_mean_bonus'])
  assert math.isclose(result['clip'] - result['threshold'], 1.5)
  for value in (0, result['threshold'], result['threshold'] + 0.75, result['clip'] + 10):
    cfg = dict(reward_mode='excess_threshold', reward_threshold=result['threshold'], clip=result['clip'])
    actual = compute_fault_bonus(value, min(value, result['clip']), result['beta'], cfg)[0]
    expected = result['beta'] * np.clip((value - result['threshold']) / 1.5, 0, 1)
    assert math.isclose(actual, expected, abs_tol=1e-10)


def test_degenerate_calibration_is_rejected():
  with unittest.TestCase().assertRaises(ValueError):
    calibrated_sparse(np.ones(1000), np.zeros(1000))
  with unittest.TestCase().assertRaises(ValueError):
    calibrated_sparse(np.ones(1000), np.tile([0, 1], 500))
  with unittest.TestCase().assertRaises(ValueError):
    calibrated_sparse(np.full(1000, np.nan), np.ones(1000))


def configured_env():
  env = Craftax(logs=True, platform='cpu')
  env._fault_spec = dict(family='action_exec', type='revisit_action_delay', severity=1.0)
  env._fault_episode = 1
  env._fault_manifest_prob = 1
  env._recent_positions.append(None)
  env._recent_actions.append(1)
  return env


def test_pending_action_release_is_not_a_new_draw():
  env = configured_env()
  _, event = env._apply_action_fault(2)
  assert event['applied'] == 1
  assert env._fault_diagnostics['new_injection'] == 1
  assert env._fault_diagnostics['draw_eligible'] == 1
  env._reset_fault_diagnostics()
  action, event = env._apply_action_fault(3)
  assert action == 2 and event['applied'] == 1
  assert env._fault_diagnostics['action_release'] == 1
  assert env._fault_diagnostics['draw_eligible'] == 0
  assert env._fault_diagnostics['new_injection'] == 0


def test_reward_release_logged_without_changing_legacy_events():
  env = configured_env()
  env._fault_spec.update(family='reward_timing', type='reward_delay_on_positive')
  reward, event = env._apply_reward_fault(1, 5)
  assert reward == 0 and event['applied'] == 1
  env._reset_fault_diagnostics()
  reward, event = env._apply_reward_fault(0, 5)
  assert reward == 1 and event['applied'] == 0
  assert env._fault_diagnostics['reward_release'] == 1
  assert env._fault_diagnostics['new_injection'] == 0


def test_cooldown_does_not_consume_randomness_and_assignment_survives_no_event():
  env = configured_env()
  env._fault_cooldown = 4
  before = copy.deepcopy(env._rng.bit_generator.state)
  action, info = env._apply_action_fault(2)
  assert action == 2 and info['applied'] == 0
  assert env._fault_diagnostics['condition_met'] == 1
  assert env._fault_diagnostics['draw_eligible'] == 0
  assert env._rng.bit_generator.state == before
  env._last_fault_info = info
  logs = env._fault_log_obs()
  assert logs['log/fault_type_id'] == 0
  assert logs['log/fault_assigned_type_id'] == _FAULT_TYPE_IDS['revisit_action_delay']


def test_new_labels_never_change_reward_and_are_written_to_trace():
  tran = dict(reward=0.5, is_first=False, **{
      'log/env_reward': 0.5, 'log/task_reward_raw': 1,
      'log/ref_latent_kl_surprise': 1.8, 'log/fault_score_prev': 0.2})
  cfg = dict(norm_mode='p95', reward_mode='excess_delta_threshold', beta=0.2,
             reward_delta_threshold=0.5, clip=2, log_only=False, w_reward=0)
  baseline = compute_transition_fault(tran, cfg, {'fault_score_p95': 1})
  for key in DIAGNOSTICS:
    tran[f'log/fault_{key}'] = 1
  tran['log/fault_assigned_type_id'] = 34
  assert compute_transition_fault(tran, cfg, {'fault_score_p95': 1}) == baseline
  record = trace_record(tran, 1, 0, 1, 1)
  assert record['log/fault_assigned_type_id'] == 34
  for key in DIAGNOSTICS:
    assert record[f'log/fault_{key}'] == 1


def diagnostic_row(first=False, **values):
  result = dict(worker=0, is_first=first, global_step=1, bug_triggered=0,
                fault_reward_bonus=0, task_reward_delivered=0)
  result.update({f'log/fault_{key}': 0 for key in DIAGNOSTICS})
  result['log/fault_assigned_type_id'] = 18
  result['log/fault_draw_probability'] = 0.03
  result.update(values)
  return result


def test_opportunity_audit_rejects_bad_denominators_and_reset_carry():
  first = diagnostic_row(True)
  row = diagnostic_row(**{'log/fault_condition_met': 1, 'log/fault_draw_eligible': 1})
  counts, _ = check_diagnostic_rows([first, row], 'sparse')
  assert counts['revisit_action_delay']['draw_eligible'] == 1
  assert counts['revisit_action_delay']['legacy_events'] == 0
  with unittest.TestCase().assertRaises(ValueError):
    check_diagnostic_rows([first, dict(row, **{'log/fault_cooldown_remaining': 4})], 'sparse')
  with unittest.TestCase().assertRaises(ValueError):
    check_diagnostic_rows([dict(first, **{'log/fault_new_injection': 1})], 'sparse')
  with unittest.TestCase().assertRaises(ValueError):
    check_diagnostic_rows([first, dict(row, **{'log/fault_assigned_type_id': 0})], 'sparse')


def test_original_and_instrumented_environment_identical():
  for profile in ('clean', 'benchmark_seen', 'benchmark_holdout', 'benchmark_sparse'):
    environment = {k: v for k, v in os.environ.items() if not k.startswith('CRAFTAX_')}
    environment.update(CRAFTAX_FAULT_SAMPLER='0' if profile == 'clean' else '1',
                       CRAFTAX_FAULT_SEED='17')
    if profile != 'clean':
      environment.update(CRAFTAX_FAULT_PROFILE=profile, CRAFTAX_FAULT_EP_PROB='1')
    with mock.patch.dict(os.environ, environment, clear=True):
      _compare_environment(profile)


def _compare_environment(profile):
  filename = os.environ.get('FATE_LEGACY_CRAFTAX')
  if not filename:
    raise unittest.SkipTest('Set FATE_LEGACY_CRAFTAX to the immutable pre-instrumentation source')
  spec = importlib.util.spec_from_file_location('legacy_craftax_test', filename)
  legacy = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(legacy)
  old = legacy.Craftax(seed=17, logs=True, platform='cpu', length=40)
  new = Craftax(seed=17, logs=True, platform='cpu', length=40)
  rng = np.random.default_rng(42)
  try:
    for index in range(100):
      action = dict(reset=index in (0, 67), action=np.int32(rng.integers(0, 17)))
      expected, actual = old.step(action), new.step(action)
      for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f'{profile}/{index}/{key}')
      assert old._rng.bit_generator.state == new._rng.bit_generator.state
      for left, right in zip(jax.tree_util.tree_leaves(old._state), jax.tree_util.tree_leaves(new._state)):
        np.testing.assert_array_equal(left, right)
      assert set(actual) == set(new.obs_space)
  finally:
    old.close()
    new.close()


class SparseQueueCommands(QueueProtocol):

  def test_sparse_parameters_survive_train_and_all_evaluations(self):
    self.env.update(SPARSE_BETA='0.03', SPARSE_THRESHOLD='1.4', SPARSE_CLIP='2.9')
    self.run_shell(f'''MAIN_PY="{self.capture}"
run_train_variant 0 sparse_surprise_matched /tmp/stats.json
run_job() {{ shift 2; "$@"; }}
EVAL_SPLITS='clean seen holdout sparse'
run_variant_evals 0 sparse_surprise_matched /tmp/checkpoint /tmp/stats.json
''')
    calls = [json.loads(line) for line in Path(self.env['CAPTURE']).read_text().splitlines()]
    assert len(calls) == 5
    for call in calls:
      args = call['argv']
      for key, value in (('fault.beta', '0.03'), ('fault.reward_threshold', '1.4'),
                         ('fault.clip', '2.9'), ('fault.reward_mode', 'excess_threshold')):
        assert args[args.index('--' + key) + 1] == value
    assert calls[0]['env']['CRAFTAX_FAULT_PROFILE'] == 'benchmark_train'
    assert calls[1]['env']['CRAFTAX_FAULT_SAMPLER'] == '0'
    for call in calls[1:]:
      assert call['env']['CRAFTAX_RND_UPDATE'] == '0'

  def test_invalid_sparse_clip_fails(self):
    self.env.update(SPARSE_BETA='0.03', SPARSE_THRESHOLD='2.0', SPARSE_CLIP='1.0')
    self.run_shell('variant_params sparse_surprise_matched', success=False)


def shell_library():
  source = (Path(__file__).parent / 'run_craftax_fate_diagnostic_tail.sh').read_text()
  return 'record()' + source.split('record()', 1)[1].split('\nwait_for_controls\n', 1)[0]


def temporary_test(function):
  @functools.wraps(function)
  def wrapped():
    with tempfile.TemporaryDirectory() as directory:
      return function(Path(directory))
  return wrapped


@temporary_test
def test_tail_parent_failure_blocks_and_completion_unblocks(tmp_path):
  env = dict(os.environ, FOLLOWUP_ROOT=str(tmp_path), CONTROLS_SERVICE='controls.service')
  for state, status, success in (('failed', '0', False), ('inactive', '1', False), ('inactive', '0', True)):
    body = f'''systemctl() {{ if [[ "$*" == *ActiveState* ]]; then echo {state}; else echo {status}; fi; }}
wait_for_controls
'''
    result = subprocess.run(['bash', '-euo', 'pipefail', '-c', shell_library() + body],
                             env=env, capture_output=True, text=True, timeout=10)
    assert (result.returncode == 0) == success, result.stderr


@temporary_test
def test_tail_smoke_budget_does_not_leak_to_pilot(tmp_path):
  capture = tmp_path / 'capture.sh'
  capture.write_text('''#!/bin/bash
printf '%s %s %s %s\n' "$SEEDS" "$TRAIN_STEPS" "$ADAPT_CLEAN_EVAL_STEPS" "$SPARSE_EVAL_STEPS" >> "$FOLLOWUP_ROOT/calls"
''')
  env = dict(os.environ, FOLLOWUP_ROOT=str(tmp_path), QUEUE=str(capture), PYTHON_BIN='true',
             TOOL='unused', TRAIN_STEPS='1000000', ADAPT_CLEAN_EVAL_STEPS='30000', SPARSE_EVAL_STEPS='60000')
  body = 'sparse_environment() { :; }\ngpu_smoke\npilot\n'
  result = subprocess.run(['bash', '-euo', 'pipefail', '-c', shell_library() + body],
                           env=env, capture_output=True, text=True, timeout=10)
  assert result.returncode == 0, result.stderr
  assert (tmp_path / 'calls').read_text().splitlines() == [
      '0 4096 256 256', '0 1000000 30000 60000', '1 1000000 30000 60000']


@temporary_test
def test_failed_smoke_never_starts_full_jobs(tmp_path):
  env = dict(os.environ, FOLLOWUP_ROOT=str(tmp_path))
  result = subprocess.run(['bash', '-euo', 'pipefail', '-c', shell_library() +
      '\nrun_stage gpu_smoke false\ntouch "$FOLLOWUP_ROOT/full_started"\n'],
      env=env, capture_output=True, text=True, timeout=10)
  assert result.returncode != 0
  assert not (tmp_path / 'full_started').exists()
  assert 'DONE' not in (tmp_path / 'status.tsv').read_text()


@temporary_test
def test_manifest_requires_completed_controls_and_pins_sources(tmp_path):
  parent, controls = tmp_path / 'parent', tmp_path / 'controls'
  parent.mkdir()
  controls.mkdir()
  (parent / 'status.tsv').write_text('now\tqueue\tDONE\n')
  (controls / 'status.tsv').write_text('now\tcontrols\tSTART\n')
  out = tmp_path / 'manifest.json'
  with unittest.TestCase().assertRaises(ValueError):
    make_manifest(parent, controls, out)
  assert not out.exists()
  (controls / 'status.tsv').write_text('now\tqueue\tDONE\n')
  checkpoint = tmp_path / 'ckpt'
  checkpoint.mkdir()
  (checkpoint / 'done').touch()
  (checkpoint / 'agent.pkl').write_bytes(b'checkpoint')
  hashes = {}
  for seed in range(5):
    stats = parent / f'core/seed_{seed}/craftax_clean_fault_stats.json'
    write_json(stats, dict(fault_score_p95=2))
    hashes[str(seed)] = digest(stats)
  protocol = dict(checkpoint=str(checkpoint), agent_sha256=digest(checkpoint / 'agent.pkl'))
  write_json(parent / 'protocol.json', protocol)
  write_json(controls / 'protocol.json', dict(protocol, calibration_sha256=hashes))
  write_json(controls / 'controls/clean_control_audit.json', {
      str(i): dict(rows=1000000, fault_events=0, fault_episodes=0) for i in range(10)})
  states = defaultdict(list)
  for seed in range(5):
    for method, group, variant in METHODS:
      root = controls / 'controls' if group == 'controls' else parent / group
      if variant:
        states[root].append(f'now\tseed={seed}\ttrain_{variant}\tDONE\n')
        write_json(root / f'seed_{seed}/train_{variant}/train/config.yaml',
                   dict(run=dict(steps=1000000), fault=dict(task_reward_source='delivered')))
      for split in SPLITS:
        name = f'eval_{variant}_{split}' if variant else f'base_{split}_eval'
        states[root].append(f'now\tseed={seed}\t{name}\tDONE\n')
        write_json(root / f'job_audits/seed_{seed}_{name}.json', {})
        write_json(root / f'seed_{seed}/{name}/config.yaml', dict(script='eval_only', seed=seed,
            run=dict(steps=30000 if split == 'clean' else 60000, from_checkpoint=str(checkpoint)),
            fault=dict(log_only=True)))
  for root, lines in states.items():
    (root / 'status.tsv').write_text(''.join(lines))
  make_manifest(parent, controls, out)
  make_manifest(parent, controls, out)
  assert len(json.loads(out.read_text())['cases']) == 140
  (checkpoint / 'agent.pkl').write_bytes(b'changed')
  with unittest.TestCase().assertRaises(ValueError):
    make_manifest(parent, controls, out)


def load_tests(loader, tests, pattern):
  for name, function in globals().items():
    if name.startswith('test_') and callable(function):
      tests.addTest(unittest.FunctionTestCase(function))
  return tests


if __name__ == '__main__':
  unittest.main()

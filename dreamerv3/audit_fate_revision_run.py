"""Fail-closed checks on completed FATE revision smoke artifacts."""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML


def require(condition, message):
  if not condition:
    raise ValueError(message)


def close(a, b):
  return math.isclose(float(a), float(b), rel_tol=2e-5, abs_tol=2e-6)


def read_config(path):
  return YAML(typ='safe').load(path.read_text())


def check_clean_replay_history(root):
  directory = Path(root) / 'train'
  config = read_config(directory / 'config.yaml')
  workers = int(config['run']['envs'])
  with (directory / 'metrics.jsonl').open() as handle:
    streams = [int(row['replay/streams']) for line in handle
               if (row := json.loads(line)).get('step', 0) >= 2 * workers
               and 'replay/streams' in row]
  require(streams, 'Clean initialization has no replay-stream audit evidence')
  require(all(count == workers for count in streams),
          f'Unsafe clean initialization: envs={workers}, recorded replay streams='
          f'{sorted(set(streams))}. Retrain the clean model with separated streams.')
  return dict(clean_root=str(Path(root).resolve()), envs=workers,
              replay_streams=sorted(set(streams)), checked_log_rows=len(streams))


def check_trace(directory, config, stats):
  fault = config['fault']
  require(fault['task_reward_source'] == 'delivered', 'Non-delivered task channel')
  require(fault['w_kl'] == 1 and fault['w_reward'] == 0,
          'Calibration/training score is not pure latent KL')
  require(fault['norm_mode'] == 'p95', 'Unexpected normalization')
  previous = {}
  workers = set()
  count = real_steps = rnd_positive = bonus_positive = 0
  last_global = 0
  training = config['script'] == 'train'
  apply_bonus = training and not fault['log_only']
  with (directory / 'fault_score_trace.jsonl').open() as handle:
    for line in handle:
      row = json.loads(line)
      count += 1
      workers.add(row['worker'])
      for key in ('fault_score_raw', 'fault_score', 'latent_kl_surprise',
                  'learning_reward', 'augmented_reward', 'rnd_reward_bonus',
                  'task_reward_delivered', 'fault_reward_bonus'):
        require(math.isfinite(row[key]), f'{directory}: non-finite {key}')
      require(row['global_step'] > last_global, 'Repeated/restarted trace step')
      last_global = row['global_step']
      require(close(row['fault_score_raw'], row['latent_kl_surprise']),
              'Score differs from KL')
      normalized = row['fault_score_raw'] / stats.get('fault_score_p95', 1.0)
      require(close(row['fault_score'], normalized), 'Incorrect p95 normalization')
      prev = 0.0 if row['is_first'] else previous.get(row['worker'], 0.0)
      require(close(row['fault_score_prev'], prev), 'Cross-worker score history')
      require(close(row['fault_score_delta'], max(0, normalized - prev)),
              'Temporal score mismatch')
      previous[row['worker']] = row['fault_score']
      require(row['task_reward_source_id'] == 0, 'Legacy task reward used')
      require(close(row['raw_reward'], row['task_reward_delivered']),
              'Base task reward differs from delivered reward')
      require(close(row['rnd_reward_bonus'],
                    row['rnd_alpha'] * row['rnd_intrinsic_reward']), 'RND bonus mismatch')
      expected = row['task_reward_delivered'] + row['rnd_reward_bonus']
      if apply_bonus:
        expected += row['fault_reward_bonus']
      require(close(row['learning_reward'], expected), 'Wrong actual learning reward')
      require(close(row['augmented_reward'], expected), 'Reward log differs from replay reward')
      if row['is_first']:
        require(close(row['fault_reward_bonus'], 0), 'Bonus paid on reset')
      else:
        real_steps += 1
      rnd_positive += row['rnd_reward_bonus'] > 0
      bonus_positive += apply_bonus and row['fault_reward_bonus'] > 0
  budget = int(config['run']['steps'])
  require(budget <= count < budget + max(10, config['run']['envs']),
          f'{directory}: incomplete or repeated rollout: {count}/{budget}')
  require(len(workers) == config['run']['envs'], 'Missing environment workers')
  return dict(rows=count, non_reset_transitions=real_steps,
              workers=len(workers), rnd_positive_steps=rnd_positive,
              fault_bonus_positive_steps=bonus_positive)


def check_training(directory, config):
  metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
  streams = [row['replay/streams'] for row in metrics if 'replay/streams' in row]
  require(streams and streams[-1] == config['run']['envs'], 'Replay streams not separated')
  losses = [value for row in metrics for key, value in row.items()
            if key.startswith('train/loss/')]
  require(losses and all(math.isfinite(x) for x in losses), 'Missing/nonfinite training losses')
  latest = (directory / 'ckpt' / 'latest').read_text().strip()
  checkpoint = directory / 'ckpt' / latest
  require((checkpoint / 'done').is_file(), 'Incomplete checkpoint')
  # Only local, trusted run checkpoints are loaded here.
  with (checkpoint / 'step.pkl').open('rb') as handle:
    step = pickle.load(handle)
  require(int(step) >= config['run']['steps'], 'Final checkpoint below budget')
  with (checkpoint / 'agent.pkl').open('rb') as handle:
    trained = pickle.load(handle)['params']
  with (Path(config['run']['from_checkpoint']) / 'agent.pkl').open('rb') as handle:
    initial = pickle.load(handle)['params']
  changed = {prefix: 0 for prefix in ('enc/', 'dyn/', 'pol/')}
  for key, value in trained.items():
    require(np.isfinite(value).all(), f'Non-finite checkpoint parameter {key}')
    for prefix in changed:
      if key.startswith(prefix) and not np.array_equal(value, initial[key]):
        changed[prefix] += 1
  require(all(changed.values()), f'Model/actor did not update: {changed}')
  return dict(checkpoint_step=int(step), changed_parameter_tensors=changed,
              finite_loss_values=len(losses), replay_streams=streams[-1])


def audit(root):
  root = Path(root)
  seed = root / 'seed_0'
  stats = json.loads((seed / 'craftax_clean_fault_stats.json').read_text())
  require(stats['fault_score_p95'] > 0, 'Invalid clean p95')
  calibration = seed / 'base_clean_eval'
  config = read_config(calibration / 'config.yaml')
  require(config['run']['steps'] == 256 and config['run']['envs'] == 1,
          'Calibration is not from the expanded smoke protocol')
  result = {'calibration': check_trace(calibration, config, {})}
  require(close(stats['fault_score_p95'], stats['latent_kl_p95']),
          'Calibration includes reward error')
  reference = config['fault']['ref_ckpt']
  for variant in ('reward_scale_probe', 'dense_matched', 'rnd_matched',
                  'excess_delta_p95_beta02'):
    directory = seed / f'train_{variant}' / 'train'
    config = read_config(directory / 'config.yaml')
    require(config['run']['envs'] == 16 and config['run']['steps'] == 4096
            and config['run']['train_ratio'] == 128
            and config['batch_size'] == 16 and config['batch_length'] == 64,
            f'{variant}: stale or undersized smoke configuration')
    require(Path(config['fault']['norm_stats']).resolve() ==
            (seed / 'craftax_clean_fault_stats.json').resolve(), 'Wrong calibration file')
    expected_log_only = variant in ('reward_scale_probe', 'rnd_matched')
    expected_beta = (0 if expected_log_only else
                     0.2 if variant == 'excess_delta_p95_beta02' else 0.001)
    require(config['fault']['log_only'] == expected_log_only and
            config['fault']['beta'] == expected_beta, 'Wrong smoke reward configuration')
    require(config['fault']['ref_ckpt'] == reference and
            config['run']['from_checkpoint'] == reference, 'Initialization mismatch')
    result[variant] = check_trace(directory, config, stats)
    result[variant].update(check_training(directory, config))
    if variant == 'rnd_matched':
      require(result[variant]['rnd_positive_steps'] > 0, 'RND inactive')
    if variant in ('dense_matched', 'excess_delta_p95_beta02'):
      require(result[variant]['fault_bonus_positive_steps'] > 0, 'Shaping inactive')
    evaluation = seed / f'eval_{variant}_seen'
    eval_config = read_config(evaluation / 'config.yaml')
    require(eval_config['script'] == 'eval_only', 'Evaluation can update model')
    require(eval_config['run']['steps'] == 256, 'Wrong smoke evaluation budget')
    latest = (directory / 'ckpt' / 'latest').read_text().strip()
    require(Path(eval_config['run']['from_checkpoint']) == directory / 'ckpt' / latest,
            'Evaluation loaded wrong checkpoint')
    result[variant]['evaluation'] = check_trace(evaluation, eval_config, stats)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--root')
  group.add_argument('--check-clean-root')
  args = parser.parse_args()
  if args.check_clean_root:
    print(json.dumps(check_clean_replay_history(args.check_clean_root), indent=2))
    return
  result = audit(args.root)
  path = Path(args.root) / 'integrity_audit.json'
  path.write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))
  print(f'PASS: {path}')


if __name__ == '__main__':
  main()

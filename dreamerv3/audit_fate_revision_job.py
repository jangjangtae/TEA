"""Validate revision jobs before the unattended queue records them as DONE."""

import argparse
import json
from pathlib import Path

from dreamerv3.audit_fate_revision_run import (
    check_clean_replay_history, check_trace, check_training, close,
    read_config, require)


def audit_job(root, seed, name):
  directory = Path(root) / f'seed_{seed}'
  if name == 'make_norm_stats':
    stats = json.loads((directory / 'craftax_clean_fault_stats.json').read_text())
    require(stats['fault_score_p95'] > 0, 'Non-positive clean p95')
    require(close(stats['fault_score_p95'], stats['latent_kl_p95']),
            'Calibration is not KL-only')
    return {'calibration_p95': stats['fault_score_p95']}
  if name.startswith('train_'):
    directory = directory / name / 'train'
  elif name.startswith(('base_', 'eval_')):
    directory = directory / name.replace('_for_stats', '')
  else:
    return None
  config = read_config(directory / 'config.yaml')
  training = name.startswith('train_')
  require(config['script'] == ('train' if training else 'eval_only'), 'Wrong job mode')
  require(config['task'] == 'craftax_classic_pixels', 'Wrong environment')
  require(config['fault']['enabled'], 'Missing reference instrumentation')
  reference = config['fault']['ref_ckpt']
  if training:
    require(config['run']['from_checkpoint'] == reference, 'Different initialization/reference')
  elif name.startswith('eval_'):
    variant = name[len('eval_'):].rsplit('_', 1)[0]
    checkpoint_root = Path(root) / f'seed_{seed}/train_{variant}/train/ckpt'
    checkpoint = checkpoint_root / (checkpoint_root / 'latest').read_text().strip()
    require(Path(config['run']['from_checkpoint']).resolve() == checkpoint.resolve(),
            'Wrong evaluation checkpoint')
  else:
    require(config['run']['from_checkpoint'] == reference, 'Wrong CleanEval checkpoint')
  stats_path = config['fault']['norm_stats']
  stats = json.loads(Path(stats_path).read_text()) if stats_path else {}
  if stats:
    require(stats['fault_score_p95'] > 0, 'Invalid normalization')
  result = check_trace(directory, config, stats)
  if training:
    result.update(check_training(directory, config))
    check_clean_replay_history(directory.parent)
    if name.startswith('train_rnd_matched'):
      require(result['rnd_positive_steps'] > 0, 'RND never contributed reward')
    if name.startswith(('train_dense_matched', 'train_excess_delta')):
      require(result['fault_bonus_positive_steps'] > 0, 'Shaping never contributed reward')
  else:
    require(result['rnd_positive_steps'] == 0, 'RND leaked into evaluation')
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root', type=Path, required=True)
  parser.add_argument('--seed', required=True)
  parser.add_argument('--name', required=True)
  args = parser.parse_args()
  result = audit_job(args.root, args.seed, args.name)
  if result is not None:
    path = args.root / 'job_audits' / f'seed_{args.seed}_{args.name}.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(f'PASS: {path}')


if __name__ == '__main__':
  main()

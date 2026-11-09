"""Clean-only matched novelty extension; no outcome-based coefficient selection."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dreamerv3.analyze_craftax_multiseed import iter_jsonl
from dreamerv3.analyze_fate_corrected_completed import aggregate, table
from dreamerv3.audit_fate_revision_run import read_config, require
from dreamerv3.calibrate_intrinsic_reward_scales import calibrate as validate_probe
from dreamerv3.fate_diagnostic_followup import (
    SPLITS, audit_eval, calibrated_sparse, digest, write_json)


VARIANTS = dict(dense_matched='CleanAdaptDenseSurprise', rnd_matched='CleanAdaptRND',
                sparse_surprise_matched='CleanAdaptSparseSurprise')
FAULT_KEYS = ('bug_triggered', 'log/fault_applied', 'log/fault_episode',
              'log/semantic_fault_applied', 'log/semantic_fault_episode')


def require_clean(row):
  require(all(row[key] == 0 for key in FAULT_KEYS), 'Fault contamination in Clean adaptation')
  require(math.isclose(row['task_reward_delivered'], row['task_reward_pre_fault'],
                      rel_tol=2e-5, abs_tol=2e-6), 'Clean task reward was corrupted')


def matched_parameters(scores, deltas, rnd):
  scores, deltas, rnd = map(np.asarray, (scores, deltas, rnd))
  require(scores.shape == deltas.shape == rnd.shape and
          all(np.isfinite(x).all() and (x >= 0).all() for x in (scores, deltas, rnd)),
          'Invalid aligned probe signals')
  sparse = calibrated_sparse(scores, deltas)
  target = sparse['fate_mean_bonus']
  dense_mean, rnd_mean = float(np.clip(scores, 0, 2).mean()), float(rnd.mean())
  require(dense_mean > 1e-8 and rnd_mean > 1e-8, 'Degenerate intrinsic probe')
  dense_beta, rnd_alpha = target / dense_mean, target / rnd_mean
  require(all(math.isfinite(x) and 0 < x <= 10 for x in (dense_beta, rnd_alpha)),
          'Extreme coefficient; inspect probe, do not silently clip')
  return dict(dense_beta=dense_beta, rnd_alpha=rnd_alpha, sparse=sparse,
      target_mean_bonus=target, dense_mean_bonus=dense_beta * dense_mean,
      rnd_mean_bonus=rnd_alpha * rnd_mean, rows=len(scores))


def calibrate(parent, probe, steps, output):
  cfg = read_config(probe / 'config.yaml')
  require(cfg['script'] == 'train' and cfg['seed'] == 0 and cfg['fault']['log_only']
          and cfg['run']['steps'] == steps, 'Unexpected probe configuration')
  original = json.loads((parent / 'protocol.json').read_text())
  require(cfg['run']['from_checkpoint'] == cfg['fault']['ref_ckpt'] == original['checkpoint']
          and digest(Path(original['checkpoint']) / 'agent.pkl') == original['agent_sha256'],
          'Different probe initialization/reference')
  require(digest(cfg['fault']['norm_stats']) ==
          digest(parent / 'core/seed_0/craftax_clean_fault_stats.json'), 'Changed probe p95')
  trace = probe / 'fault_score_trace.jsonl'
  # Existing checks verify alpha=0, delivered reward, KL-only scoring and clipping.
  validation = validate_probe(SimpleNamespace(trace=str(trace), fate_beta=0.2,
      delta_threshold=0.5, clip=2.0, warmup_fraction=0.2, min_rows=512, grid_multipliers='1'))
  values, raw_deltas, rnd = [], [], []
  previous = {}
  count = 0
  for row in iter_jsonl(trace):
    count += 1
    require_clean(row)
    require(row['global_step'] == count, 'Mixed or restarted probe trace')
    raw = float(row['fault_score_raw'])
    worker = row['worker']
    if not row['is_first']:
      require(worker in previous, 'Missing per-worker reset/history')
      if row['global_step'] >= validation['warmup_step']:
        values.append(raw)
        raw_deltas.append(max(0, raw - previous[worker]))
        rnd.append(row['rnd_intrinsic_reward'])
    previous[worker] = raw
  require(steps <= count < steps + 16, 'Incomplete calibration probe')
  seeds = {}
  for seed in range(5):
    path = parent / f'core/seed_{seed}/craftax_clean_fault_stats.json'
    p95 = float(json.loads(path.read_text())['fault_score_p95'])
    require(math.isfinite(p95) and p95 > 0, 'Invalid original clean scale')
    entry = matched_parameters(np.asarray(values) / p95, np.asarray(raw_deltas) / p95, rnd)
    entry.update(p95=p95, stats=str(path), stats_sha256=digest(path),
                 reference=original['checkpoint'], reference_sha256=original['agent_sha256'])
    seeds[str(seed)] = entry
  # Recomputed temporal differences must agree with the established probe scorer.
  require(math.isclose(seeds['0']['target_mean_bonus'], validation['fate']['target_mean_bonus'],
                      rel_tol=2e-5, abs_tol=2e-7), 'Temporal alignment mismatch')
  write_json(output, dict(seeds=seeds, probe=str(probe), probe_steps=steps,
      trace_sha256=digest(trace), config_sha256=digest(probe / 'config.yaml'),
      warmup_fraction=0.2, rows=count,
      rule='Common task-only Clean probe; original per-seed p95; mean matched to FATE; '
           'SparseSurprise also rate matched. No held-out performance used.',
      reused_policy=False, calibration_cost_shared=True), immutable=True)
  print(json.dumps(seeds, indent=2))


def audit_train(directory, steps, variant, expected=None):
  cfg = read_config(directory / 'config.yaml')
  f = cfg['fault']
  require(cfg['script'] == 'train' and cfg['run']['steps'] == steps and
          cfg['run']['envs'] == 16 and f['task_reward_source'] == 'delivered' and
          cfg['run']['from_checkpoint'] == f['ref_ckpt'], 'Changed training protocol')
  if expected:
    require(f['ref_ckpt'] == expected['reference'] and
            digest(Path(f['ref_ckpt']) / 'agent.pkl') == expected['reference_sha256'],
            'Changed adaptation initialization/reference')
    stats = Path(f['norm_stats'])
    require(digest(stats) == expected['stats_sha256'], 'Normalization changed')
    if variant == 'dense_matched':
      require(f['reward_mode'] == 'dense' and not f['log_only'] and
              math.isclose(f['beta'], expected['dense_beta']), 'Incorrect Dense coefficient')
    elif variant == 'sparse_surprise_matched':
      require(f['reward_mode'] == 'excess_threshold' and not f['log_only'], 'Incorrect sparse mode')
      for key, config_key in (('beta', 'beta'), ('threshold', 'reward_threshold'), ('clip', 'clip')):
        require(math.isclose(f[config_key], expected['sparse'][key]), 'Incorrect sparse coefficient')
    else:
      require(f['log_only'], 'RND must not also pay reference bonus')
  sums = dict(task=0.0, bonus=0.0)
  positive = count = non_reset = 0
  for row in iter_jsonl(directory / 'fault_score_trace.jsonl'):
    count += 1
    require_clean(row)
    require(row['global_step'] == count, 'Duplicate or restarted training trace')
    intrinsic = row['learning_reward'] - row['task_reward_delivered']
    if variant in ('rnd_matched', 'reward_scale_probe'):
      alpha = expected['rnd_alpha'] if expected else 0
      require(math.isclose(row['rnd_alpha'], alpha, rel_tol=2e-5, abs_tol=2e-7), 'Incorrect RND alpha')
      bonus = 0 if row['is_first'] else alpha * row['rnd_intrinsic_reward']
      require(math.isclose(row['rnd_reward_bonus'], bonus, rel_tol=2e-5, abs_tol=2e-6), 'Incorrect RND bonus')
    else:
      require(row['rnd_reward_bonus'] == 0, 'Unexpected RND shaping')
      if variant == 'dense_matched':
        score = np.clip(row['fault_score'], 0, f['clip'])
      else:
        score = np.clip((row['fault_score'] - f['reward_threshold']) /
                        (f['clip'] - f['reward_threshold']), 0, 1)
      bonus = 0 if row['is_first'] else f['beta'] * score
    require(math.isclose(intrinsic, bonus, rel_tol=2e-5, abs_tol=2e-6), 'Learning reward mismatch')
    positive += int(bonus > 0)
    if not row['is_first']:
      non_reset += 1
      sums['task'] += row['task_reward_delivered']
      sums['bonus'] += float(bonus)
  require(steps <= count < steps + 16, 'Incomplete training')
  require(positive > 0 or variant == 'reward_scale_probe', 'Intrinsic reward never paid')
  result = dict(rows=count, fault_events=0, fault_episodes=0, positive_steps=positive,
      bonus_positive_rate=positive / max(1, non_reset),
      mean_task_reward=sums['task'] / max(1, non_reset),
      mean_bonus=sums['bonus'] / max(1, non_reset), config_sha256=digest(directory / 'config.yaml'))
  write_json(directory / 'clean_novelty_audit.json', result)
  print(f'PASS Clean {variant}: {directory} {result}')


def audit_seed(root, seed, steps, calibration):
  expected = json.loads(calibration.read_text())['seeds'][str(seed)]
  for variant, method in VARIANTS.items():
    directory = root / f'seed_{seed}/train_{variant}/train'
    audit_train(directory, steps, variant, expected)
    ckpts = directory / 'ckpt'
    checkpoint = ckpts / (ckpts / 'latest').read_text().strip()
    require((checkpoint / 'done').is_file(), 'Incomplete checkpoint')
    for split in SPLITS:
      path = root / f'seed_{seed}/eval_{variant}_{split}'
      cfg = read_config(path / 'config.yaml')
      budget = 256 if steps < 1000000 else (30000 if split == 'clean' else 60000)
      audit_eval(path, method, split, seed, checkpoint, budget)


def protocol(root):
  source = json.loads((root / 'protocol.json').read_text())
  require(source['pilot'] is None, 'Old fault-trained pilot is not part of this study')
  write_json(root / 'study_protocol.json', dict(
      methods=VARIANTS, seeds=list(range(5)), adaptation_profile='clean', adaptation_steps=1000000,
      checkpoint=source['reference'], checkpoint_sha256=source['reference_sha256'],
      independent_clean_checkpoints=1, train_envs=16, train_ratio=128, replay_size=100000,
      task_reward_source='delivered', evaluation_steps=dict(clean=30000, seen=60000, holdout=60000, sparse=60000),
      probe_steps=200000, probe_seed=0, probe_profile='clean', probe_reward='task-only; RND alpha=0',
      discard_probe_warmup_fraction=0.2, coefficient_sweep=False, final_checkpoint_only=True,
      actual_budget_match_guaranteed=False, study_type='exploratory',
      reuse_existing_clean_taskonly_and_fate=True,
      order=['bootstrap_smoke', 'clean_probe', 'clean_calibration', 'calibrated_smoke',
             'clean_novelty', 'reevaluation', 'summary']), immutable=True)


def summary(root):
  rows, types = [], []
  for group in ('reeval', 'clean_novelty'):
    for path in sorted((root / group).glob('seed_*/*/opportunity_audit.json')):
      data = json.loads(path.read_text())
      rows.append(data['result'])
      types.extend(data['subtypes'])
  require(len(rows) == 200 and len({(r['method'], r['seed'], r['split']) for r in rows}) == 200,
          'Expected 140 existing-policy and 60 new-policy evaluations')
  for row in rows:
    row['cohort'] = 'corrected_5seed'
  means = aggregate(rows)
  require(len(means) == 40 and all(x['n'] == 5 for x in means), 'Incomplete five-seed groups')
  index = {(x['method'], x['split'], x['seed']): x for x in rows}
  effects = []
  for baseline in ('CleanAdaptTaskOnly', *VARIANTS.values()):
    for split in SPLITS:
      for metric in ('task_return', 'events_per_10k', 'coverage'):
        if split == 'clean' and metric != 'task_return':
          continue
        differences = [index['CleanAdaptFATE', split, seed][metric] - index[baseline, split, seed][metric]
                       for seed in range(5)]
        effects.append(dict(comparison=f'CleanAdaptFATE - {baseline}', split=split, metric=metric,
            mean_difference=statistics.mean(differences), sd_difference=statistics.stdev(differences),
            positive_seeds=sum(x > 1e-9 for x in differences), negative_seeds=sum(x < -1e-9 for x in differences),
            tied_seeds=sum(abs(x) <= 1e-9 for x in differences),
            **{f'seed_{i}_difference': x for i, x in enumerate(differences)}))
  output = root / 'summary'
  output.mkdir(exist_ok=True)
  for name, records in (('per_seed_results', rows), ('per_subtype_opportunities', types),
                        ('five_seed_summary', means), ('paired_clean_differences', effects)):
    with (output / f'{name}.csv').open('w') as handle:
      writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in records for k in row)))
      writer.writeheader()
      writer.writerows(records)
  explanation = (
      '# Clean novelty extension\n\n'
      'Ten methods, five matched adaptation/evaluation seeds, one shared clean checkpoint.\n'
      'Three new Clean-adapt methods; previous TaskOnly/FATE controls are reused.\n'
      'All returns are delivered task returns with no intrinsic bonus or evaluation updates.\n'
      'Bonus diagnostics are hypothetical: existing policies share a FATE monitor; '
      'new policies log their own reward formula (RND evaluation bonus is zero). '
      'Do not compare these monitor means as realized rewards or common FATE scores.\n'
      'Opportunity fields are conditional on the assigned subtype, not all counterfactual faults.\n'
      'Legacy events include delayed action releases; injections and releases are separate.\n'
      'Calibration rate/mean matching does not enforce equal realized training rewards.\n'
      'Existing fault-trained RND/Dense used a different, faulted common probe.\n'
      'Clean-vs-fault comparisons for those baselines also change coefficient calibration.\n'
      'Exploratory results, not a preregistered untouched evaluation set.\n'
      'Paired differences are descriptive; no confidence intervals or significance claims.\n')
  for split in SPLITS:
    explanation += '\n## ' + split + '\n\n' + '\n'.join(table([r for r in means if r['split'] == split])) + '\n'
  (output / 'README.md').write_text(explanation)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest='command', required=True)
  p = sub.add_parser('calibrate')
  p.add_argument('--parent', type=Path, required=True)
  p.add_argument('--probe', type=Path, required=True)
  p.add_argument('--steps', type=int, required=True)
  p.add_argument('--out', type=Path, required=True)
  p = sub.add_parser('audit-train')
  p.add_argument('--directory', type=Path, required=True)
  p.add_argument('--steps', type=int, required=True)
  p.add_argument('--variant', choices=['reward_scale_probe'], required=True)
  p = sub.add_parser('audit-seed')
  p.add_argument('--root', type=Path, required=True)
  p.add_argument('--seed', type=int, required=True)
  p.add_argument('--steps', type=int, required=True)
  p.add_argument('--calibration', type=Path, required=True)
  p = sub.add_parser('env')
  p.add_argument('calibration', type=Path)
  p.add_argument('seed')
  for command in ('protocol', 'summary'):
    p = sub.add_parser(command)
    p.add_argument('root', type=Path)
  args = parser.parse_args()
  if args.command == 'calibrate':
    calibrate(args.parent, args.probe, args.steps, args.out)
  elif args.command == 'audit-train':
    audit_train(args.directory, args.steps, args.variant)
  elif args.command == 'audit-seed':
    audit_seed(args.root, args.seed, args.steps, args.calibration)
  elif args.command == 'env':
    row = json.loads(args.calibration.read_text())['seeds'][args.seed]
    require(digest(row['stats']) == row['stats_sha256'], 'p95 source changed')
    fields = dict(DENSE_MATCHED_BETA=row['dense_beta'], RND_MATCHED_ALPHA=row['rnd_alpha'],
        SPARSE_BETA=row['sparse']['beta'], SPARSE_THRESHOLD=row['sparse']['threshold'], SPARSE_CLIP=row['sparse']['clip'])
    for key, value in fields.items():
      require(math.isfinite(value) and value > 0, f'Invalid {key}')
      print(f'export {key}={value!r}')
  elif args.command == 'protocol':
    protocol(args.root)
  else:
    summary(args.root)


if __name__ == '__main__':
  main()

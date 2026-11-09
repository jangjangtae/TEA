"""Calibration, provenance and read-only diagnostics for the corrected FATE tail."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dreamerv3.analyze_craftax_multiseed import EXPECTED_BUG_TYPES, iter_jsonl
from dreamerv3.analyze_fate_corrected_completed import job_states, summarize_trace
from dreamerv3.audit_fate_revision_run import check_trace, read_config, require
from dreamerv3.fault_score import BUG_TYPE_NAMES


SPLITS = ('clean', 'seen', 'holdout', 'sparse')
DIAGNOSTICS = ('condition_met', 'draw_eligible', 'draw_success', 'new_injection',
               'action_release', 'reward_release', 'affected_step',
               'action_changed', 'reward_changed', 'cooldown_remaining')
METHODS = (
    ('CleanEval', 'core', None), ('TaskOnly', 'core', 'taskonly'),
    ('FATE', 'core', 'excess_delta_p95_beta02'),
    ('DenseSurprise', 'matched', 'dense_matched'),
    ('RND', 'matched', 'rnd_matched'),
    ('CleanAdaptTaskOnly', 'controls', 'taskonly'),
    ('CleanAdaptFATE', 'controls', 'excess_delta_p95_beta02'))


def digest(path):
  result = hashlib.sha256()
  with Path(path).open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      result.update(block)
  return result.hexdigest()


def write_json(path, value, immutable=False):
  path = Path(path)
  if immutable and path.exists():
    require(json.loads(path.read_text()) == value, f'Changed protocol: {path}')
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
  temporary.replace(path)


def stages(root):
  result = {}
  for line in (Path(root) / 'status.tsv').read_text().splitlines():
    fields = line.split('\t')
    if len(fields) == 3:
      result[fields[1]] = fields[2]
  return result


def calibrated_sparse(normalized, delta, beta=0.2):
  normalized, delta = np.asarray(normalized), np.asarray(delta)
  require(normalized.size >= 100 and normalized.shape == delta.shape,
          'Too little clean calibration data')
  require(np.isfinite(normalized).all() and np.isfinite(delta).all(), 'Nonfinite scores')
  fate = np.clip((delta - 0.5) / 1.5, 0, 1)
  target_rate = float(np.mean(fate > 0))
  require(0 < target_rate < 1 and float(fate.mean()) > 1e-8, 'Degenerate FATE calibration')
  threshold = float(np.quantile(normalized, 1 - target_rate))
  require(threshold > 0, 'Nonpositive absolute-surprise threshold')
  # Same continuous ramp width as FATE, shifted to a calibrated absolute score.
  clip = threshold + 1.5
  sparse = np.clip((normalized - threshold) / (clip - threshold), 0, 1)
  actual_rate = float(np.mean(sparse > 0))
  require(abs(actual_rate - target_rate) <= max(2 / len(sparse), 0.002),
          'Score ties prevent a close activation-rate match')
  require(float(sparse.mean()) > 1e-8, 'Near-zero SparseSurprise calibration mean')
  coefficient = float(beta * fate.mean() / sparse.mean())
  require(math.isfinite(coefficient) and 0 < coefficient <= 10,
          'Unsafe coefficient; inspect calibration instead of silently clipping')
  return dict(threshold=threshold, clip=clip, beta=coefficient, rows=len(sparse),
      fate_beta=beta, fate_positive_rate=target_rate, sparse_positive_rate=actual_rate,
      fate_mean_bonus=float(beta * fate.mean()),
      sparse_mean_bonus=float(coefficient * sparse.mean()), ramp_width=1.5)


def calibrate(parent, output):
  seeds = {}
  for seed in (0, 1):
    root = parent / f'core/seed_{seed}'
    trace = root / 'base_clean_eval/fault_score_trace.jsonl'
    stats_path = root / 'craftax_clean_fault_stats.json'
    stats = json.loads(stats_path.read_text())
    require(stats['fault_score_p95'] > 0, 'Invalid p95')
    scores, deltas = [], []
    previous = {}
    count = 0
    for row in iter_jsonl(trace):
      count += 1
      require(row['worker'] == 0 and row['global_step'] == count, 'Invalid clean rollout')
      require(not row['bug_triggered'] and not row.get('log/fault_episode', 0),
              'Fault contamination in clean calibration')
      value = row['fault_score_raw'] / stats['fault_score_p95']
      prev = 0 if row['is_first'] else previous.get(row['worker'], 0)
      previous[row['worker']] = value
      if not row['is_first']:
        scores.append(value)
        deltas.append(max(0, value - prev))
    require(count == 30000, 'Incomplete clean calibration')
    entry = calibrated_sparse(scores, deltas)
    entry.update(p95=stats['fault_score_p95'], trace=str(trace),
                 trace_sha256=digest(trace), stats_sha256=digest(stats_path))
    seeds[str(seed)] = entry
  record = dict(rule='clean_rate_and_mean_match_to_fate', seeds=seeds,
      note='No fault labels or evaluation outcomes used. Pilot only; no coefficient selection.')
  write_json(output, record, immutable=True)
  for seed, row in seeds.items():
    print(f"seed={seed}: threshold={row['threshold']:.6g}, beta={row['beta']:.6g}, "
          f"rate={row['sparse_positive_rate']:.6g}, mean={row['sparse_mean_bonus']:.6g}")


def make_manifest(parent, controls, output, pilot=True):
  require(stages(parent).get('queue') == 'DONE', 'Main queue incomplete')
  require(stages(controls).get('queue') == 'DONE', 'Clean controls incomplete')
  control_audit = json.loads((controls / 'controls/clean_control_audit.json').read_text())
  require(len(control_audit) == 10 and all(
      x['rows'] >= 1000000 and x['fault_events'] == x['fault_episodes'] == 0
      for x in control_audit.values()), 'Missing clean-control integrity audit')
  p = json.loads((parent / 'protocol.json').read_text())
  c = json.loads((controls / 'protocol.json').read_text())
  reference = Path(p['checkpoint'])
  require(c['checkpoint'] == str(reference) and c['agent_sha256'] == p['agent_sha256'],
          'Different clean initialization')
  require(digest(reference / 'agent.pkl') == p['agent_sha256'], 'Reference changed')
  roots = dict(core=parent / 'core', matched=parent / 'matched', controls=controls / 'controls')
  states = {key: job_states(root) for key, root in roots.items()}
  cases = []
  for seed in range(5):
    stats = parent / f'core/seed_{seed}/craftax_clean_fault_stats.json'
    require(digest(stats) == c['calibration_sha256'][str(seed)], 'Calibration changed')
    for method, group, variant in METHODS:
      root = roots[group]
      for split in SPLITS:
        name = f'eval_{variant}_{split}' if variant else f'base_{split}_eval'
        require(states[group].get((seed, name)) == 'DONE', f'Incomplete source {group}/{seed}/{name}')
        require((root / f'job_audits/seed_{seed}_{name}.json').is_file(), 'Missing source audit')
        cfg = read_config(root / f'seed_{seed}/{name}/config.yaml')
        budget = 30000 if split == 'clean' else 60000
        require(cfg['script'] == 'eval_only' and cfg['seed'] == seed and
                cfg['run']['steps'] == budget and cfg['fault']['log_only'], 'Source protocol mismatch')
        checkpoint = Path(cfg['run']['from_checkpoint'])
        require((checkpoint / 'done').is_file(), 'Checkpoint incomplete')
        if variant:
          require(states[group].get((seed, f'train_{variant}')) == 'DONE', 'Training incomplete')
          trained = read_config(root / f'seed_{seed}/train_{variant}/train/config.yaml')
          require(trained['run']['steps'] == 1000000 and
                  trained['fault']['task_reward_source'] == 'delivered', 'Unexpected source training')
        cases.append(dict(method=method, seed=seed, split=split, steps=budget,
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint / 'agent.pkl'),
            stats=str(stats), stats_sha256=digest(stats),
            profile='clean' if split == 'clean' else f'benchmark_{split}',
            source_config=str(root / f'seed_{seed}/{name}/config.yaml'),
            source_config_sha256=digest(root / f'seed_{seed}/{name}/config.yaml')))
  write_json(output, dict(cases=cases, reference=str(reference),
      reference_sha256=p['agent_sha256'], fault_episode_probability=0.5,
      manifest_multiplier=1.0, independent_clean_checkpoints=1,
      evaluation_monitor='same hypothetical FATE bonus for all fixed policies',
      pilot=(dict(method='SparseSurprise', seeds=[0, 1], train_steps=1000000,
                  coefficient_selection=False) if pilot else None)), immutable=True)


def check_diagnostic_rows(rows, split):
  counts = {name: Counter() for name in EXPECTED_BUG_TYPES.get(split, ())}
  total = Counter()
  assigned = {}
  reward_sums = defaultdict(float)
  for row in rows:
    require('log/fault_assigned_type_id' in row, 'Missing opportunity instrumentation')
    values = {key: int(row[f'log/fault_{key}']) for key in DIAGNOSTICS}
    for key, value in values.items():
      require(value >= 0 and (key == 'cooldown_remaining' or value <= 1), f'Invalid {key}')
    probability = float(row['log/fault_draw_probability'])
    require(math.isfinite(probability) and 0 <= probability <= 1, 'Invalid draw probability')
    kind = BUG_TYPE_NAMES.get(int(row['log/fault_assigned_type_id']), 'unknown')
    require(kind == 'none' or kind in counts, f'Unexpected assigned subtype {split}/{kind}')
    require(row.get('log/rnd_update_enabled', 0) == 0, 'RND updated in fixed evaluation')
    require(values['draw_eligible'] <= values['condition_met'], 'Draw without condition')
    require(values['draw_success'] <= values['draw_eligible'], 'Success without draw')
    require(values['new_injection'] <= values['draw_success'], 'Injection without accepted draw')
    require(not (values['draw_eligible'] and values['cooldown_remaining']), 'Draw during cooldown')
    require(not (values['action_release'] and values['new_injection']), 'Action release reinjected')
    require(values['affected_step'] == int(bool(row['bug_triggered'] or values['reward_release'])),
            'Affected-step/legacy-event mismatch')
    worker = row['worker']
    if row['is_first']:
      require(not any(values.values()), 'Diagnostics carried across reset')
      assigned[worker] = kind
      if kind != 'none':
        counts[kind]['assigned_episodes'] += 1
      continue
    require(assigned.get(worker) == kind, 'Assigned subtype changed inside episode')
    total['non_reset_steps'] += 1
    for key in ('fault_reward_bonus', 'task_reward_delivered'):
      reward_sums[key] += row[key]
    total['bonus_positive_steps'] += row['fault_reward_bonus'] > 0
    if kind == 'none':
      require(not any(values.values()) and not row['bug_triggered'], 'Fault in inactive episode')
      continue
    item = counts[kind]
    item['assigned_steps'] += 1
    item.update(values)
    item['legacy_events'] += row['bug_triggered']
    if row['bug_triggered'] and 'first_event_step' not in item:
      item['first_event_step'] = row['global_step']
  total.update({f'mean_{key}': value / max(1, total['non_reset_steps'])
                for key, value in reward_sums.items()})
  return counts, dict(total)


def audit_eval(directory, method, split, seed, checkpoint, steps):
  cfg = read_config(directory / 'config.yaml')
  require(cfg['script'] == 'eval_only' and cfg['fault']['log_only'] and
          cfg['run']['envs'] == 1 and cfg['seed'] == seed and cfg['run']['steps'] == steps,
          'Wrong evaluation protocol')
  require(Path(cfg['run']['from_checkpoint']) == checkpoint, 'Wrong checkpoint')
  stats = json.loads(Path(cfg['fault']['norm_stats']).read_text())
  audit = check_trace(directory, cfg, stats)
  trace = directory / 'fault_score_trace.jsonl'
  counts, diagnostics = check_diagnostic_rows(iter_jsonl(trace), split)
  scores = list(iter_jsonl(directory / 'scores.jsonl'))
  if not scores and steps < 30000:
    # A smoke rollout may end before the first complete episode.
    result = dict(steps=audit['rows'], episodes=0, task_return=None)
  else:
    # The eval driver advances in blocks of ten; short smoke budgets can round up.
    budget = audit['rows'] if steps < 30000 else steps
    result, _ = summarize_trace(iter_jsonl(trace), scores, split, budget)
  result['requested_steps'] = steps
  result.update(method=method, split=split, seed=seed, **diagnostics)
  types = []
  for subtype, values in sorted(counts.items()):
    types.append(dict(method=method, split=split, seed=seed, subtype=subtype,
        **{key: values[key] for key in ('assigned_episodes', 'assigned_steps', *DIAGNOSTICS, 'legacy_events')},
        first_event_step=values.get('first_event_step')))
  write_json(directory / 'opportunity_audit.json', dict(audit=audit, result=result, subtypes=types,
      checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint / 'agent.pkl'),
      config_sha256=digest(directory / 'config.yaml'), trace_sha256=digest(trace)))


def audit_pilot(root, seed, steps):
  variant = 'sparse_surprise_matched'
  directory = root / f'seed_{seed}/train_{variant}/train'
  cfg = read_config(directory / 'config.yaml')
  require(cfg['run']['steps'] == steps and cfg['fault']['reward_mode'] == 'excess_threshold'
          and not cfg['fault']['log_only'], 'Wrong SparseSurprise pilot')
  positive = count = 0
  f = cfg['fault']
  require(f['clip'] > f['reward_threshold'], 'Invalid SparseSurprise ramp')
  for row in iter_jsonl(directory / 'fault_score_trace.jsonl'):
    expected = 0 if row['is_first'] else f['beta'] * np.clip(
        (row['fault_score'] - f['reward_threshold']) / (f['clip'] - f['reward_threshold']), 0, 1)
    require(math.isclose(row['fault_reward_bonus'], expected, rel_tol=2e-5, abs_tol=2e-6),
            'SparseSurprise bonus differs from calibrated formula')
    positive += expected > 0
    count += 1
  require(positive > 0 and steps <= count < steps + 16, 'Inactive or incomplete pilot')
  checkpoint_root = directory / 'ckpt'
  checkpoint = checkpoint_root / (checkpoint_root / 'latest').read_text().strip()
  for split in SPLITS:
    evaluation = root / f'seed_{seed}/eval_{variant}_{split}'
    eval_cfg = read_config(evaluation / 'config.yaml')
    for key in ('beta', 'reward_threshold', 'clip'):
      require(eval_cfg['fault'][key] == f[key], f'Lost pilot eval parameter {key}')
    audit_eval(evaluation, 'SparseSurprise', split, seed, checkpoint, eval_cfg['run']['steps'])
  write_json(directory / 'sparse_reward_audit.json', dict(rows=count, positive_steps=positive,
      beta=f['beta'], threshold=f['reward_threshold'], clip=f['clip']))


def summarize(root):
  rows, types = [], []
  for path in sorted((root / 'reeval').glob('seed_*/*/opportunity_audit.json')) + sorted(
      (root / 'pilot').glob('seed_*/eval_*/opportunity_audit.json')):
    data = json.loads(path.read_text())
    rows.append(data['result'])
    types.extend(data['subtypes'])
  require(len(rows) == 148, f'Expected 140 fixed-policy and 8 pilot evaluations, got {len(rows)}')
  output = root / 'summary'
  output.mkdir(exist_ok=True)
  for name, records in (('per_seed_results', rows), ('per_subtype_opportunities', types)):
    with (output / f'{name}.csv').open('w') as handle:
      writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in records for k in row)))
      writer.writeheader()
      writer.writerows(records)
  (output / 'README.md').write_text(
      '# FATE diagnostic follow-up\n\n'
      'Seven existing policies: seeds 0-4. SparseSurprise pilot: seeds 0-1 only.\n'
      'Compare the pilot with matched seeds, not the five-seed consistency count.\n'
      'Shared clean initialization; evaluations are not new training replicates.\n'
      'Reevaluation policies use a common hypothetical FATE monitor; pilot logs use SparseSurprise.\n'
      'Legacy events include action releases; new_injection excludes releases.\n'
      'condition_met is the selected operator predicate; draw_eligible additionally excludes cooldown.\n'
      'No counterfactual eligibility for operators not assigned to the episode is claimed.\n'
      'affected_step includes reward releases, not the unbounded downstream influence of a fault.\n'
      'Missing subtype hits stay in the coverage denominator.\n')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest='command', required=True)
  p = sub.add_parser('calibrate')
  p.add_argument('--parent', type=Path, required=True)
  p.add_argument('--out', type=Path, required=True)
  p = sub.add_parser('manifest')
  p.add_argument('--parent', type=Path, required=True)
  p.add_argument('--controls', type=Path, required=True)
  p.add_argument('--out', type=Path, required=True)
  p.add_argument('--no-pilot', action='store_true')
  p = sub.add_parser('case-lines')
  p.add_argument('manifest', type=Path)
  p = sub.add_parser('sparse-env')
  p.add_argument('calibration', type=Path)
  p.add_argument('seed')
  p = sub.add_parser('audit-eval')
  p.add_argument('--directory', type=Path, required=True)
  p.add_argument('--method', required=True)
  p.add_argument('--split', choices=SPLITS, required=True)
  p.add_argument('--seed', type=int, required=True)
  p.add_argument('--checkpoint', type=Path, required=True)
  p.add_argument('--steps', type=int, required=True)
  p = sub.add_parser('audit-pilot')
  p.add_argument('--root', type=Path, required=True)
  p.add_argument('--seed', type=int, required=True)
  p.add_argument('--steps', type=int, required=True)
  p = sub.add_parser('summary')
  p.add_argument('root', type=Path)
  args = parser.parse_args()
  if args.command == 'calibrate':
    calibrate(args.parent, args.out)
  elif args.command == 'manifest':
    make_manifest(args.parent, args.controls, args.out, pilot=not args.no_pilot)
  elif args.command == 'case-lines':
    data = json.loads(args.manifest.read_text())
    require(digest(Path(data['reference']) / 'agent.pkl') == data['reference_sha256'], 'Reference changed')
    writer = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    for row in data['cases']:
      require(digest(Path(row['checkpoint']) / 'agent.pkl') == row['checkpoint_sha256'], 'Policy changed')
      require(digest(row['stats']) == row['stats_sha256'], 'p95 changed')
      writer.writerow([row[k] for k in ('seed', 'method', 'split', 'checkpoint', 'stats', 'profile', 'steps')])
  elif args.command == 'sparse-env':
    row = json.loads(args.calibration.read_text())['seeds'][args.seed]
    for key in ('beta', 'threshold', 'clip'):
      value = float(row[key])
      require(math.isfinite(value) and value > 0, f'Invalid sparse parameter {key}')
      print(f'export SPARSE_{key.upper()}={value!r}')
  elif args.command == 'audit-eval':
    audit_eval(args.directory, args.method, args.split, args.seed, args.checkpoint, args.steps)
  elif args.command == 'audit-pilot':
    audit_pilot(args.root, args.seed, args.steps)
  else:
    summarize(args.root)


if __name__ == '__main__':
  main()

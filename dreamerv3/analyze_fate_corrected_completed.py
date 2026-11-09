"""Summarize only completed, audited corrected FATE evaluations on CPU."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from dreamerv3.analyze_craftax_multiseed import EXPECTED_BUG_TYPES, iter_jsonl, write_csv
from dreamerv3.audit_fate_revision_run import read_config, require


SPLITS = ('clean', 'seen', 'holdout', 'sparse')
MAIN_METHODS = ('CleanEval', 'TaskOnly', 'FATE', 'DenseSurprise-matched', 'RND-matched')
METRICS = ('task_return', 'events_per_10k', 'coverage', 'unique_types')


def job_states(root):
  result = {}
  for line in (root / 'status.tsv').read_text().splitlines():
    fields = line.split('\t')
    if len(fields) == 4:
      result[(int(fields[1].split('=')[1]), fields[2])] = fields[3]
  return result


def summarize_trace(rows, scores, split, budget):
  types = Counter()
  count = resets = 0
  last_step = 0
  first_event = None
  for row in rows:
    count += 1
    require(row['global_step'] > last_step, 'Duplicate or restarted evaluation step')
    last_step = row['global_step']
    require(row['worker'] == 0, 'Expected single-env evaluation')
    resets += bool(row['is_first'])
    require(math.isfinite(row['learning_reward']), 'Nonfinite evaluation reward')
    require(row['rnd_reward_bonus'] == 0, 'RND active during evaluation')
    require(math.isclose(row['learning_reward'], row['task_reward_delivered'],
                        rel_tol=2e-5, abs_tol=2e-6), 'Evaluation reward was shaped')
    if row['bug_triggered']:
      require(not row['is_first'], 'Fault on reset')
      kind = row['bug_type']
      require(kind in EXPECTED_BUG_TYPES.get(split, set()), f'Unexpected fault: {split}/{kind}')
      types[kind] += 1
      if first_event is None:
        first_event = count
  require(count == budget, f'Wrong evaluation length: {count}/{budget}')
  values = [row['episode/score'] for row in scores]
  require(values and all(math.isfinite(x) for x in values), 'Missing or nonfinite episode returns')
  expected = EXPECTED_BUG_TYPES.get(split, set())
  events = sum(types.values())
  return dict(steps=count, non_reset_steps=count - resets, episodes=len(values),
              task_return=statistics.mean(values), event_count=events,
              events_per_10k=events * 10000 / count, unique_types=len(types),
              coverage=len(types) / len(expected) if expected else None,
              time_to_first_fault=first_event,
              discovered_types=';'.join(sorted(types))), types


def read_case(root, seed, name, method, split, cohort):
  audit_path = root / f'job_audits/seed_{seed}_{name}.json'
  audit = json.loads(audit_path.read_text())
  path = root / f'seed_{seed}' / name
  cfg = read_config(path / 'config.yaml')
  budget = 30000 if split == 'clean' else 60000
  require(cfg['script'] == 'eval_only' and cfg['run']['envs'] == 1, 'Not frozen evaluation')
  require(cfg['run']['steps'] == budget and cfg['seed'] == seed, 'Wrong evaluation budget/seed')
  require(cfg['fault']['task_reward_source'] == 'delivered' and cfg['fault']['log_only'],
          'Wrong evaluation reward channel')
  row, counts = summarize_trace(iter_jsonl(path / 'fault_score_trace.jsonl'),
      iter_jsonl(path / 'scores.jsonl'), split, budget)
  require(row['steps'] == audit['rows'] and row['non_reset_steps'] == audit['non_reset_transitions'],
          'Evaluation disagrees with completed-job audit')
  row.update(cohort=cohort, method=method, seed=seed, split=split, path=str(path),
             audit_path=str(audit_path), config_sha256=hashlib.sha256((path / 'config.yaml').read_bytes()).hexdigest())
  type_rows = [dict(cohort=cohort, method=method, split=split, seed=seed,
                    subtype=kind, events=counts[kind], hit=int(counts[kind] > 0))
               for kind in sorted(EXPECTED_BUG_TYPES.get(split, set()))]
  return row, type_rows


def aggregate(rows):
  groups = defaultdict(list)
  for row in rows:
    groups[row['cohort'], row['method'], row['split']].append(row)
  result = []
  for (cohort, method, split), items in groups.items():
    seeds = sorted(item['seed'] for item in items)
    require(len(seeds) == len(set(seeds)), 'Duplicate seed in group')
    sets = [set(x['discovered_types'].split(';')) - {''} for x in items]
    consistent = set.intersection(*sets)
    row = dict(cohort=cohort, method=method, split=split, n=len(items),
               seeds=';'.join(map(str, seeds)), consistent_types=len(consistent) if split != 'clean' else None,
               consistent_type_names=';'.join(sorted(consistent)),
               union_types=len(set.union(*sets)) if split != 'clean' else None,
               event_count_total=sum(x['event_count'] for x in items))
    for metric in METRICS:
      values = [x[metric] for x in items if x[metric] is not None]
      row[f'{metric}_mean'] = statistics.mean(values) if values else None
      row[f'{metric}_sd'] = statistics.stdev(values) if len(values) > 1 else None
    result.append(row)
  return result


def paired_effects(rows):
  index = {(x['method'], x['split'], x['seed']): x for x in rows}
  result = []
  for baseline in MAIN_METHODS:
    if baseline == 'FATE':
      continue
    for split in SPLITS:
      for metric in METRICS:
        if split == 'clean' and metric != 'task_return':
          continue
        differences = [index['FATE', split, seed][metric] - index[baseline, split, seed][metric]
                       for seed in range(5)]
        result.append(dict(comparison=f'FATE - {baseline}', split=split, metric=metric,
            n=5, mean_difference=statistics.mean(differences), sd_difference=statistics.stdev(differences),
            positive_seeds=sum(x > 1e-9 for x in differences), negative_seeds=sum(x < -1e-9 for x in differences),
            tied_seeds=sum(abs(x) <= 1e-9 for x in differences),
            **{f'seed_{i}_difference': x for i, x in enumerate(differences)}))
  return result


def save_csv(path, rows):
  if rows:
    write_csv(path, rows, list(rows[0]))


def table(rows):
  lines = ['| Method | n | Return (mean +/- SD) | Events/10K | Coverage | Consistent types |',
           '|---|---:|---:|---:|---:|---:|']
  for row in rows:
    def fmt(metric):
      mean, sd = row[f'{metric}_mean'], row[f'{metric}_sd']
      if mean is None:
        return '-'
      return f'{mean:.3f}' + (f' +/- {sd:.3f}' if sd is not None else '')
    consistent = row['consistent_types']
    lines.append(f"| {row['method']} | {row['n']} | {fmt('task_return')} | {fmt('events_per_10k')} | {fmt('coverage')} | {consistent if consistent is not None else '-'} |")
  return lines


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--parent', type=Path, required=True)
  parser.add_argument('--controls', type=Path, required=True)
  parser.add_argument('--out', type=Path, required=True)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)
  roots = {name: args.parent / name for name in ('core', 'matched', 'weight_check')}
  roots['controls'] = args.controls / 'controls'
  # Snapshot eligibility before reading traces; never mix in live/incomplete seeds.
  states = {name: job_states(root) for name, root in roots.items()}
  variants = ('taskonly', 'excess_delta_p95_beta02')
  controls_seeds = [seed for seed in range(5) if all(
      states['controls'].get((seed, f'eval_{variant}_{split}')) == 'DONE'
      for variant in variants for split in SPLITS)]
  controls_complete = controls_seeds == list(range(5))
  controls_cohort = 'controls_complete' if controls_complete else 'controls_partial'
  cases = []
  for seed in range(5):
    for split in SPLITS:
      cases.append(('core', seed, f'base_{split}_eval', 'CleanEval', split, 'main'))
      for stage, variant, method in (
          ('core', 'taskonly', 'TaskOnly'), ('core', 'excess_delta_p95_beta02', 'FATE'),
          ('matched', 'dense_matched', 'DenseSurprise-matched'), ('matched', 'rnd_matched', 'RND-matched')):
        cases.append((stage, seed, f'eval_{variant}_{split}', method, split, 'main'))
  for variant, method in (
      ('dense_matched_half', 'Dense 0.5x'), ('dense_matched_double', 'Dense 2x'),
      ('rnd_matched_half', 'RND 0.5x'), ('rnd_matched_double', 'RND 2x')):
    cases.append(('weight_check', 0, f'eval_{variant}_seen', method, 'seen', 'weight_check'))
  for seed in controls_seeds:
    for variant, method in zip(variants, ('Clean-adapt TaskOnly', 'Clean-adapt FATE')):
      for split in SPLITS:
        cases.append(('controls', seed, f'eval_{variant}_{split}', method, split, controls_cohort))
  rows, types = [], []
  for i, (stage, seed, name, method, split, cohort) in enumerate(cases, 1):
    require(states[stage].get((seed, name)) == 'DONE', f'Incomplete evaluation: {stage}/{seed}/{name}')
    if not name.startswith('base_'):
      variant = name[len('eval_'):].rsplit('_', 1)[0]
      require(states[stage].get((seed, f'train_{variant}')) == 'DONE', 'Training incomplete')
    row, subtypes = read_case(roots[stage], seed, name, method, split, cohort)
    rows.append(row)
    types.extend(subtypes)
    print(f'[{i}/{len(cases)}] {method} seed={seed} {split}: '
          f'return={row["task_return"]:.3f} events={row["event_count"]} types={row["unique_types"]}', flush=True)
  main_rows = [x for x in rows if x['cohort'] == 'main']
  summary = aggregate(main_rows)
  weights = [x for x in rows if x['cohort'] == 'weight_check']
  weights += [dict(x, cohort='weight_check', method={'DenseSurprise-matched': 'Dense 1x',
      'RND-matched': 'RND 1x'}[x['method']]) for x in main_rows
      if x['seed'] == 0 and x['split'] == 'seen' and x['method'] in ('DenseSurprise-matched', 'RND-matched')]
  matched_controls = [x for x in rows if x['cohort'] == controls_cohort]
  matched_controls += [dict(x, cohort=controls_cohort) for x in main_rows
      if x['seed'] in controls_seeds and x['method'] in ('CleanEval', 'TaskOnly', 'FATE')]
  controls_summary = aggregate(matched_controls)
  hits = []
  for entry in summary:
    for kind in sorted(EXPECTED_BUG_TYPES.get(entry['split'], set())):
      selected = [x for x in types if x['cohort'] == 'main' and x['method'] == entry['method']
                  and x['split'] == entry['split'] and x['subtype'] == kind]
      require(len(selected) == 5, 'Incomplete subtype hit denominator')
      hits.append(dict(method=entry['method'], split=entry['split'], subtype=kind,
          seed_hits=sum(x['hit'] for x in selected), n=5,
          total_events=sum(x['events'] for x in selected),
          **{f'seed_{x["seed"]}_events': x['events'] for x in selected}))
  for name, content in (
      ('all_completed_per_seed', rows), ('main_per_seed', main_rows), ('main_summary', summary),
      ('subtype_per_seed', types), ('main_subtype_seed_hits', hits),
      ('paired_main_differences', paired_effects(main_rows)), ('weight_check_seed0', weights),
      ('clean_controls_matched_summary', controls_summary)):
    save_csv(args.out / f'{name}.csv', content)
  notes = ['# Corrected FATE completed results', '',
      'Only completed, audited no-update evaluations. Main comparisons: five adaptation/evaluation seeds,',
      'one shared corrected clean checkpoint. No legacy AAAI results are pooled.',
      'Return is mean completed-episode delivered return within each seed, then equal-weight mean across seeds.',
      'Events/10K uses all driver callbacks (including reset), matching the existing paper analysis.',
      'Coverage denominator: Seen 8; Holdout/Sparse 7. Clean coverage is not applicable.',
      'Consistent types is a set intersection across the listed seeds, NOT a mean with SD/SEM.',
      'All +/- values below are sample SD across seeds, not SEM. No significance claim is made.', '']
  for split in SPLITS:
    notes += [f'## Main: {split}', ''] + table([x for x in summary if x['split'] == split]) + ['']
  notes += ['## Weight check', '', 'Seen, seed 0 only. The 1x conditions reuse matched main seed 0.', '']
  notes += table(aggregate(weights)) + ['']
  controls_title = 'Five-seed' if controls_complete else 'Partial'
  controls_note = ('All five matched seeds are complete.' if controls_complete else
                   'Do not compare their consistency count directly to n=5.')
  notes += [f'## {controls_title} clean-adaptation controls', '',
            f'Matched seeds: {controls_seeds}. {controls_note}', '']
  for split in SPLITS:
    notes += [f'### {split}', ''] + table([x for x in controls_summary if x['split'] == split]) + ['']
  (args.out / 'README.md').write_text('\n'.join(notes) + '\n')
  manifest = dict(created=datetime.now().astimezone().isoformat(), parent=str(args.parent),
      controls=str(args.controls), main_seeds=list(range(5)), completed_control_seeds=controls_seeds,
      evaluation_cases=len(cases), analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      parent_protocol=json.loads((args.parent / 'protocol.json').read_text()),
      reward_scale_calibration=json.loads((args.parent / 'reward_scale_calibration.json').read_text()))
  (args.out / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
  print(f'SAVED: {args.out}', flush=True)


if __name__ == '__main__':
  main()

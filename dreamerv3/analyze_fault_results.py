#!/usr/bin/env python3
"""Summarize Dreamer fault-score experiments.

This script collects tester_eval summary.json files and optionally reads
per-step JSONL traces for ranking/window analyses. It is intentionally
file-based so old experiment folders can be compared without rerunning models.
"""

import argparse
import csv
import datetime as _dt
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np


DEFAULT_ROOTS = [
    '/home/railab/logdir/fault_3day_leave_20260528_170600',
    '/home/railab/logdir/fault_nextday_20260602_182008',
    '/home/railab/logdir/fault_followup3_20260604_175648',
    '/home/railab/logdir/fault_semantic_holdout_20260608_134437_systemd',
]

SUMMARY_FIELDS = [
    'step_auroc',
    'step_auprc',
    'step_precision',
    'step_recall',
    'step_f1',
    'episode_detection_rate',
    'mean_time_to_detect',
    'step_fault_applied_rate',
    'step_alarm_rate',
    'clean_false_alarm_episode_rate',
    'episode_score_mean',
    'episode_task_score_mean',
    'episode_training_reward_mean',
    'episode_tester_bonus_mean',
    'episode_rnd_intrinsic_reward_mean',
    'episode_length_mean',
    'clean_score_mean',
    'fault_score_mean',
    'clean_task_score_mean',
    'fault_task_score_mean',
    'clean_training_reward_mean',
    'fault_training_reward_mean',
    'max_fault_score_mean',
    'max_latent_kl_surprise_mean',
    'episode_unique_states_mean',
    'episode_revisit_ratio_mean',
    'recent_novel_rate_mean',
    'unique_action_bigrams_mean',
    'semantic_context_episode_rate',
    'semantic_fault_episode_rate',
    'semantic_fault_given_context_episode_rate',
    'semantic_context_step_rate',
    'semantic_context_step_count',
    'semantic_fault_step_rate',
    'semantic_fault_step_count',
    'semantic_fault_given_context_step_rate',
    'semantic_context_step_mean',
    'semantic_ctx_upgrade_collect_count_episode_rate',
    'semantic_ctx_retry_craft_count_episode_rate',
    'semantic_ctx_relocate_station_count_episode_rate',
    'semantic_ctx_valid_progress_count_episode_rate',
    'semantic_ctx_station_reuse_count_episode_rate',
    'semantic_ctx_delayed_after_use_count_episode_rate',
    'semantic_ctx_upgrade_collect_step_rate',
    'semantic_ctx_upgrade_collect_step_count',
    'semantic_ctx_retry_craft_step_rate',
    'semantic_ctx_retry_craft_step_count',
    'semantic_ctx_relocate_station_step_rate',
    'semantic_ctx_relocate_station_step_count',
    'semantic_ctx_valid_progress_step_rate',
    'semantic_ctx_valid_progress_step_count',
    'semantic_ctx_station_reuse_step_rate',
    'semantic_ctx_station_reuse_step_count',
    'semantic_ctx_delayed_after_use_step_rate',
    'semantic_ctx_delayed_after_use_step_count',
    'trace_rows',
    'trace_fault_episode_rate',
    'trace_fault_applied_rate',
]

CORE_SPLITS = ['clean', 'seen', 'holdout', 'semantic_holdout']
CORE_METRICS = [
    'step_auroc',
    'step_auprc',
    'episode_detection_rate',
    'step_fault_applied_rate',
    'episode_task_score_mean',
    'clean_false_alarm_episode_rate',
]

PARETO_OBJECTIVES = {
    'task_vs_discovery': ['episode_task_score_mean', 'step_fault_applied_rate'],
    'task_vs_signal': ['episode_task_score_mean', 'step_auroc'],
    'task_vs_ranking': ['episode_task_score_mean', 'step_auprc'],
    'task_discovery_signal': [
        'episode_task_score_mean', 'step_fault_applied_rate', 'step_auroc'],
}

BASELINE_RUNS = ('task_only_repeat', 'task_only', 'reference')
COMPETENCE_THRESHOLDS = (0.9, 0.8)

TRACE_LABELS = [
    ('fault_manifested', 'fault_manifested'),
    ('fault_trigger_context', 'fault_trigger_context'),
    ('semantic_trigger_context', 'semantic_trigger_context'),
    ('legacy_fault_applied', 'fault_applied'),
]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--roots', nargs='*', default=DEFAULT_ROOTS,
      help='Experiment roots to search recursively for tester_eval/summary.json.')
  parser.add_argument(
      '--summary', nargs='*', default=[],
      help='Specific summary.json files to include in addition to --roots.')
  parser.add_argument(
      '--outdir', default='',
      help='Output directory. Defaults to /home/railab/logdir/fault_analysis_<time>.')
  parser.add_argument(
      '--trace-analysis', action='store_true',
      help='Read <split>_steps.jsonl files and compute top-k/window analyses.')
  parser.add_argument(
      '--splits', default='clean,seen,holdout,semantic_holdout',
      help='Comma separated split names to include.')
  parser.add_argument(
      '--top-fracs', default='0.001,0.005,0.01,0.05',
      help='Comma separated fractions for precision@top-k trace analysis.')
  parser.add_argument(
      '--window', type=int, default=20,
      help='Half-window around first fault step for event-window analysis.')
  return parser.parse_args()


def safe_float(value):
  if value is None:
    return ''
  try:
    value = float(value)
  except Exception:
    return value
  if math.isnan(value) or math.isinf(value):
    return ''
  return value


def fmt(value, digits=4):
  value = safe_float(value)
  if value == '':
    return 'nan'
  if isinstance(value, float):
    return f'{value:.{digits}f}'
  return str(value)


def write_csv(path, rows, fieldnames):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def write_json(path, obj):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as f:
    json.dump(obj, f, indent=2, ensure_ascii=False)


def discover_summaries(roots, explicit):
  paths = []
  for item in explicit:
    path = Path(item).expanduser()
    if path.exists():
      paths.append(path)
  for root in roots:
    root = Path(root).expanduser()
    if not root.exists():
      continue
    paths.extend(sorted(root.glob('**/tester_eval/summary.json')))
  seen = set()
  unique = []
  for path in paths:
    resolved = path.resolve()
    if resolved in seen:
      continue
    seen.add(resolved)
    unique.append(path)
  return unique


def infer_experiment(path):
  # .../<eval_name>/tester_eval/summary.json
  eval_dir = path.parent.parent
  root = eval_dir.parent
  raw = eval_dir.name

  aliases = [
      ('reference_semantic_eval', 'reference'),
      ('reference', 'reference'),
      ('eval_fault_adapt_beta005', 'fault_beta0.05'),
      ('beta005_semantic_eval', 'fault_beta0.05'),
      ('eval_fault_adapt_beta01_repeat', 'fault_beta0.1_repeat'),
      ('beta01_repeat_semantic_eval', 'fault_beta0.1_repeat'),
      ('eval_fault_adapt_beta01', 'fault_beta0.1'),
      ('eval_fault_adapt_beta02', 'fault_beta0.2'),
      ('eval_fault_adapt_beta05', 'fault_beta0.5'),
      ('eval_task_only_fault_logging_repeat', 'task_only_repeat'),
      ('task_only_repeat_semantic_eval', 'task_only_repeat'),
      ('eval_task_only_fault_logging', 'task_only'),
  ]
  label = raw
  for key, value in aliases:
    if raw == key:
      label = value
      break

  if 'semantic_holdout' in root.name or raw.endswith('_semantic_eval'):
    suite = 'semantic_holdout_eval'
  elif 'followup3' in root.name:
    suite = 'followup_repeat'
  elif 'nextday' in root.name:
    suite = 'task_beta05'
  elif '3day' in root.name:
    suite = 'initial_fault'
  else:
    suite = root.name

  return {
      'suite': suite,
      'run': label,
      'eval_dir': str(eval_dir),
      'root': str(root),
      'raw_eval_name': raw,
      'summary_path': str(path),
  }


def load_summary_rows(paths, splits):
  rows = []
  bug_rows = []
  metadata = []
  for path in paths:
    info = infer_experiment(path)
    with path.open('r', encoding='utf-8') as f:
      data = json.load(f)
    metadata.append({
        **info,
        'threshold_quantile': safe_float(data.get('threshold_quantile')),
        'threshold_value': safe_float(data.get('threshold_value')),
    })
    for split in splits:
      split_data = data.get(split)
      if not isinstance(split_data, dict):
        continue
      row = {
          **info,
          'split': split,
          'threshold_quantile': safe_float(data.get('threshold_quantile')),
          'threshold_value': safe_float(data.get('threshold_value')),
      }
      for field in SUMMARY_FIELDS:
        row[field] = safe_float(split_data.get(field))
      if row.get('episode_task_score_mean') == '':
        row['episode_task_score_mean'] = row.get('episode_score_mean', '')
      if row.get('episode_training_reward_mean') == '':
        row['episode_training_reward_mean'] = row.get('episode_score_mean', '')
      if row.get('clean_task_score_mean') == '':
        row['clean_task_score_mean'] = row.get('clean_score_mean', '')
      if row.get('fault_task_score_mean') == '':
        row['fault_task_score_mean'] = row.get('fault_score_mean', '')
      row['theory_episode_false_alarm'] = theory_episode_false_alarm(split_data)
      rows.append(row)

      for family, count in split_data.get('trace_fault_family_counts', {}).items():
        bug_rows.append({
            **info,
            'split': split,
            'kind': 'family',
            'name': family,
            'count': count,
        })
      for typ, count in split_data.get('trace_fault_type_counts', {}).items():
        bug_rows.append({
            **info,
            'split': split,
            'kind': 'type',
            'name': typ,
            'count': count,
        })
  return metadata, rows, bug_rows


def theory_episode_false_alarm(split_data):
  alarm = split_data.get('step_alarm_rate')
  length = split_data.get('episode_length_mean')
  try:
    alarm = float(alarm)
    length = float(length)
  except Exception:
    return ''
  if math.isnan(alarm) or math.isnan(length):
    return ''
  alarm = min(max(alarm, 0.0), 1.0)
  return 1.0 - (1.0 - alarm) ** max(length, 0.0)


def markdown_table(headers, rows):
  out = []
  out.append('| ' + ' | '.join(headers) + ' |')
  out.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
  for row in rows:
    out.append('| ' + ' | '.join(str(row.get(h, '')) for h in headers) + ' |')
  return '\n'.join(out)


def make_core_markdown(summary_rows):
  grouped = {}
  for row in summary_rows:
    grouped.setdefault(row['suite'], []).append(row)

  lines = []
  lines.append('# Fault Result Summary')
  lines.append('')
  lines.append('Generated from tester_eval summary.json files.')
  lines.append('')

  for suite, rows in sorted(grouped.items()):
    lines.append(f'## {suite}')
    table_rows = []
    for row in sorted(rows, key=lambda x: (x['run'], CORE_SPLITS.index(x['split']) if x['split'] in CORE_SPLITS else 99)):
      if row['split'] == 'clean':
        continue
      table_rows.append({
          'run': row['run'],
          'split': row['split'],
          'AUROC': fmt(row.get('step_auroc')),
          'AUPRC': fmt(row.get('step_auprc'), digits=5),
          'ep_det': fmt(row.get('episode_detection_rate')),
          'fault_rate': fmt(row.get('step_fault_applied_rate'), digits=5),
          'task_score': fmt(row.get('episode_task_score_mean')),
          'clean_FA_ep': fmt(row.get('clean_false_alarm_episode_rate')),
      })
    if table_rows:
      lines.append(markdown_table(
          ['run', 'split', 'AUROC', 'AUPRC', 'ep_det', 'fault_rate', 'task_score', 'clean_FA_ep'],
          table_rows))
    lines.append('')

  lines.extend(make_pareto_markdown(summary_rows))
  lines.extend(make_competence_markdown(summary_rows))
  lines.extend(make_takeaway_section(summary_rows))
  return '\n'.join(lines) + '\n'


def make_takeaway_section(rows):
  lines = []
  lines.append('## Quick Takeaways')
  lines.append('')

  by_split = {}
  for row in rows:
    if row['split'] == 'clean':
      continue
    auroc = safe_float(row.get('step_auroc'))
    if auroc == '':
      continue
    by_split.setdefault(row['split'], []).append((float(auroc), row))

  for split in ['seen', 'holdout', 'semantic_holdout']:
    vals = sorted(by_split.get(split, []), key=lambda x: x[0], reverse=True)
    if not vals:
      continue
    best_val, best_row = vals[0]
    worst_val, worst_row = vals[-1]
    lines.append(
        f'- Best {split} AUROC: {best_row["run"]} '
        f'({best_val:.4f}, suite={best_row["suite"]}); '
        f'lowest: {worst_row["run"]} ({worst_val:.4f}).')

  semantic = by_split.get('semantic_holdout', [])
  if semantic:
    avg_sem = mean(v for v, _ in semantic)
    lines.append(
        f'- Mean semantic_holdout AUROC across available runs: {avg_sem:.4f}. '
        'Values near 0.5 suggest weak transfer to high-level semantic bugs.')

  false_alarm_rows = [
      row for row in rows
      if row['split'] != 'clean' and safe_float(row.get('clean_false_alarm_episode_rate')) != ''
  ]
  if false_alarm_rows:
    avg_fa = mean(float(row['clean_false_alarm_episode_rate']) for row in false_alarm_rows)
    lines.append(
        f'- Mean episode-level clean false alarm rate: {avg_fa:.4f}. '
        'Episode detection should be interpreted carefully; ranking metrics are safer.')

  lines.append('')
  lines.append('Suggested framing:')
  lines.append('')
  lines.append('- Seen/held-out subtype results test low-level transition fault transfer.')
  lines.append('- Semantic holdout results test a harder family-level/high-level setting.')
  lines.append('- Reward adaptation should be treated as a secondary use case unless it clearly beats task-only baselines.')
  lines.append('')
  return lines


def metric_value(row, metric):
  value = safe_float(row.get(metric))
  if value == '':
    return None
  try:
    value = float(value)
  except Exception:
    return None
  if math.isnan(value) or math.isinf(value):
    return None
  return value


def task_score_value(row):
  value = metric_value(row, 'episode_task_score_mean')
  if value is None:
    value = metric_value(row, 'episode_score_mean')
  return value


def choose_baseline(group):
  for name in BASELINE_RUNS:
    matches = [row for row in group if row['run'] == name]
    if matches:
      return matches[0]
  return None


def dominates(a, b, metrics):
  a_vals = [metric_value(a, metric) for metric in metrics]
  b_vals = [metric_value(b, metric) for metric in metrics]
  if any(v is None for v in a_vals) or any(v is None for v in b_vals):
    return False
  return all(x >= y for x, y in zip(a_vals, b_vals)) and any(
      x > y for x, y in zip(a_vals, b_vals))


def pareto_analysis(summary_rows):
  points = []
  frontier = []

  candidates = [
      row for row in summary_rows
      if row.get('split') != 'clean'
  ]

  groups = {}
  for row in candidates:
    groups.setdefault((row['suite'], row['split']), []).append(row)

  for objective, metrics in PARETO_OBJECTIVES.items():
    for (suite, split), rows in sorted(groups.items()):
      valid = [
          row for row in rows
          if all(metric_value(row, metric) is not None for metric in metrics)
      ]
      for row in valid:
        dominators = [
            other['run'] for other in valid
            if other is not row and dominates(other, row, metrics)
        ]
        dominated = bool(dominators)
        out = {
            'objective': objective,
            'suite': suite,
            'split': split,
            'run': row['run'],
            'is_pareto_frontier': int(not dominated),
            'dominated_by_count': len(dominators),
            'dominated_by': ';'.join(dominators),
            'raw_eval_name': row['raw_eval_name'],
            'root': row['root'],
            'eval_dir': row['eval_dir'],
            'summary_path': row['summary_path'],
        }
        for metric in metrics:
          out[metric] = row.get(metric)
        # Always include these for easier plotting even if not in objective.
        for metric in [
            'episode_score_mean', 'step_fault_applied_rate',
            'episode_task_score_mean', 'step_auroc', 'step_auprc', 'episode_detection_rate',
            'clean_false_alarm_episode_rate']:
          out.setdefault(metric, row.get(metric))
        points.append(out)
        if not dominated:
          frontier.append(out.copy())

  return points, frontier


def baseline_delta_rows(summary_rows):
  rows = []
  groups = {}
  for row in summary_rows:
    if row.get('split') == 'clean':
      continue
    groups.setdefault((row['suite'], row['split']), []).append(row)

  metrics = [
      'episode_score_mean',
      'episode_task_score_mean',
      'episode_training_reward_mean',
      'step_fault_applied_rate',
      'step_auroc',
      'step_auprc',
      'episode_detection_rate',
      'clean_false_alarm_episode_rate',
  ]

  for (suite, split), group in sorted(groups.items()):
    baseline = choose_baseline(group)
    if baseline is None:
      continue
    for row in group:
      out = {
          'suite': suite,
          'split': split,
          'run': row['run'],
          'baseline_run': baseline['run'],
          'raw_eval_name': row['raw_eval_name'],
          'root': row['root'],
          'eval_dir': row['eval_dir'],
          'summary_path': row['summary_path'],
      }
      for metric in metrics:
        value = metric_value(row, metric)
        base = metric_value(baseline, metric)
        out[metric] = row.get(metric)
        out[f'{metric}_baseline'] = baseline.get(metric)
        out[f'{metric}_delta'] = (
            value - base if value is not None and base is not None else '')
      task = task_score_value(row)
      base_task = task_score_value(baseline)
      retention = (
          task / base_task
          if task is not None and base_task is not None and abs(base_task) > 1e-8
          else '')
      out['task_score_retention'] = retention
      for threshold in COMPETENCE_THRESHOLDS:
        key = str(threshold).replace('.', '')
        out[f'competence_valid_{key}'] = (
            int(retention >= threshold) if retention != '' else '')
      rows.append(out)
  return rows


def competence_rows(summary_rows):
  rows = []
  groups = {}
  for row in summary_rows:
    if row.get('split') == 'clean':
      continue
    groups.setdefault((row['suite'], row['split']), []).append(row)

  for (suite, split), group in sorted(groups.items()):
    baseline = choose_baseline(group)
    if baseline is None:
      continue
    base_task = task_score_value(baseline)
    base_fault_rate = metric_value(baseline, 'step_fault_applied_rate')
    base_auroc = metric_value(baseline, 'step_auroc')
    base_auprc = metric_value(baseline, 'step_auprc')
    for row in group:
      task = task_score_value(row)
      fault_rate = metric_value(row, 'step_fault_applied_rate')
      auroc = metric_value(row, 'step_auroc')
      auprc = metric_value(row, 'step_auprc')
      retention = (
          task / base_task
          if task is not None and base_task is not None and abs(base_task) > 1e-8
          else '')
      out = {
          'suite': suite,
          'split': split,
          'run': row['run'],
          'baseline_run': baseline['run'],
          'task_score': task if task is not None else '',
          'baseline_task_score': base_task if base_task is not None else '',
          'task_score_retention': retention,
          'fault_rate': fault_rate if fault_rate is not None else '',
          'fault_rate_delta': (
              fault_rate - base_fault_rate
              if fault_rate is not None and base_fault_rate is not None else ''),
          'auroc': auroc if auroc is not None else '',
          'auroc_delta': (
              auroc - base_auroc
              if auroc is not None and base_auroc is not None else ''),
          'auprc': auprc if auprc is not None else '',
          'auprc_delta': (
              auprc - base_auprc
              if auprc is not None and base_auprc is not None else ''),
          'raw_eval_name': row['raw_eval_name'],
          'root': row['root'],
          'eval_dir': row['eval_dir'],
          'summary_path': row['summary_path'],
      }
      for threshold in COMPETENCE_THRESHOLDS:
        key = str(threshold).replace('.', '')
        out[f'competence_valid_{key}'] = (
            int(retention >= threshold) if retention != '' else '')
      rows.append(out)
  return rows


def make_competence_markdown(summary_rows):
  rows = competence_rows(summary_rows)
  lines = []
  lines.append('## Competence-Constrained View')
  lines.append('')
  lines.append(
      'Task score is treated as a constraint. Retention is computed against '
      'the task-only/reference baseline within the same suite and split.')
  lines.append('')

  valid90 = [
      row for row in rows
      if row.get('competence_valid_09') == 1 and row['run'] != row['baseline_run']
  ]
  if not valid90:
    lines.append('- No non-baseline run keeps task score retention >= 0.9 in the current summaries.')
    lines.append('')
    return lines

  table = []
  for row in sorted(valid90, key=lambda x: (x['suite'], x['split'], x['run'])):
    table.append({
        'suite': row['suite'],
        'split': row['split'],
        'run': row['run'],
        'baseline': row['baseline_run'],
        'retention': fmt(row.get('task_score_retention')),
        'fault_d': fmt(row.get('fault_rate_delta'), 5),
        'AUROC_d': fmt(row.get('auroc_delta')),
        'AUPRC_d': fmt(row.get('auprc_delta'), 5),
    })
  lines.append(markdown_table(
      ['suite', 'split', 'run', 'baseline', 'retention', 'fault_d', 'AUROC_d', 'AUPRC_d'],
      table))
  lines.append('')
  return lines


def make_pareto_markdown(summary_rows):
  _, frontier = pareto_analysis(summary_rows)
  lines = []
  lines.append('## Pareto View')
  lines.append('')
  lines.append(
      'A run is on the Pareto frontier if no other run in the same suite/split '
      'is at least as good on all selected objectives and better on one.')
  lines.append('')

  for objective in PARETO_OBJECTIVES:
    rows = [row for row in frontier if row['objective'] == objective]
    if not rows:
      continue
    lines.append(f'### {objective}')
    table = []
    for row in sorted(rows, key=lambda x: (x['suite'], x['split'], x['run'])):
      table.append({
          'suite': row['suite'],
          'split': row['split'],
          'run': row['run'],
          'score': fmt(row.get('episode_score_mean')),
          'task_score': fmt(row.get('episode_task_score_mean')),
          'fault_rate': fmt(row.get('step_fault_applied_rate'), 5),
          'AUROC': fmt(row.get('step_auroc')),
          'AUPRC': fmt(row.get('step_auprc'), 5),
      })
    lines.append(markdown_table(
        ['suite', 'split', 'run', 'task_score', 'fault_rate', 'AUROC', 'AUPRC'],
        table))
    lines.append('')
  return lines


def fallback_auroc(y_true, y_score):
  y_true = np.asarray(y_true).astype(bool)
  y_score = np.asarray(y_score, dtype=np.float64)
  n_pos = int(y_true.sum())
  n_neg = int((~y_true).sum())
  if n_pos == 0 or n_neg == 0:
    return ''
  order = np.argsort(y_score, kind='mergesort')
  ranks = np.empty_like(order, dtype=np.float64)
  sorted_scores = y_score[order]
  start = 0
  rank = 1.0
  while start < len(order):
    end = start + 1
    while end < len(order) and sorted_scores[end] == sorted_scores[start]:
      end += 1
    avg_rank = (rank + rank + (end - start) - 1) / 2.0
    ranks[order[start:end]] = avg_rank
    rank += end - start
    start = end
  sum_pos_ranks = ranks[y_true].sum()
  return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fallback_average_precision(y_true, y_score):
  y_true = np.asarray(y_true).astype(bool)
  y_score = np.asarray(y_score, dtype=np.float64)
  n_pos = int(y_true.sum())
  if n_pos == 0:
    return ''
  order = np.argsort(-y_score, kind='mergesort')
  truth = y_true[order].astype(np.float64)
  tp = np.cumsum(truth)
  precision = tp / np.arange(1, len(truth) + 1)
  return float((precision * truth).sum() / n_pos)


def load_step_records(path):
  records = []
  with path.open('r', encoding='utf-8') as f:
    for line in f:
      if not line.strip():
        continue
      row = json.loads(line)
      fault_applied = int(float(row.get('fault_applied', 0)) > 0.5)
      fault_manifested = int(float(row.get(
          'fault_manifested', fault_applied)) > 0.5)
      fault_trigger_context = int(float(row.get(
          'fault_trigger_context', fault_manifested)) > 0.5)
      semantic_trigger_context = int(float(row.get(
          'semantic_trigger_context', 0)) > 0.5)
      records.append({
          'episode_id': int(row.get('episode_id', 0)),
          'episode_step': int(row.get('episode_step', 0)),
          'fault_applied': fault_applied,
          'fault_manifested': fault_manifested,
          'fault_trigger_context': fault_trigger_context,
          'semantic_trigger_context': semantic_trigger_context,
          'lowlevel_trigger_context': int(float(row.get(
              'lowlevel_trigger_context', 0)) > 0.5),
          'fault_episode': int(float(row.get('fault_episode', 0)) > 0.5),
          'fault_exists_episode': int(float(row.get(
              'fault_exists_episode', row.get('fault_episode', 0))) > 0.5),
          'fault_manifest_prob': float(row.get('fault_manifest_prob', 0.0)),
          'fault_score': float(row.get('fault_score', row.get('ref_bug_score', 0.0))),
          'latent_kl_surprise': float(row.get('latent_kl_surprise', row.get('ref_bug_kl', 0.0))),
          'reward_prediction_error': float(row.get('reward_prediction_error', 0.0)),
      })
  return records


def trace_analysis(paths, splits, top_fracs, window):
  metric_rows = []
  window_rows = []

  for summary_path in paths:
    info = infer_experiment(summary_path)
    eval_dir = summary_path.parent
    for split in splits:
      steps_path = eval_dir / f'{split}_steps.jsonl'
      if not steps_path.exists():
        continue
      records = load_step_records(steps_path)
      if not records:
        continue

      scores = np.asarray([r['fault_score'] for r in records], dtype=np.float64)
      for label, key in TRACE_LABELS:
        y_true = np.asarray([r.get(key, 0) for r in records], dtype=np.int32)
        pos = scores[y_true == 1]
        neg = scores[y_true == 0]

        row = {
            **info,
            'split': split,
            'label': label,
            'n_steps': int(len(records)),
            'n_fault_steps': int(y_true.sum()),
            'fault_step_rate': float(y_true.mean()) if len(y_true) else '',
            'score_mean_all': float(scores.mean()) if len(scores) else '',
            'score_mean_fault': float(pos.mean()) if len(pos) else '',
            'score_mean_normal': float(neg.mean()) if len(neg) else '',
            'score_p95_fault': float(np.quantile(pos, 0.95)) if len(pos) else '',
            'score_p95_normal': float(np.quantile(neg, 0.95)) if len(neg) else '',
            'score_p99_fault': float(np.quantile(pos, 0.99)) if len(pos) else '',
            'score_p99_normal': float(np.quantile(neg, 0.99)) if len(neg) else '',
            'auroc_from_steps': fallback_auroc(y_true, scores),
            'auprc_from_steps': fallback_average_precision(y_true, scores),
        }

        order = np.argsort(-scores, kind='mergesort')
        for frac in top_fracs:
          k = max(1, int(round(len(order) * frac)))
          top = y_true[order[:k]]
          row[f'precision_at_top_{frac:g}'] = float(top.mean()) if len(top) else ''
          row[f'n_top_{frac:g}'] = int(k)
        metric_rows.append(row)

      window_rows.extend(event_window_rows(
          info, split, records, window, label='fault_manifested',
          event_key='fault_manifested'))
      window_rows.extend(event_window_rows(
          info, split, records, window, label='fault_trigger_context',
          event_key='fault_trigger_context'))

  return metric_rows, window_rows


def event_window_rows(info, split, records, window, label, event_key):
  by_ep = {}
  for rec in records:
    by_ep.setdefault(rec['episode_id'], []).append(rec)

  sums = {rel: 0.0 for rel in range(-window, window + 1)}
  counts = {rel: 0 for rel in range(-window, window + 1)}
  event_count = 0

  for ep_records in by_ep.values():
    ep_records.sort(key=lambda x: x['episode_step'])
    first_fault_idx = None
    for idx, rec in enumerate(ep_records):
      if rec.get(event_key, 0):
        first_fault_idx = idx
        break
    if first_fault_idx is None:
      continue
    event_count += 1
    for rel in range(-window, window + 1):
      idx = first_fault_idx + rel
      if 0 <= idx < len(ep_records):
        sums[rel] += ep_records[idx]['fault_score']
        counts[rel] += 1

  if event_count == 0:
    return []

  rows = []
  for rel in range(-window, window + 1):
    rows.append({
        **info,
        'split': split,
        'event_label': label,
        'relative_step': rel,
        'mean_fault_score': sums[rel] / counts[rel] if counts[rel] else '',
        'n': counts[rel],
        'n_events': event_count,
    })
  return rows


def main():
  args = parse_args()
  splits = [x.strip() for x in args.splits.split(',') if x.strip()]
  top_fracs = [float(x.strip()) for x in args.top_fracs.split(',') if x.strip()]
  if args.outdir:
    outdir = Path(args.outdir).expanduser()
  else:
    stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path(f'/home/railab/logdir/fault_analysis_{stamp}')
  outdir.mkdir(parents=True, exist_ok=True)

  summary_paths = discover_summaries(args.roots, args.summary)
  if not summary_paths:
    raise SystemExit('No summary.json files found.')

  metadata, summary_rows, bug_rows = load_summary_rows(summary_paths, splits)
  pareto_points, pareto_frontier = pareto_analysis(summary_rows)
  delta_rows = baseline_delta_rows(summary_rows)
  constrained_rows = competence_rows(summary_rows)

  write_json(outdir / 'metadata.json', {
      'roots': args.roots,
      'summary_paths': [str(x) for x in summary_paths],
      'splits': splits,
      'trace_analysis': bool(args.trace_analysis),
  })
  write_csv(
      outdir / 'summary_metrics.csv',
      summary_rows,
      ['suite', 'run', 'split', 'raw_eval_name', 'root', 'eval_dir',
       'threshold_quantile', 'threshold_value'] + SUMMARY_FIELDS +
      ['theory_episode_false_alarm', 'summary_path'])
  write_csv(
      outdir / 'bug_counts.csv',
      bug_rows,
      ['suite', 'run', 'split', 'kind', 'name', 'count',
       'raw_eval_name', 'root', 'eval_dir', 'summary_path'])
  write_csv(
      outdir / 'runs.csv',
      metadata,
      ['suite', 'run', 'raw_eval_name', 'root', 'eval_dir',
       'threshold_quantile', 'threshold_value', 'summary_path'])
  pareto_fields = [
      'objective', 'suite', 'split', 'run', 'is_pareto_frontier',
      'dominated_by_count', 'dominated_by',
      'episode_score_mean', 'episode_task_score_mean',
      'step_fault_applied_rate', 'step_auroc', 'step_auprc',
      'episode_detection_rate',
      'clean_false_alarm_episode_rate',
      'raw_eval_name', 'root', 'eval_dir', 'summary_path']
  write_csv(outdir / 'pareto_points.csv', pareto_points, pareto_fields)
  write_csv(outdir / 'pareto_frontier.csv', pareto_frontier, pareto_fields)
  delta_fields = [
      'suite', 'split', 'run', 'baseline_run',
      'episode_score_mean', 'episode_score_mean_baseline',
      'episode_score_mean_delta',
      'episode_task_score_mean', 'episode_task_score_mean_baseline',
      'episode_task_score_mean_delta',
      'episode_training_reward_mean',
      'episode_training_reward_mean_baseline',
      'episode_training_reward_mean_delta',
      'task_score_retention',
      'competence_valid_09',
      'competence_valid_08',
      'step_fault_applied_rate', 'step_fault_applied_rate_baseline',
      'step_fault_applied_rate_delta',
      'step_auroc', 'step_auroc_baseline', 'step_auroc_delta',
      'step_auprc', 'step_auprc_baseline', 'step_auprc_delta',
      'episode_detection_rate', 'episode_detection_rate_baseline',
      'episode_detection_rate_delta',
      'clean_false_alarm_episode_rate',
      'clean_false_alarm_episode_rate_baseline',
      'clean_false_alarm_episode_rate_delta',
      'raw_eval_name', 'root', 'eval_dir', 'summary_path']
  write_csv(outdir / 'baseline_deltas.csv', delta_rows, delta_fields)
  competence_fields = [
      'suite', 'split', 'run', 'baseline_run',
      'task_score', 'baseline_task_score', 'task_score_retention',
      'competence_valid_09', 'competence_valid_08',
      'fault_rate', 'fault_rate_delta',
      'auroc', 'auroc_delta',
      'auprc', 'auprc_delta',
      'raw_eval_name', 'root', 'eval_dir', 'summary_path']
  write_csv(outdir / 'competence_constrained.csv', constrained_rows, competence_fields)

  report = make_core_markdown(summary_rows)
  (outdir / 'report.md').write_text(report, encoding='utf-8')

  if args.trace_analysis:
    metric_rows, window_rows = trace_analysis(
        summary_paths, splits, top_fracs, args.window)
    trace_fields = [
        'suite', 'run', 'split', 'label', 'n_steps', 'n_fault_steps',
        'fault_step_rate', 'score_mean_all', 'score_mean_fault',
        'score_mean_normal', 'score_p95_fault', 'score_p95_normal',
        'score_p99_fault', 'score_p99_normal', 'auroc_from_steps',
        'auprc_from_steps',
    ]
    for frac in top_fracs:
      trace_fields.extend([f'precision_at_top_{frac:g}', f'n_top_{frac:g}'])
    trace_fields.extend(['raw_eval_name', 'root', 'eval_dir', 'summary_path'])
    write_csv(outdir / 'trace_ranking_metrics.csv', metric_rows, trace_fields)
    write_csv(
        outdir / 'event_window_fault_score.csv',
        window_rows,
        ['suite', 'run', 'split', 'event_label', 'relative_step', 'mean_fault_score',
         'n', 'n_events', 'raw_eval_name', 'root', 'eval_dir', 'summary_path'])

  print(f'Found {len(summary_paths)} summary files.')
  print(f'Wrote: {outdir}')
  print(f'- {outdir / "report.md"}')
  print(f'- {outdir / "summary_metrics.csv"}')
  print(f'- {outdir / "bug_counts.csv"}')
  print(f'- {outdir / "pareto_points.csv"}')
  print(f'- {outdir / "pareto_frontier.csv"}')
  print(f'- {outdir / "baseline_deltas.csv"}')
  print(f'- {outdir / "competence_constrained.csv"}')
  if args.trace_analysis:
    print(f'- {outdir / "trace_ranking_metrics.csv"}')
    print(f'- {outdir / "event_window_fault_score.csv"}')


if __name__ == '__main__':
  main()

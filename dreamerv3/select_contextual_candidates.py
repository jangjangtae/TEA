#!/usr/bin/env python3
"""Turn a short contextual-fault pilot into a conservative long-run shortlist."""

import argparse
import csv
import json
import math
from pathlib import Path


CORE_SPLITS = ('seen', 'holdout', 'sparse')
BASELINES = ('taskonly', 'excess_delta_p95_beta02')


def number(row, key, default=0.0):
  try:
    value = float(row.get(key, default))
    return value if math.isfinite(value) else default
  except (TypeError, ValueError):
    return default


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--analysis', required=True)
  parser.add_argument('--task-retention', type=float, default=0.85)
  parser.add_argument('--clean-fp-tolerance', type=float, default=0.05)
  parser.add_argument('--top-k', type=int, default=2)
  return parser.parse_args()


def main():
  args = parse_args()
  analysis = Path(args.analysis).expanduser()
  with (analysis / 'aggregate_metrics.csv').open() as f:
    rows = list(csv.DictReader(f))
  eval_rows = {
      (row['variant'], row['split']): row for row in rows
      if row.get('phase') == 'eval'}
  variants = sorted({variant for variant, _ in eval_rows})
  baseline = {
      split: eval_rows.get(('taskonly', split), {})
      for split in ('clean',) + CORE_SPLITS}

  summaries = []
  for variant in variants:
    if variant == 'reference' or variant in BASELINES:
      continue
    split_rows = [eval_rows.get((variant, split), {}) for split in CORE_SPLITS]
    if not all(split_rows) or not all(baseline[split] for split in CORE_SPLITS):
      continue
    retention = []
    bug_delta = []
    auroc_delta = []
    coverage_delta = []
    unique_delta = []
    for split, row in zip(CORE_SPLITS, split_rows):
      base = baseline[split]
      task = number(row, 'episode_score_mean_mean')
      base_task = number(base, 'episode_score_mean_mean')
      retention.append(task / base_task if abs(base_task) > 1e-8 else 0.0)
      bug_delta.append(number(row, 'fault_applied_rate_mean') - number(base, 'fault_applied_rate_mean'))
      auroc_delta.append(number(row, 'auroc_mean', 0.5) - number(base, 'auroc_mean', 0.5))
      coverage_delta.append(number(row, 'semantic_context_coverage_mean') - number(base, 'semantic_context_coverage_mean'))
      unique_delta.append(number(row, 'unique_suspicious_context_count_mean') - number(base, 'unique_suspicious_context_count_mean'))

    clean = eval_rows.get((variant, 'clean'), {})
    clean_base = baseline['clean']
    clean_fp_delta = (
        number(clean, 'suspicious_context_rate_mean') -
        number(clean_base, 'suspicious_context_rate_mean'))
    min_retention = min(retention)
    mean_retention = sum(retention) / len(retention)
    mean_bug_delta = sum(bug_delta) / len(bug_delta)
    mean_auroc_delta = sum(auroc_delta) / len(auroc_delta)
    mean_coverage_delta = sum(coverage_delta) / len(coverage_delta)
    mean_unique_delta = sum(unique_delta) / len(unique_delta)

    task_ok = min_retention >= args.task_retention
    clean_fp_ok = clean_fp_delta <= args.clean_fp_tolerance
    behavior_signal = mean_coverage_delta > 0 or mean_unique_delta > 0
    fault_signal = mean_bug_delta > 0 or mean_auroc_delta > 0.01
    verdict = (
        'GO' if task_ok and clean_fp_ok and behavior_signal and fault_signal else
        'WATCH' if task_ok and clean_fp_ok and (behavior_signal or fault_signal) else
        'NO_GO')
    score = (
        200.0 * mean_bug_delta + mean_auroc_delta +
        0.002 * mean_unique_delta + 0.05 * (mean_retention - 1.0) -
        max(clean_fp_delta, 0.0))
    summaries.append({
        'variant': variant,
        'verdict': verdict,
        'score': score,
        'task_retention_min': min_retention,
        'task_retention_mean': mean_retention,
        'bug_rate_delta_mean': mean_bug_delta,
        'auroc_delta_mean': mean_auroc_delta,
        'semantic_coverage_delta_mean': mean_coverage_delta,
        'unique_suspicious_delta_mean': mean_unique_delta,
        'clean_suspicious_rate_delta': clean_fp_delta,
    })

  rank = {'GO': 2, 'WATCH': 1, 'NO_GO': 0}
  summaries.sort(key=lambda x: (rank[x['verdict']], x['score']), reverse=True)
  candidates = [x['variant'] for x in summaries if x['verdict'] != 'NO_GO']
  candidates = candidates[:args.top_k]
  recommended = list(BASELINES) + candidates

  (analysis / 'pilot_candidate_summary.json').write_text(
      json.dumps({'recommended': recommended, 'variants': summaries}, indent=2),
      encoding='utf-8')
  (analysis / 'recommended_variants.txt').write_text(
      ' '.join(recommended) + '\n', encoding='utf-8')
  lines = [
      '# Contextual Pilot Decision', '',
      f'Task retention gate: {args.task_retention:.0%}.',
      f'Clean suspicious-rate tolerance: +{args.clean_fp_tolerance:.1%}.',
      'This is a one-seed feasibility gate, not final evidence.', '',
      '## Ranking', '']
  for item in summaries:
    lines.append(
        f"- **{item['verdict']}** `{item['variant']}`: "
        f"retention(min)={item['task_retention_min']:.3f}, "
        f"bug_delta={item['bug_rate_delta_mean']:.5f}, "
        f"AUROC_delta={item['auroc_delta_mean']:.4f}, "
        f"coverage_delta={item['semantic_coverage_delta_mean']:.1f}, "
        f"unique_delta={item['unique_suspicious_delta_mean']:.1f}, "
        f"clean_FP_delta={item['clean_suspicious_rate_delta']:.4f}")
  lines += ['', '## Week-Long Shortlist', '', f"`{' '.join(recommended)}`", '']
  (analysis / 'pilot_decision.md').write_text('\n'.join(lines), encoding='utf-8')
  print('Recommended variants:', ' '.join(recommended))
  print('Decision:', analysis / 'pilot_decision.md')


if __name__ == '__main__':
  main()

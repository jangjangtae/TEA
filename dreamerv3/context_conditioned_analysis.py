#!/usr/bin/env python3
"""Context-conditioned fault-score analysis for tester_eval traces."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
  from sklearn.metrics import average_precision_score, roc_auc_score
  _HAVE_SKLEARN = True
except Exception:
  _HAVE_SKLEARN = False


CONDITIONS = {
    'all': None,
    'semantic_context': ['semantic_trigger_context'],
    'ctx_upgrade_collect': ['semantic_ctx_upgrade_collect'],
    'ctx_retry_craft': ['semantic_ctx_retry_craft'],
    'ctx_relocate_station': ['semantic_ctx_relocate_station'],
    'ctx_valid_progress': ['semantic_ctx_valid_progress'],
    'ctx_station_reuse': ['semantic_ctx_station_reuse'],
    'ctx_delayed_after_use': ['semantic_ctx_delayed_after_use'],
    'reward_event': ['task_reward'],
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--roots', nargs='+', required=True,
      help='Experiment roots to search for tester_eval summary/steps files.')
  parser.add_argument(
      '--outdir', required=True,
      help='Output directory for context_conditioned_metrics.csv.')
  parser.add_argument(
      '--splits', default='seen,holdout,semantic_holdout',
      help='Comma separated split names to analyze.')
  parser.add_argument(
      '--top-fracs', default='0.001,0.005,0.01,0.05',
      help='Comma separated fractions for precision@top-k.')
  parser.add_argument(
      '--labels', default='fault_manifested,fault_trigger_context',
      help='Comma separated label columns to evaluate.')
  return parser.parse_args()


def discover_eval_dirs(roots):
  eval_dirs = []
  for root in roots:
    root = Path(root).expanduser()
    if not root.exists():
      continue
    for summary in root.rglob('tester_eval/summary.json'):
      eval_dirs.append(summary.parent)
  return sorted(set(eval_dirs))


def load_summary(path):
  try:
    with path.open('r', encoding='utf-8') as f:
      return json.load(f)
  except Exception:
    return {}


def infer_run(eval_dir):
  name = eval_dir.parent.name
  mapping = (
      ('reference', 'reference'),
      ('beta005', 'fault_beta0.05'),
      ('beta01', 'fault_beta0.1_repeat'),
      ('beta02', 'fault_beta0.2'),
      ('beta05', 'fault_beta0.5'),
      ('task_only', 'task_only_repeat'),
      ('oracle', 'oracle'),
      ('tester', 'tester_reward'),
      ('gated', 'fault_gated'),
      ('ungated', 'fault_ungated'),
  )
  for key, value in mapping:
    if key in name:
      return value
  return name


def fallback_auroc(y_true, y_score):
  y_true = np.asarray(y_true).astype(bool)
  y_score = np.asarray(y_score, dtype=np.float64)
  n_pos = int(y_true.sum())
  n_neg = int((~y_true).sum())
  if n_pos == 0 or n_neg == 0:
    return np.nan
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
  return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fallback_ap(y_true, y_score):
  y_true = np.asarray(y_true).astype(bool)
  y_score = np.asarray(y_score, dtype=np.float64)
  n_pos = int(y_true.sum())
  if n_pos == 0:
    return np.nan
  order = np.argsort(-y_score, kind='mergesort')
  hits = y_true[order].astype(np.float64)
  precision = np.cumsum(hits) / (np.arange(len(hits)) + 1)
  return float((precision * hits).sum() / n_pos)


def bool_series(df, col):
  if col not in df:
    return pd.Series(False, index=df.index)
  if col == 'task_reward':
    return pd.to_numeric(df[col], errors='coerce').fillna(0).abs() > 1e-6
  return pd.to_numeric(df[col], errors='coerce').fillna(0) > 0


def condition_mask(df, name, cols):
  if cols is None:
    return pd.Series(True, index=df.index)
  mask = pd.Series(False, index=df.index)
  for col in cols:
    mask |= bool_series(df, col)
  return mask


def metrics_for(df, top_fracs, label):
  if len(df) == 0:
    return None
  score_col = 'fault_score' if 'fault_score' in df else 'ref_bug_score'
  fallback = df.get('fault_applied', 0)
  y_true = pd.to_numeric(df.get(label, fallback), errors='coerce').fillna(0).astype(int)
  y_score = pd.to_numeric(df[score_col], errors='coerce').fillna(0.0)
  row = {
      'n_steps': int(len(df)),
      'n_fault_steps': int(y_true.sum()),
      'fault_step_rate': float(y_true.mean()) if len(y_true) else np.nan,
      'score_mean_all': float(y_score.mean()),
      'score_mean_fault': float(y_score[y_true > 0].mean()) if int(y_true.sum()) else np.nan,
      'score_mean_normal': float(y_score[y_true == 0].mean()) if int((y_true == 0).sum()) else np.nan,
      'score_p95_fault': float(np.quantile(y_score[y_true > 0], 0.95)) if int(y_true.sum()) else np.nan,
      'score_p95_normal': float(np.quantile(y_score[y_true == 0], 0.95)) if int((y_true == 0).sum()) else np.nan,
      'score_p99_fault': float(np.quantile(y_score[y_true > 0], 0.99)) if int(y_true.sum()) else np.nan,
      'score_p99_normal': float(np.quantile(y_score[y_true == 0], 0.99)) if int((y_true == 0).sum()) else np.nan,
  }
  if len(np.unique(y_true)) > 1:
    if _HAVE_SKLEARN:
      row['auroc'] = float(roc_auc_score(y_true, y_score))
      row['auprc'] = float(average_precision_score(y_true, y_score))
    else:
      row['auroc'] = fallback_auroc(y_true, y_score)
      row['auprc'] = fallback_ap(y_true, y_score)
  else:
    row['auroc'] = np.nan
    row['auprc'] = np.nan

  order = np.argsort(-y_score.values, kind='mergesort')
  labels = y_true.values
  for frac in top_fracs:
    n_top = max(1, int(round(len(df) * frac)))
    row[f'precision_at_top_{frac:g}'] = float(labels[order[:n_top]].mean())
    row[f'n_top_{frac:g}'] = int(n_top)
  return row


def main():
  args = parse_args()
  splits = [x.strip() for x in args.splits.split(',') if x.strip()]
  top_fracs = [float(x.strip()) for x in args.top_fracs.split(',') if x.strip()]
  labels = [x.strip() for x in args.labels.split(',') if x.strip()]
  outdir = Path(args.outdir).expanduser()
  outdir.mkdir(parents=True, exist_ok=True)

  rows = []
  for eval_dir in discover_eval_dirs(args.roots):
    summary = load_summary(eval_dir / 'summary.json')
    root = next((p for p in eval_dir.parents if p.name.startswith('fault_')), eval_dir.parent)
    for split in splits:
      path = eval_dir / f'{split}_steps.jsonl'
      if not path.exists():
        continue
      try:
        df = pd.read_json(path, orient='records', lines=True)
      except ValueError:
        continue
      for cond, cols in CONDITIONS.items():
        mask = condition_mask(df, cond, cols)
        sub = df[mask]
        for label in labels:
          mets = metrics_for(sub, top_fracs, label)
          if mets is None:
            continue
          rows.append({
              'suite': 'semantic_holdout_eval' if 'semantic' in str(eval_dir) else 'eval',
              'run': infer_run(eval_dir),
              'split': split,
              'condition': cond,
              'label': label,
              'raw_eval_name': eval_dir.parent.name,
              'root': str(root),
              'eval_dir': str(eval_dir.parent),
              'summary_path': str(eval_dir / 'summary.json'),
              'summary_split_auroc': summary.get(split, {}).get('step_auroc', np.nan),
              **mets,
          })

  fields = [
      'suite', 'run', 'split', 'condition', 'label', 'n_steps', 'n_fault_steps',
      'fault_step_rate', 'score_mean_all', 'score_mean_fault',
      'score_mean_normal', 'score_p95_fault', 'score_p95_normal',
      'score_p99_fault', 'score_p99_normal', 'auroc', 'auprc',
  ]
  for frac in top_fracs:
    fields.extend([f'precision_at_top_{frac:g}', f'n_top_{frac:g}'])
  fields.extend(['summary_split_auroc', 'raw_eval_name', 'root', 'eval_dir', 'summary_path'])

  outpath = outdir / 'context_conditioned_metrics.csv'
  with outpath.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({key: row.get(key, '') for key in fields})

  print(f'Wrote {len(rows)} rows to {outpath}')


if __name__ == '__main__':
  main()

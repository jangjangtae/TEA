#!/usr/bin/env python3
"""Match dense-surprise and RND bonus budgets to FATE on one probe trace.

The probe policy must receive task reward only. It may still update the RND
predictor and log frozen-reference scores. This lets all three intrinsic signals
be measured on the same transitions without using fault labels or evaluation
results to choose their reward coefficients.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def _mean(values):
  return sum(values) / len(values)


def _percentile(values, quantile):
  values = sorted(values)
  if not values:
    return float('nan')
  position = (len(values) - 1) * quantile
  lower = int(math.floor(position))
  upper = int(math.ceil(position))
  if lower == upper:
    return values[lower]
  weight = position - lower
  return values[lower] * (1.0 - weight) + values[upper] * weight


def _summary(values):
  return {
      'count': len(values),
      'mean': _mean(values),
      'p50': _percentile(values, 0.50),
      'p95': _percentile(values, 0.95),
      'p99': _percentile(values, 0.99),
      'max': max(values),
      'positive_rate': sum(value > 0.0 for value in values) / len(values),
  }


def _load_rows(path):
  rows = []
  with path.open(encoding='utf-8') as infile:
    for lineno, line in enumerate(infile, 1):
      line = line.strip()
      if not line:
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON at {path}:{lineno}: {exc}') from exc
      if not row.get('is_first', False):
        rows.append(row)
  return rows


def _finite(row, key):
  try:
    value = float(row[key])
  except (KeyError, TypeError, ValueError):
    return None
  return value if math.isfinite(value) else None


def _scaled(value, multiplier):
  return float(value * multiplier)


def calibrate(args):
  for name in ('fate_beta', 'delta_threshold', 'clip', 'warmup_fraction'):
    if not math.isfinite(getattr(args, name)):
      raise ValueError(f'{name} must be finite')
  if args.fate_beta <= 0 or args.delta_threshold < 0 or args.min_rows < 1:
    raise ValueError('Invalid bonus coefficient, threshold, or minimum row count')
  if not 0.0 <= args.warmup_fraction < 1.0:
    raise ValueError('warmup_fraction must be in [0, 1)')
  trace = Path(args.trace).expanduser().resolve()
  rows = _load_rows(trace)
  if not rows:
    raise ValueError(f'No non-reset transitions found in {trace}')

  steps = [_finite(row, 'global_step') for row in rows]
  if any(step is None for step in steps):
    raise ValueError(f'Invalid or missing global_step in {trace}')
  if any(b <= a for a, b in zip(steps, steps[1:])):
    raise ValueError('Probe steps must be strictly increasing; mixed/resumed trace')
  # Fail closed: alpha=0, delivered task reward, no applied intrinsic reward.
  # These fields are checked before discarding warmup rows.
  for row in rows:
    values = {key: _finite(row, key) for key in (
        'rnd_alpha', 'rnd_reward_bonus', 'task_reward_source_id',
        'learning_reward', 'task_reward_delivered', 'augmented_reward',
        'fault_score', 'clipped_fault_score', 'fault_score_delta',
        'rnd_intrinsic_reward', 'latent_kl_surprise', 'fault_score_raw')}
    if any(value is None for value in values.values()):
      raise ValueError('Probe has missing/non-finite reward or score diagnostics')
    if any(values[key] != 0.0 for key in (
        'rnd_alpha', 'rnd_reward_bonus', 'task_reward_source_id')):
      raise ValueError('Expected a delivered-reward task-only probe with RND alpha=0')
    if not all(math.isclose(values[key], values['task_reward_delivered'],
        rel_tol=1e-6, abs_tol=1e-7) for key in ('learning_reward', 'augmented_reward')):
      raise ValueError('Probe learning reward differs from delivered task reward')
    if not math.isclose(values['clipped_fault_score'],
        max(0.0, min(args.clip, values['fault_score'])), rel_tol=1e-6, abs_tol=1e-7):
      raise ValueError('Probe and coefficient calibration use different clipping')
    if not math.isclose(values['fault_score_raw'], values['latent_kl_surprise'],
        rel_tol=1e-6, abs_tol=1e-7):
      raise ValueError('Expected a latent-KL-only probe score')
  first_step, last_step = min(steps), max(steps)
  warmup_step = first_step + args.warmup_fraction * (last_step - first_step)
  rows = [
      row for row in rows
      if (_finite(row, 'global_step') is not None and
          _finite(row, 'global_step') >= warmup_step)
  ]
  if len(rows) < args.min_rows:
    raise ValueError(
        f'Only {len(rows)} transitions remain after warmup; '
        f'need at least {args.min_rows}.')

  fate_scores = []
  dense_scores = []
  rnd_scores = []
  denominator = args.clip - args.delta_threshold
  if denominator <= 0.0:
    raise ValueError('--clip must be greater than --delta-threshold')

  for row in rows:
    delta = _finite(row, 'fault_score_delta')
    dense = _finite(row, 'clipped_fault_score')
    rnd = _finite(row, 'rnd_intrinsic_reward')
    if min(delta, dense, rnd) < 0.0:
      raise ValueError('Probe contains a negative intrinsic signal')
    fate_scores.append(max(0.0, min(1.0, (
        delta - args.delta_threshold) / denominator)))
    dense_scores.append(max(0.0, dense))
    rnd_scores.append(max(0.0, rnd))

  if len(fate_scores) < args.min_rows:
    raise ValueError(
        'Probe trace does not contain enough aligned FATE, DenseSurprise, and '
        'RND signals. Run the reward_scale_probe variant with the updated code.')

  fate = _summary(fate_scores)
  dense = _summary(dense_scores)
  rnd = _summary(rnd_scores)
  target_mean_bonus = args.fate_beta * fate['mean']
  if target_mean_bonus <= 0.0:
    raise ValueError(
        'The probe produced no positive FATE bonus after warmup; increase the '
        'probe length before calibrating reward scales.')
  if dense['mean'] <= 0.0 or rnd['mean'] <= 0.0:
    raise ValueError('DenseSurprise and RND probe means must both be positive.')

  dense_beta = target_mean_bonus / dense['mean']
  rnd_alpha = target_mean_bonus / rnd['mean']
  multipliers = [float(value) for value in args.grid_multipliers.split(',')]
  if not multipliers or any(
      not math.isfinite(value) or value <= 0.0 for value in multipliers):
    raise ValueError('--grid-multipliers must contain positive numbers')

  digest = hashlib.sha256(trace.read_bytes()).hexdigest()
  return {
      'protocol_version': 2,
      'selection_protocol': (
          'Shared task-only adaptation probe; no fault labels or evaluation '
          'outcomes are used for coefficient matching.'),
      'trace': str(trace),
      'trace_sha256': digest,
      'rows_after_warmup': len(fate_scores),
      'step_range': [first_step, last_step],
      'warmup_fraction': args.warmup_fraction,
      'warmup_step': warmup_step,
      'fate': {
          'beta': args.fate_beta,
          'delta_threshold': args.delta_threshold,
          'clip': args.clip,
          'score': fate,
          'target_mean_bonus': target_mean_bonus,
      },
      'dense_surprise': {
          'score': dense,
          'matched_beta': dense_beta,
          'validation_grid': [
              _scaled(dense_beta, multiplier) for multiplier in multipliers],
      },
      'rnd': {
          'score': rnd,
          'matched_alpha': rnd_alpha,
          'validation_grid': [
              _scaled(rnd_alpha, multiplier) for multiplier in multipliers],
      },
      'grid_multipliers': multipliers,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--trace', required=True)
  parser.add_argument('--out', required=True)
  parser.add_argument('--env-out')
  parser.add_argument('--fate-beta', type=float, default=0.2)
  parser.add_argument('--delta-threshold', type=float, default=0.5)
  parser.add_argument('--clip', type=float, default=2.0)
  parser.add_argument('--warmup-fraction', type=float, default=0.2)
  parser.add_argument('--grid-multipliers', default='0.5,1.0,2.0')
  parser.add_argument('--min-rows', type=int, default=1000)
  args = parser.parse_args()
  if not 0.0 <= args.warmup_fraction < 1.0:
    parser.error('--warmup-fraction must be in [0, 1)')

  result = calibrate(args)
  out = Path(args.out).expanduser()
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
  print(f'Wrote {out}')
  print(
      f"FATE target mean bonus: {result['fate']['target_mean_bonus']:.8g}\n"
      f"DenseSurprise matched beta: "
      f"{result['dense_surprise']['matched_beta']:.8g}\n"
      f"RND matched alpha: {result['rnd']['matched_alpha']:.8g}")

  if args.env_out:
    env_out = Path(args.env_out).expanduser()
    env_out.parent.mkdir(parents=True, exist_ok=True)
    env_out.write_text(
        'DENSE_MATCHED_BETA=' +
        f"{result['dense_surprise']['matched_beta']:.17g}\n" +
        'RND_MATCHED_ALPHA=' +
        f"{result['rnd']['matched_alpha']:.17g}\n")
    print(f'Wrote {env_out}')


if __name__ == '__main__':
  main()

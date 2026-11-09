import collections
import json
import os
from collections import deque
from functools import partial as bind
from pathlib import Path

import numpy as np
import pandas as pd
import elements
import embodied

try:
  from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
  _HAVE_SKLEARN = True
except Exception:
  _HAVE_SKLEARN = False


DEFAULT_EVAL_SPLITS = ('clean', 'seen', 'holdout')

DEFAULT_SEMANTIC_HOLDOUT_SUBTYPES = (
    'tool_collect_desync_on_upgrade,'
    'craft_result_missing_on_retry,'
    'station_place_ghost_on_relocate,'
    'achievement_unlock_missing_after_valid_progress,'
    'station_usable_flag_broken_after_relocate,'
    'recipe_precondition_mischeck_on_retry,'
    'delayed_inventory_desync_after_station_use')

SEMANTIC_CONTEXT_KEYS = (
    'upgrade_collect',
    'retry_craft',
    'relocate_station',
    'valid_progress',
    'station_reuse',
    'delayed_after_use',
)


def _fallback_auroc(y_true, y_score):
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
  sum_pos_ranks = ranks[y_true].sum()
  return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fallback_average_precision(y_true, y_score):
  y_true = np.asarray(y_true).astype(bool)
  y_score = np.asarray(y_score, dtype=np.float64)
  n_pos = int(y_true.sum())
  if n_pos == 0:
    return np.nan
  order = np.argsort(-y_score, kind='mergesort')
  truth = y_true[order].astype(np.float64)
  tp = np.cumsum(truth)
  precision = tp / np.arange(1, len(truth) + 1)
  return float((precision * truth).sum() / n_pos)


def _scalar(x, default=0.0):
  if x is None:
    return default
  arr = np.asarray(x)
  if arr.size == 0:
    return default
  return float(arr.reshape(-1)[0])


class CoverageTracker:

  def __init__(self, stride=8, recent_window=200):
    self.stride = int(stride)
    self.recent_window = int(recent_window)

    self.global_counts = collections.defaultdict(int)
    self.episode_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    self.recent_novel = collections.deque(maxlen=self.recent_window)

    self.action_sets = collections.defaultdict(set)
    self.action_bigram_sets = collections.defaultdict(set)
    self.action_hist = collections.defaultdict(lambda: deque(maxlen=2))

  def _hash(self, image):
    image = np.asarray(image)
    if image.ndim != 3:
      return None
    small = image[::self.stride, ::self.stride]
    if small.shape[-1] == 3:
      small = small.mean(-1)
    small = small.astype(np.uint8, copy=False)
    return small.tobytes()

  def reset_episode(self, worker):
    self.episode_counts[worker].clear()
    self.action_sets[worker].clear()
    self.action_bigram_sets[worker].clear()
    self.action_hist[worker].clear()

  def step(self, worker, image, action):
    key = self._hash(image)

    is_novel = 0.0
    global_unique = 0.0
    episode_unique = 0.0
    revisit_ratio = 0.0
    recent_novel_rate = 0.0
    unique_actions = 0.0
    unique_action_bigrams = 0.0

    if key is not None:
      is_novel = 1.0 if self.global_counts[key] == 0 else 0.0
      self.global_counts[key] += 1
      self.episode_counts[worker][key] += 1
      self.recent_novel.append(is_novel)

      global_unique = float(len(self.global_counts))
      episode_unique = float(len(self.episode_counts[worker]))

      ep_steps = float(sum(self.episode_counts[worker].values()))
      revisit_ratio = 0.0 if ep_steps <= 0 else 1.0 - (episode_unique / ep_steps)
      recent_novel_rate = float(np.mean(self.recent_novel)) if self.recent_novel else 0.0

    if action is not None:
      try:
        a = int(np.asarray(action).reshape(-1)[0])
        self.action_sets[worker].add(a)
        self.action_hist[worker].append(a)
        if len(self.action_hist[worker]) == 2:
          self.action_bigram_sets[worker].add(tuple(self.action_hist[worker]))
      except Exception:
        pass

    unique_actions = float(len(self.action_sets[worker]))
    unique_action_bigrams = float(len(self.action_bigram_sets[worker]))

    return {
        'is_novel_state': float(is_novel),
        'global_unique_states': global_unique,
        'episode_unique_states': episode_unique,
        'episode_revisit_ratio': float(revisit_ratio),
        'recent_novel_rate': float(recent_novel_rate),
        'unique_actions': unique_actions,
        'unique_action_bigrams': unique_action_bigrams,
    }


class EvalCollector:

  def __init__(self, split_name, coverage_stride=8, coverage_recent_window=200):
    self.split_name = split_name
    self.coverage = CoverageTracker(
        stride=coverage_stride,
        recent_window=coverage_recent_window,
    )
    self.total_steps = 0
    self.global_episode_id = 0
    self.worker_episode_ids = {}
    self.ep_buffers = {}
    self.step_rows = []

  def on_step(self, tran, worker):
    if tran['is_first']:
      self.coverage.reset_episode(worker)
      self.global_episode_id += 1
      self.worker_episode_ids[worker] = self.global_episode_id
      self.ep_buffers[worker] = {
          'episode_id': self.global_episode_id,
          'score': 0.0,
          'training_reward': 0.0,
          'tester_bonus': 0.0,
          'rnd_intrinsic_reward': 0.0,
          'length': 0,
          'fault_episode': 0,
          'fault_exists_episode': 0,
          'fault_applied_count': 0,
          'fault_manifested_count': 0,
          'fault_trigger_context_count': 0,
          'lowlevel_trigger_context_count': 0,
          'semantic_fault_applied_count': 0,
          'semantic_trigger_context_count': 0,
          'fault_manifest_prob_max': 0.0,
          'max_ref_bug_score': -1e9,
          'max_ref_bug_kl': -1e9,
          'max_fault_score': -1e9,
          'max_latent_kl_surprise': -1e9,
      }
      for key in SEMANTIC_CONTEXT_KEYS:
        self.ep_buffers[worker][f'semantic_ctx_{key}_count'] = 0

    episode_id = self.worker_episode_ids[worker]
    buf = self.ep_buffers[worker]

    training_reward = _scalar(tran.get('reward', 0.0))
    task_reward = _scalar(tran.get('log/task_reward_raw', training_reward))
    tester_bonus = _scalar(tran.get('log/tester_bonus', 0.0))
    rnd_intrinsic_reward = _scalar(tran.get('log/rnd_intrinsic_reward', 0.0))
    fault_episode = int(_scalar(tran.get('log/fault_episode', 0.0)) > 0.5)
    fault_applied = int(_scalar(tran.get('log/fault_applied', 0.0)) > 0.5)
    fault_exists_episode = int(_scalar(
        tran.get('log/fault_exists_episode', fault_episode)) > 0.5)
    lowlevel_trigger_context = int(_scalar(
        tran.get('log/lowlevel_trigger_context', 0.0)) > 0.5)
    fault_manifested = int(_scalar(
        tran.get('log/fault_manifested', fault_applied)) > 0.5)
    fault_manifest_prob = _scalar(tran.get('log/fault_manifest_prob', 0.0))
    fault_profile_id = int(_scalar(tran.get('log/fault_profile_id', 0.0)))
    semantic_fault_episode = int(
        _scalar(tran.get('log/semantic_fault_episode', 0.0)) > 0.5)
    semantic_fault_applied = int(
        _scalar(tran.get('log/semantic_fault_applied', 0.0)) > 0.5)
    semantic_trigger_context = int(
        _scalar(tran.get('log/semantic_trigger_context', 0.0)) > 0.5)
    semantic_context_flags = {
        'upgrade_collect': int(_scalar(
            tran.get('log/semantic_ctx_upgrade_collect', 0.0)) > 0.5),
        'retry_craft': int(_scalar(
            tran.get('log/semantic_ctx_retry_craft', 0.0)) > 0.5),
        'relocate_station': int(_scalar(
            tran.get('log/semantic_ctx_relocate_station', 0.0)) > 0.5),
        'valid_progress': int(_scalar(
            tran.get('log/semantic_ctx_valid_progress', 0.0)) > 0.5),
        'station_reuse': int(_scalar(
            tran.get('log/semantic_ctx_station_reuse', 0.0)) > 0.5),
        'delayed_after_use': int(_scalar(
            tran.get('log/semantic_ctx_delayed_after_use', 0.0)) > 0.5),
    }
    fault_trigger_context = int(_scalar(tran.get(
        'log/fault_trigger_context',
        max(lowlevel_trigger_context, semantic_trigger_context))) > 0.5)
    fault_episode = max(fault_episode, semantic_fault_episode, fault_exists_episode)
    fault_applied = max(fault_applied, semantic_fault_applied, fault_manifested)
    fault_manifested = max(fault_manifested, fault_applied)
    fault_score = _scalar(tran.get(
        'log/ref_fault_score', tran.get('log/ref_bug_score', 0.0)))
    latent_kl = _scalar(tran.get(
        'log/ref_latent_kl_surprise', tran.get('log/ref_bug_kl', 0.0)))
    reward_error = _scalar(tran.get(
        'log/ref_reward_prediction_error',
        tran.get('log/ref_bug_reward_err', 0.0)))
    reward_pred = _scalar(tran.get('log/ref_reward_pred', 0.0))
    ref_bug_score = fault_score
    ref_bug_kl = latent_kl
    action = tran.get('action', None)

    cov = self.coverage.step(worker, tran.get('image'), action)

    self.total_steps += 1
    buf['score'] += task_reward
    buf['training_reward'] += training_reward
    buf['tester_bonus'] += tester_bonus
    buf['rnd_intrinsic_reward'] += rnd_intrinsic_reward
    buf['length'] += 1
    buf['fault_episode'] = max(buf['fault_episode'], fault_episode)
    buf['fault_exists_episode'] = max(
        buf['fault_exists_episode'], fault_exists_episode)
    buf['fault_applied_count'] += fault_applied
    buf['fault_manifested_count'] += fault_manifested
    buf['fault_trigger_context_count'] += fault_trigger_context
    buf['lowlevel_trigger_context_count'] += lowlevel_trigger_context
    buf['semantic_fault_applied_count'] += semantic_fault_applied
    buf['semantic_trigger_context_count'] += semantic_trigger_context
    buf['fault_manifest_prob_max'] = max(
        buf['fault_manifest_prob_max'], float(fault_manifest_prob))
    for key, value in semantic_context_flags.items():
      buf[f'semantic_ctx_{key}_count'] += int(value)
    buf['max_ref_bug_score'] = max(buf['max_ref_bug_score'], ref_bug_score)
    buf['max_ref_bug_kl'] = max(buf['max_ref_bug_kl'], ref_bug_kl)
    buf['max_fault_score'] = max(buf['max_fault_score'], fault_score)
    buf['max_latent_kl_surprise'] = max(
        buf['max_latent_kl_surprise'], latent_kl)

    row = {
        'split': self.split_name,
        'transition_index': self.total_steps,
        'episode_id': episode_id,
        'episode_step': buf['length'],
        'reward': float(training_reward),
        'training_reward': float(training_reward),
        'task_reward': float(task_reward),
        'tester_bonus': float(tester_bonus),
        'rnd_intrinsic_reward': float(rnd_intrinsic_reward),
        'fault_episode': int(fault_episode),
        'fault_exists_episode': int(fault_exists_episode),
        'fault_applied': int(fault_applied),
        'fault_manifested': int(fault_manifested),
        'fault_trigger_context': int(fault_trigger_context),
        'lowlevel_trigger_context': int(lowlevel_trigger_context),
        'fault_manifest_prob': float(fault_manifest_prob),
        'fault_profile_id': int(fault_profile_id),
        'semantic_fault_episode': int(semantic_fault_episode),
        'semantic_fault_applied': int(semantic_fault_applied),
        'semantic_trigger_context': int(semantic_trigger_context),
        'fault_score': float(fault_score),
        'latent_kl_surprise': float(latent_kl),
        'reward_prediction_error': float(reward_error),
        'reward_pred': float(reward_pred),
        'ref_bug_score': float(ref_bug_score),
        'ref_bug_kl': float(ref_bug_kl),
        'is_last': int(bool(tran['is_last'])),
        'is_terminal': int(bool(tran['is_terminal'])),
        'is_novel_state': cov['is_novel_state'],
        'global_unique_states': cov['global_unique_states'],
        'episode_unique_states': cov['episode_unique_states'],
        'episode_revisit_ratio': cov['episode_revisit_ratio'],
        'recent_novel_rate': cov['recent_novel_rate'],
        'unique_actions': cov['unique_actions'],
        'unique_action_bigrams': cov['unique_action_bigrams'],
    }
    for key, value in semantic_context_flags.items():
      row[f'semantic_ctx_{key}'] = int(value)
    self.step_rows.append(row)

  def to_dataframes(self):
    step_df = pd.DataFrame(self.step_rows)

    if len(step_df) == 0:
      ep_df = pd.DataFrame(columns=[
          'split', 'episode_id', 'episode_score', 'length', 'fault_episode',
          'fault_exists_episode', 'episode_training_reward',
          'episode_tester_bonus', 'episode_rnd_intrinsic_reward',
          'fault_applied_count', 'fault_manifested_count',
          'fault_trigger_context_count', 'lowlevel_trigger_context_count',
          'semantic_fault_applied_count', 'semantic_trigger_context_count',
          'fault_manifest_prob_max', 'fault_profile_id',
          'max_ref_bug_score', 'max_ref_bug_kl',
          'max_fault_score', 'max_latent_kl_surprise',
      ])
      return step_df, ep_df

    agg = {
        'split': 'first',
        'task_reward': 'sum',
        'training_reward': 'sum',
        'tester_bonus': 'sum',
        'rnd_intrinsic_reward': 'sum',
        'episode_step': 'max',
        'fault_episode': 'max',
        'fault_exists_episode': 'max',
        'fault_applied': 'sum',
        'fault_manifested': 'sum',
        'fault_trigger_context': 'sum',
        'lowlevel_trigger_context': 'sum',
        'fault_manifest_prob': 'max',
        'fault_profile_id': 'max',
        'semantic_fault_applied': 'sum',
        'semantic_trigger_context': 'sum',
        'fault_score': 'max',
        'latent_kl_surprise': 'max',
        'ref_bug_score': 'max',
        'ref_bug_kl': 'max',
        'episode_unique_states': 'max',
        'episode_revisit_ratio': 'last',
        'recent_novel_rate': 'last',
        'unique_actions': 'max',
        'unique_action_bigrams': 'max',
    }
    for key in SEMANTIC_CONTEXT_KEYS:
      col = f'semantic_ctx_{key}'
      if col in step_df:
        agg[col] = 'sum'

    ep_df = (
        step_df.groupby('episode_id', as_index=False)
        .agg(agg)
        .rename(columns={
            'task_reward': 'episode_score',
            'training_reward': 'episode_training_reward',
            'tester_bonus': 'episode_tester_bonus',
            'rnd_intrinsic_reward': 'episode_rnd_intrinsic_reward',
            'episode_step': 'length',
            'fault_applied': 'fault_applied_count',
            'fault_manifested': 'fault_manifested_count',
            'fault_trigger_context': 'fault_trigger_context_count',
            'lowlevel_trigger_context': 'lowlevel_trigger_context_count',
            'fault_manifest_prob': 'fault_manifest_prob_max',
            'semantic_fault_applied': 'semantic_fault_applied_count',
            'semantic_trigger_context': 'semantic_trigger_context_count',
            'fault_score': 'max_fault_score',
            'latent_kl_surprise': 'max_latent_kl_surprise',
            'ref_bug_score': 'max_ref_bug_score',
            'ref_bug_kl': 'max_ref_bug_kl',
        })
    )
    ep_df['episode_task_score'] = ep_df['episode_score']
    for key in SEMANTIC_CONTEXT_KEYS:
      col = f'semantic_ctx_{key}'
      if col in ep_df:
        ep_df = ep_df.rename(columns={col: f'{col}_count'})
    return step_df, ep_df


def configure_split_env(split_name, outdir):
  trace_path = outdir / f"trace_{split_name}.jsonl"
  if trace_path.exists():
    trace_path.unlink()

  os.environ['CRAFTER_TRACE_PATH'] = str(trace_path)
  os.environ['CRAFTER_TESTER_REWARD'] = os.getenv(
      'TESTER_EVAL_CRAFTER_TESTER_REWARD', '0')
  os.environ['CRAFTER_USE_RND'] = os.getenv('TESTER_EVAL_CRAFTER_USE_RND', '0')
  os.environ['CRAFTER_RND_UPDATE'] = os.getenv(
      'TESTER_EVAL_CRAFTER_RND_UPDATE', '0')
  if os.getenv('TESTER_EVAL_FAULT_FREQ_TIER'):
    os.environ['CRAFTER_FAULT_FREQ_TIER'] = os.getenv(
        'TESTER_EVAL_FAULT_FREQ_TIER')
  os.environ['CRAFTER_SEMANTIC_FAULT_SAMPLER'] = '0'
  os.environ['CRAFTER_SEMANTIC_FAULT_EP_PROB'] = '0.0'
  os.environ['CRAFTER_SEMANTIC_FAULT_MANIFEST_PROB'] = '0.0'
  os.environ.pop('CRAFTER_SEMANTIC_FAULT_PROFILE', None)
  os.environ.pop('CRAFTER_SEMANTIC_SUBTYPES', None)

  if split_name == 'clean':
    os.environ['CRAFTER_FAULT_SAMPLER'] = '0'
    os.environ['CRAFTER_FAULT'] = '0'
    os.environ['CRAFTER_FAULT_PROFILE'] = os.getenv(
        'TESTER_EVAL_CLEAN_FAULT_PROFILE', 'benchmark_train')

  elif split_name == 'seen':
    os.environ['CRAFTER_FAULT_SAMPLER'] = '1'
    os.environ['CRAFTER_FAULT'] = '0'
    os.environ['CRAFTER_FAULT_PROFILE'] = os.getenv(
        'TESTER_EVAL_SEEN_FAULT_PROFILE', 'benchmark_seen')

  elif split_name == 'holdout':
    os.environ['CRAFTER_FAULT_SAMPLER'] = '1'
    os.environ['CRAFTER_FAULT'] = '0'
    os.environ['CRAFTER_FAULT_PROFILE'] = os.getenv(
        'TESTER_EVAL_HOLDOUT_FAULT_PROFILE', 'benchmark_holdout')

  elif split_name == 'semantic_holdout':
    os.environ['CRAFTER_FAULT_SAMPLER'] = '0'
    os.environ['CRAFTER_FAULT'] = '0'
    os.environ['CRAFTER_FAULT_PROFILE'] = os.getenv(
        'TESTER_EVAL_CLEAN_FAULT_PROFILE', 'benchmark_train')
    os.environ['CRAFTER_SEMANTIC_FAULT_SAMPLER'] = '1'
    os.environ['CRAFTER_SEMANTIC_FAULT_PROFILE'] = os.getenv(
        'TESTER_EVAL_SEMANTIC_FAULT_PROFILE', 'eval_holdout')
    os.environ['CRAFTER_SEMANTIC_FAULT_EP_PROB'] = os.getenv(
        'TESTER_EVAL_SEMANTIC_FAULT_EP_PROB', '0.5')
    os.environ['CRAFTER_SEMANTIC_FAULT_MANIFEST_PROB'] = os.getenv(
        'TESTER_EVAL_SEMANTIC_FAULT_MANIFEST_PROB', '1.0')
    os.environ['CRAFTER_SEMANTIC_SUBTYPES'] = os.getenv(
        'TESTER_EVAL_SEMANTIC_SUBTYPES',
        DEFAULT_SEMANTIC_HOLDOUT_SUBTYPES)

  else:
    raise ValueError(split_name)

  return trace_path


def parse_eval_splits():
  raw = os.getenv('TESTER_EVAL_SPLITS', ','.join(DEFAULT_EVAL_SPLITS))
  splits = [x.strip() for x in raw.split(',') if x.strip()]
  if not splits:
    splits = list(DEFAULT_EVAL_SPLITS)
  if 'clean' not in splits:
    splits.insert(0, 'clean')
  elif splits[0] != 'clean':
    splits = ['clean'] + [x for x in splits if x != 'clean']
  return splits


def split_eval_steps(split_name):
  return int(os.getenv(
      f'TESTER_EVAL_{split_name.upper()}_STEPS',
      os.getenv('TESTER_EVAL_STEPS', '50000')))


def maybe_load_existing_split(split_name, outdir, eval_steps):
  if os.getenv('TESTER_EVAL_RESUME_EXISTING', '0').lower() not in (
      '1', 'true', 'yes'):
    return None

  step_path = outdir / f"{split_name}_steps.jsonl"
  ep_path = outdir / f"{split_name}_episodes.jsonl"
  trace_path = outdir / f"trace_{split_name}.jsonl"
  if not step_path.exists() or not ep_path.exists():
    return None

  try:
    step_df = pd.read_json(step_path, orient='records', lines=True)
    ep_df = pd.read_json(ep_path, orient='records', lines=True)
  except ValueError:
    return None

  if len(step_df) < eval_steps:
    print(
        f"Existing split {split_name} is incomplete "
        f"({len(step_df)}/{eval_steps}); rerunning.")
    return None

  print(
      f"Reusing completed split {split_name}: "
      f"{len(step_df)} steps, {len(ep_df)} episodes.")
  return step_df, ep_df, trace_path


def save_jsonl(df, path):
  if len(df) == 0:
    path.write_text("", encoding="utf-8")
    return
  df.to_json(path, orient='records', lines=True, force_ascii=False)


def compute_threshold(clean_step_df, quantile=0.99):
  score_key = 'fault_score' if 'fault_score' in clean_step_df else 'ref_bug_score'
  scores = pd.to_numeric(clean_step_df[score_key], errors='coerce').dropna()
  if len(scores) == 0:
    return 0.0
  return float(np.quantile(scores.values, quantile))


def compute_detection_metrics(step_df, ep_df, threshold):
  result = {}

  if len(step_df) == 0:
    return result

  y_true = pd.to_numeric(step_df['fault_applied'], errors='coerce').fillna(0).astype(int).values
  score_key = 'fault_score' if 'fault_score' in step_df else 'ref_bug_score'
  y_score = pd.to_numeric(step_df[score_key], errors='coerce').fillna(0.0).values
  y_pred = (y_score >= threshold).astype(int)

  result['threshold'] = float(threshold)
  result['step_fault_applied_rate'] = float(np.mean(y_true))
  result['step_alarm_rate'] = float(np.mean(y_pred))

  label_cols = (
      'fault_manifested',
      'fault_trigger_context',
      'lowlevel_trigger_context',
      'semantic_trigger_context',
      'fault_exists_episode',
  )
  for col in label_cols:
    if col not in step_df:
      continue
    label = pd.to_numeric(
        step_df[col], errors='coerce').fillna(0).astype(int).values
    result[f'{col}_step_rate'] = float(np.mean(label))
    result[f'{col}_step_count'] = int(np.sum(label))
    if len(np.unique(label)) > 1:
      if _HAVE_SKLEARN:
        result[f'{col}_auroc'] = float(roc_auc_score(label, y_score))
        result[f'{col}_auprc'] = float(average_precision_score(label, y_score))
      else:
        result[f'{col}_auroc'] = _fallback_auroc(label, y_score)
        result[f'{col}_auprc'] = _fallback_average_precision(label, y_score)
      tp = int(((label == 1) & (y_pred == 1)).sum())
      fp = int(((label == 0) & (y_pred == 1)).sum())
      fn = int(((label == 1) & (y_pred == 0)).sum())
      prec = tp / max(tp + fp, 1)
      rec = tp / max(tp + fn, 1)
      result[f'{col}_threshold_precision'] = float(prec)
      result[f'{col}_threshold_recall'] = float(rec)
    else:
      result[f'{col}_auroc'] = np.nan
      result[f'{col}_auprc'] = np.nan
      result[f'{col}_threshold_precision'] = np.nan
      result[f'{col}_threshold_recall'] = np.nan

  if y_true.sum() > 0:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    result['step_precision'] = float(prec)
    result['step_recall'] = float(rec)
    result['step_f1'] = float(f1)
  else:
    result['step_precision'] = np.nan
    result['step_recall'] = np.nan
    result['step_f1'] = np.nan

  if len(np.unique(y_true)) > 1:
    if _HAVE_SKLEARN:
      result['step_auroc'] = float(roc_auc_score(y_true, y_score))
      result['step_auprc'] = float(average_precision_score(y_true, y_score))
    else:
      result['step_auroc'] = _fallback_auroc(y_true, y_score)
      result['step_auprc'] = _fallback_average_precision(y_true, y_score)
  else:
    result['step_auroc'] = np.nan
    result['step_auprc'] = np.nan

  # episode-level detection
  detected_flags = []
  time_to_detect = []
  false_alarm_clean = []

  for ep_id, grp in step_df.groupby('episode_id'):
    grp = grp.sort_values('episode_step')
    fault_ep = int(grp['fault_episode'].max())
    alarms = grp[grp[score_key] >= threshold]
    faults = grp[grp['fault_applied'] > 0]

    if fault_ep == 1:
      if len(faults) == 0:
        detected_flags.append(0)
        continue

      first_fault_step = int(faults['episode_step'].iloc[0])
      alarms_after_fault = alarms[alarms['episode_step'] >= first_fault_step]

      if len(alarms_after_fault) > 0:
        detected_flags.append(1)
        ttd = int(alarms_after_fault['episode_step'].iloc[0]) - first_fault_step
        time_to_detect.append(ttd)
      else:
        detected_flags.append(0)
    else:
      false_alarm_clean.append(1 if len(alarms) > 0 else 0)

  if len(detected_flags) > 0:
    result['episode_detection_rate'] = float(np.mean(detected_flags))
  else:
    result['episode_detection_rate'] = np.nan

  if len(time_to_detect) > 0:
    result['mean_time_to_detect'] = float(np.mean(time_to_detect))
  else:
    result['mean_time_to_detect'] = np.nan

  if len(false_alarm_clean) > 0:
    result['clean_false_alarm_episode_rate'] = float(np.mean(false_alarm_clean))
  else:
    result['clean_false_alarm_episode_rate'] = np.nan

  # policy / coverage summary
  if len(ep_df) > 0:
    result['episode_score_mean'] = float(ep_df['episode_score'].mean())
    result['episode_task_score_mean'] = float(ep_df['episode_score'].mean())
    if 'episode_training_reward' in ep_df:
      result['episode_training_reward_mean'] = float(
          ep_df['episode_training_reward'].mean())
    if 'episode_tester_bonus' in ep_df:
      result['episode_tester_bonus_mean'] = float(
          ep_df['episode_tester_bonus'].mean())
    if 'episode_rnd_intrinsic_reward' in ep_df:
      result['episode_rnd_intrinsic_reward_mean'] = float(
          ep_df['episode_rnd_intrinsic_reward'].mean())
    result['episode_length_mean'] = float(ep_df['length'].mean())

    clean_eps = ep_df[ep_df['fault_episode'] == 0]
    fault_eps = ep_df[ep_df['fault_episode'] == 1]

    result['clean_score_mean'] = float(clean_eps['episode_score'].mean()) if len(clean_eps) else np.nan
    result['fault_score_mean'] = float(fault_eps['episode_score'].mean()) if len(fault_eps) else np.nan
    result['clean_task_score_mean'] = result['clean_score_mean']
    result['fault_task_score_mean'] = result['fault_score_mean']
    if 'episode_training_reward' in ep_df:
      result['clean_training_reward_mean'] = (
          float(clean_eps['episode_training_reward'].mean()) if len(clean_eps) else np.nan)
      result['fault_training_reward_mean'] = (
          float(fault_eps['episode_training_reward'].mean()) if len(fault_eps) else np.nan)
    for col in (
        'fault_exists_episode',
        'fault_manifested_count',
        'fault_trigger_context_count',
        'lowlevel_trigger_context_count',
        'fault_manifest_prob_max',
    ):
      if col in ep_df:
        values = pd.to_numeric(ep_df[col], errors='coerce').fillna(0.0)
        result[f'{col}_episode_mean'] = float(values.mean())
        if col.endswith('_count'):
          result[f'{col}_episode_rate'] = float((values > 0).mean())
    if 'max_fault_score' in ep_df:
      result['max_fault_score_mean'] = float(ep_df['max_fault_score'].mean())
      result['max_latent_kl_surprise_mean'] = float(
          ep_df['max_latent_kl_surprise'].mean())
    result['episode_unique_states_mean'] = float(ep_df['episode_unique_states'].mean())
    result['episode_revisit_ratio_mean'] = float(ep_df['episode_revisit_ratio'].mean())
    result['recent_novel_rate_mean'] = float(ep_df['recent_novel_rate'].mean())
    result['unique_action_bigrams_mean'] = float(ep_df['unique_action_bigrams'].mean())

    if 'semantic_trigger_context_count' in ep_df:
      context_eps = ep_df[ep_df['semantic_trigger_context_count'] > 0]
      semantic_fault_eps = ep_df[ep_df.get(
          'semantic_fault_applied_count', pd.Series(0, index=ep_df.index)) > 0]
      result['semantic_context_episode_rate'] = float(
          len(context_eps) / max(len(ep_df), 1))
      result['semantic_fault_episode_rate'] = float(
          len(semantic_fault_eps) / max(len(ep_df), 1))
      result['semantic_fault_given_context_episode_rate'] = (
          float(len(semantic_fault_eps[
              semantic_fault_eps['semantic_trigger_context_count'] > 0]) /
              max(len(context_eps), 1))
          if len(context_eps) else np.nan)
      result['semantic_context_step_mean'] = float(
          ep_df['semantic_trigger_context_count'].mean())
      for key in SEMANTIC_CONTEXT_KEYS:
        col = f'semantic_ctx_{key}_count'
        if col in ep_df:
          result[f'{col}_episode_rate'] = float(
              (ep_df[col] > 0).mean())

  if 'semantic_trigger_context' in step_df:
    semantic_context = pd.to_numeric(
        step_df['semantic_trigger_context'], errors='coerce').fillna(0).astype(int)
    result['semantic_context_step_rate'] = float(semantic_context.mean())
    result['semantic_context_step_count'] = int(semantic_context.sum())
    if 'semantic_fault_applied' in step_df:
      semantic_fault = pd.to_numeric(
          step_df['semantic_fault_applied'], errors='coerce').fillna(0).astype(int)
      result['semantic_fault_step_rate'] = float(semantic_fault.mean())
      result['semantic_fault_step_count'] = int(semantic_fault.sum())
      result['semantic_fault_given_context_step_rate'] = (
          float(semantic_fault[semantic_context > 0].sum() /
                max(int(semantic_context.sum()), 1))
          if int(semantic_context.sum()) > 0 else np.nan)
    for key in SEMANTIC_CONTEXT_KEYS:
      col = f'semantic_ctx_{key}'
      if col in step_df:
        vals = pd.to_numeric(
            step_df[col], errors='coerce').fillna(0).astype(int)
        result[f'{col}_step_rate'] = float(vals.mean())
        result[f'{col}_step_count'] = int(vals.sum())

  return result


def trace_subtype_summary(trace_path):
  if not trace_path.exists():
    return {}
  rows = []
  with trace_path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  if not rows:
    return {}

  trace_df = pd.DataFrame(rows)
  if 'fault_type' not in trace_df.columns:
    return {}

  def series(name, fallback=None, default=0.0):
    if name in trace_df.columns:
      return trace_df[name]
    if fallback and fallback in trace_df.columns:
      return trace_df[fallback]
    return pd.Series(default, index=trace_df.index)

  manifested = pd.to_numeric(
      series('fault_manifested', fallback='fault_applied'),
      errors='coerce').fillna(0).astype(int)
  trigger = pd.to_numeric(
      series('fault_trigger_context'),
      errors='coerce').fillna(0).astype(int)
  exists_episode = pd.to_numeric(
      series('fault_exists_episode', fallback='fault_episode'),
      errors='coerce').fillna(0).astype(int)
  applied = trace_df[manifested == 1]
  subtype_counts = applied['fault_type'].value_counts().to_dict()
  family_counts = applied['fault_family'].value_counts().to_dict()
  manifest_prob = pd.to_numeric(
      series('fault_manifest_prob'),
      errors='coerce').fillna(0.0)
  fault_episode = pd.to_numeric(
      series('fault_episode'), errors='coerce').fillna(0).astype(int)
  fault_applied = pd.to_numeric(
      series('fault_applied'), errors='coerce').fillna(0).astype(int)

  return {
      'trace_rows': int(len(trace_df)),
      'trace_fault_episode_rate': float(fault_episode.mean()),
      'trace_fault_applied_rate': float(fault_applied.mean()),
      'trace_fault_exists_episode_rate': float(exists_episode.mean()),
      'trace_fault_trigger_context_rate': float(trigger.mean()),
      'trace_fault_manifested_rate': float(manifested.mean()),
      'trace_fault_manifest_prob_mean': float(manifest_prob.mean()),
      'trace_fault_type_counts': subtype_counts,
      'trace_fault_family_counts': family_counts,
  }


def run_split(split_name, make_agent, make_env, args, tester_ckpt, ref_ckpt, outdir):
  from dreamerv3 import fault_score as faultlib

  eval_steps = split_eval_steps(split_name)
  existing = maybe_load_existing_split(split_name, outdir, eval_steps)
  if existing is not None:
    return existing

  trace_path = configure_split_env(split_name, outdir)

  tester_agent = make_agent()
  ref_agent = make_agent()

  load_regex = args.from_checkpoint_regex if hasattr(args, 'from_checkpoint_regex') else None

  if load_regex is None:
    elements.checkpoint.load(tester_ckpt, dict(agent=tester_agent.load))
    elements.checkpoint.load(ref_ckpt, dict(agent=ref_agent.load))
  else:
    elements.checkpoint.load(tester_ckpt, dict(agent=bind(tester_agent.load, regex=load_regex)))
    elements.checkpoint.load(ref_ckpt, dict(agent=bind(ref_agent.load, regex=load_regex)))

  def init_dual_policy(batch_size):
    return {
        'tester': tester_agent.init_policy(batch_size),
        'ref': ref_agent.init_policy(batch_size),
    }

  def dual_policy(carry, obs, mode='eval'):
    tester_carry, acts, tester_outs = tester_agent.policy(
        carry['tester'], obs, mode='eval')
    ref_carry, _, ref_outs = ref_agent.policy(
        carry['ref'], obs, mode='eval')
    ref_carry = (*ref_carry[:-1], tester_carry[-1])

    outs = dict(tester_outs)
    faultlib.add_reference_outputs(outs, ref_outs)
    return {'tester': tester_carry, 'ref': ref_carry}, acts, outs

  coverage_stride = int(os.getenv('TESTER_NOVELTY_STRIDE', '8'))
  coverage_recent_window = int(os.getenv('TESTER_COVERAGE_RECENT_WINDOW', '200'))

  collector = EvalCollector(
      split_name=split_name,
      coverage_stride=coverage_stride,
      coverage_recent_window=coverage_recent_window,
  )

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(collector.on_step)

  driver.reset(init_dual_policy)

  while collector.total_steps < eval_steps:
    driver(dual_policy, steps=10)

  step_df, ep_df = collector.to_dataframes()
  save_jsonl(step_df, outdir / f"{split_name}_steps.jsonl")
  save_jsonl(ep_df, outdir / f"{split_name}_episodes.jsonl")

  return step_df, ep_df, trace_path


def tester_eval(make_agent, make_env, make_logger, args):
  del make_logger  # This evaluation script writes files directly.

  outdir = Path(args.logdir) / "tester_eval"
  outdir.mkdir(parents=True, exist_ok=True)

  tester_ckpt = os.getenv('TESTER_EVAL_CHECKPOINT', args.from_checkpoint)
  ref_ckpt = os.getenv('TESTER_REF_CHECKPOINT', args.from_checkpoint)

  if not tester_ckpt:
    raise ValueError('TESTER_EVAL_CHECKPOINT or --run.from_checkpoint is required.')

  print("Tester eval output dir:", outdir)
  print("Tester checkpoint:", tester_ckpt)
  print("Reference checkpoint:", ref_ckpt)

  splits = parse_eval_splits()
  print("Evaluation splits:", ','.join(splits))

  # 1) clean
  clean_steps, clean_eps, clean_trace = run_split(
      'clean', make_agent, make_env, args, tester_ckpt, ref_ckpt, outdir)

  threshold_q = float(os.getenv('TESTER_EVAL_THRESHOLD_Q', '0.99'))
  threshold = compute_threshold(clean_steps, quantile=threshold_q)
  print(f"Clean anomaly threshold (q={threshold_q}): {threshold:.6f}")

  summaries = {}

  summaries['clean'] = compute_detection_metrics(clean_steps, clean_eps, threshold)
  summaries['clean'].update(trace_subtype_summary(clean_trace))

  for split_name in splits:
    if split_name == 'clean':
      continue
    step_df, ep_df, trace_path = run_split(
        split_name, make_agent, make_env, args, tester_ckpt, ref_ckpt, outdir)
    summaries[split_name] = compute_detection_metrics(step_df, ep_df, threshold)
    summaries[split_name].update(trace_subtype_summary(trace_path))

  summaries['threshold_quantile'] = threshold_q
  summaries['threshold_value'] = threshold

  with (outdir / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2, ensure_ascii=False)

  print("\n=== Evaluation Summary ===")
  print(json.dumps(summaries, indent=2, ensure_ascii=False))

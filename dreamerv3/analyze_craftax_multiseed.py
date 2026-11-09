#!/usr/bin/env python3
"""Analyze Craftax multi-seed fault-seeking runs.

The queue writes one directory per seed and variant. This script reads those
JSONL traces directly, aggregates metrics across seeds, and produces compact
tables and figures for the task-performance versus bug-seeking tradeoff.
"""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_BUG_TYPES = {
    "seen": {
        "delay_after_success",
        "remap_after_success_switch",
        "sticky_after_repeat_switch",
        "reward_delay_on_positive",
        "reward_scale_half_on_positive_switch",
        "tool_collect_desync_on_upgrade",
        "craft_result_missing_on_retry",
        "station_place_ghost_on_relocate",
    },
    "holdout": {
        "revisit_action_delay",
        "delayed_switch_failure",
        "remap_after_repeat_switch",
        "reward_zero_after_repeat_switch",
        "reward_delay_after_two_rewards",
        "achievement_unlock_missing_after_valid_progress",
        "delayed_inventory_desync_after_station_use",
    },
}
EXPECTED_BUG_TYPES["sparse"] = EXPECTED_BUG_TYPES["holdout"]


METRICS = [
    "steps",
    "reward_mean",
    "augmented_reward_mean",
    "task_score_mean",
    "episode_score_mean",
    "episode_length_mean",
    "fault_applied_rate",
    "fault_trigger_rate",
    "semantic_fault_rate",
    "fault_episode_rate",
    "fault_score_mean",
    "fault_score_p95",
    "fault_score_delta_mean",
    "latent_kl_mean",
    "reward_error_mean",
    "fault_reward_bonus_mean",
    "fault_reward_score_mean",
    "novelty_bound_score_mean",
    "novelty_bound_trigger_rate",
    "constraint_active_rate",
    "constraint_lambda_mean",
    "constraint_lambda_max",
    "constraint_violation_mean",
    "task_constraint_feasible_rate",
    "bug_event_count",
    "bug_events_per_10k",
    "bug_prevalence",
    "bug_found",
    "unique_bug_types",
    "expected_bug_types",
    "bug_type_coverage_fraction",
    "bug_discovery_auc",
    "bug_discovery_auc_norm",
    "unexpected_bug_types",
    "time_to_first_bug_steps",
    "episodes_to_first_bug",
    "semantic_context_coverage",
    "semantic_context_coverage_per_1k",
    "unique_suspicious_context_count",
    "unique_suspicious_context_per_1k",
    "suspicious_context_rate",
    "repeated_suspicious_context_ratio",
    "threshold_bug_recall",
    "threshold_false_positive_rate",
    "clean_false_positive_rate",
    "manifestation_given_trigger",
    "bug_per_task_score",
    "auroc",
    "auroc_above_chance",
    "auprc",
    "auprc_lift",
    "precision_at_top1pct",
    "precision_at_top5pct",
    "fault_score_bug_mean",
    "fault_score_normal_mean",
    "fault_score_bug_normal_gap",
    "unique_actions",
]

PLOT_METRICS = [
    ("episode_score_mean", "Episode score"),
    ("fault_applied_rate", "Bug manifestation rate"),
    ("auroc", "Fault-score AUROC"),
    ("auprc", "Fault-score AUPRC"),
]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", required=True, help="Craftax multi-seed run root.")
  parser.add_argument("--outdir", default="", help="Defaults to <root>/analysis.")
  parser.add_argument("--baseline", default="taskonly")
  parser.add_argument("--formats", default="png,pdf")
  parser.add_argument(
      "--error-bars", choices=("sem", "std", "none"), default="sem",
      help="Error bars for aggregate plots. Use std for visible seed variance.")
  parser.add_argument(
      "--seeds", type=int, nargs="*", default=None,
      help="Only include these seed ids, for balanced partial-run analyses.")
  parser.add_argument(
      "--eval-only", action="store_true",
      help="Skip large training traces and analyze evaluation directories only.")
  return parser.parse_args()


def read_jsonl(path):
  rows = []
  if not path.exists():
    return rows
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      if line.strip():
        rows.append(json.loads(line))
  return rows


def iter_jsonl(path):
  if not path.exists():
    return
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      if line.strip():
        yield json.loads(line)


def write_csv(path, rows, fieldnames):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def write_json(path, obj):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def safe_float(value, default=0.0):
  try:
    value = float(value)
  except Exception:
    return default
  if not math.isfinite(value):
    return default
  return value


def error_key(metric, mode):
  if mode == "none":
    return None
  if metric.endswith("_mean_mean"):
    return metric[:-len("_mean_mean")] + f"_mean_{mode}"
  if metric.endswith("_mean"):
    return metric[:-len("_mean")] + f"_{mode}"
  return f"{metric}_{mode}"


def error_values(rows, metric, mode):
  key = error_key(metric, mode)
  if not key:
    return None
  return [safe_float(row.get(key), 0.0) for row in rows]


def value(row, *keys, default=0.0):
  for key in keys:
    if key in row:
      return safe_float(row.get(key), default)
  return default


def optional_value(row, *keys):
  for key in keys:
    if key in row:
      return safe_float(row.get(key), float("nan"))
  return None


def values(rows, *keys):
  return np.asarray([value(row, *keys) for row in rows], np.float64)


def binary(rows, *keys):
  return np.asarray([1 if value(row, *keys) > 0.5 else 0 for row in rows], np.int32)


def has_any(rows, *keys):
  return any(any(key in row for key in keys) for row in rows)


def optional_values(rows, *keys):
  if not has_any(rows, *keys):
    return np.asarray([], np.float64)
  return values(rows, *keys)


def optional_binary(rows, *keys):
  if not has_any(rows, *keys):
    return np.asarray([], np.int32)
  return binary(rows, *keys)


def percentile(arr, q):
  arr = np.asarray(arr, np.float64)
  arr = arr[np.isfinite(arr)]
  if arr.size == 0:
    return float("nan")
  return float(np.percentile(arr, q))


def semantic_context(row):
  return (
      int(value(row, "action")),
      int(value(row, "inventory_bucket")),
      int(value(row, "nearby_tile")),
      int(value(row, "achievement_stage")),
      int(value(row, "nearby_mob")),
  )


def canonical_bug_type(row):
  bug_type = str(row.get("bug_type", "")).strip()
  if bug_type.lower() in ("", "none", "null", "nan"):
    return ""
  return bug_type


def context_diversity(rows):
  context_keys = (
      "inventory_bucket", "nearby_tile", "achievement_stage", "nearby_mob")
  if not rows or not all(has_any(rows, key) for key in context_keys):
    return {
        "coverage": float("nan"),
        "unique_suspicious": float("nan"),
        "repeat_ratio": float("nan"),
    }
  contexts = set()
  suspicious_contexts = set()
  episode_seen = defaultdict(set)
  suspicious_hits = 0
  suspicious_repeats = 0
  for row in rows:
    if bool(value(row, "is_first")):
      continue
    context = semantic_context(row)
    contexts.add(context)
    if has_any([row], "suspicious_context", "log/suspicious_context") and value(
        row, "suspicious_context", "log/suspicious_context") > 0.5:
      suspicious_hits += 1
      episode = int(value(row, "episode_id", "episode_index"))
      if context in episode_seen[episode]:
        suspicious_repeats += 1
      episode_seen[episode].add(context)
      suspicious_contexts.add(context)
  return {
      "coverage": len(contexts),
      "unique_suspicious": len(suspicious_contexts),
      "repeat_ratio": suspicious_repeats / max(suspicious_hits, 1),
  }


def bug_discovery(rows, labels, split):
  bug_indices = np.flatnonzero(labels)
  bug_types = set()
  auc_sum = 0.0
  for row, label in zip(rows, labels):
    if label:
      bug_type = canonical_bug_type(row)
      if bug_type:
        bug_types.add(bug_type)
    auc_sum += len(bug_types)
  expected = EXPECTED_BUG_TYPES.get(split, set())
  first_index = int(bug_indices[0]) if bug_indices.size else len(rows)
  episodes = []
  for row in rows[:first_index + 1]:
    episode = int(value(row, "episode_id", "episode_index", default=-1))
    if episode >= 0 and episode not in episodes:
      episodes.append(episode)
  return {
      "event_count": int(labels.sum()),
      "found": int(bool(bug_indices.size)),
      # No-hit runs are right-censored one step beyond the evaluation budget.
      "first_step": first_index + 1,
      "first_episode": len(episodes) if bug_indices.size else len(episodes) + 1,
      "types": bug_types,
      "expected_count": len(expected),
      "coverage": len(bug_types & expected) / len(expected) if expected else float("nan"),
      "auc": auc_sum / max(len(rows), 1),
      "auc_norm": (
          auc_sum / max(len(rows), 1) / len(expected)
          if expected else float("nan")),
      "unexpected": len(bug_types - expected) if expected else 0,
  }


def auroc(labels, scores):
  labels = np.asarray(labels, np.int32)
  scores = np.asarray(scores, np.float64)
  mask = np.isfinite(scores)
  labels = labels[mask]
  scores = scores[mask]
  pos = labels == 1
  neg = labels == 0
  npos = int(pos.sum())
  nneg = int(neg.sum())
  if npos == 0 or nneg == 0:
    return float("nan")
  order = np.argsort(scores)
  ranks = np.empty_like(order, dtype=np.float64)
  ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
  # Average tied ranks.
  unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
  del unique
  if np.any(counts > 1):
    sums = np.bincount(inverse, ranks)
    avg = sums / counts
    ranks = avg[inverse]
  rank_sum_pos = float(ranks[pos].sum())
  return (rank_sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def auprc(labels, scores):
  labels = np.asarray(labels, np.int32)
  scores = np.asarray(scores, np.float64)
  mask = np.isfinite(scores)
  labels = labels[mask]
  scores = scores[mask]
  positives = int(labels.sum())
  if positives == 0:
    return float("nan")
  order = np.argsort(-scores)
  y = labels[order]
  tp = np.cumsum(y)
  fp = np.cumsum(1 - y)
  precision = tp / np.maximum(tp + fp, 1)
  recall = tp / positives
  prev_recall = np.concatenate([[0.0], recall[:-1]])
  return float(np.sum((recall - prev_recall) * precision))


def precision_at(labels, scores, frac):
  labels = np.asarray(labels, np.int32)
  scores = np.asarray(scores, np.float64)
  mask = np.isfinite(scores)
  labels = labels[mask]
  scores = scores[mask]
  if labels.size == 0:
    return float("nan")
  k = max(1, int(math.ceil(labels.size * frac)))
  order = np.argsort(-scores)[:k]
  return float(labels[order].mean())


def infer_case(path, root):
  rel = path.relative_to(root)
  parts = rel.parts
  seed = ""
  variant = ""
  split = ""
  phase = "eval"
  for part in parts:
    match = re.match(r"seed_(\d+)$", part)
    if match:
      seed = int(match.group(1))
      break
  name = path.name
  if name == "train" and path.parent.name.startswith("train_"):
    phase = "train"
    variant = path.parent.name.replace("train_", "", 1)
    split = "train"
  elif name.startswith("base_") and name.endswith("_eval"):
    phase = "eval"
    variant = "reference"
    split = name[len("base_"):-len("_eval")]
  elif name.startswith("eval_"):
    phase = "eval"
    body = name[len("eval_"):]
    for candidate in ("clean", "seen", "holdout", "sparse", "diagnostic"):
      suffix = f"_{candidate}"
      if body.endswith(suffix):
        variant = body[:-len(suffix)]
        split = candidate
        break
  return {
      "seed": seed,
      "variant": variant or name,
      "split": split or "unknown",
      "phase": phase,
      "path": str(path),
  }


def summarize_case_stream(path, root):
  """Summarize a case without retaining JSON trace rows in memory."""
  info = infer_case(path, root)
  fault_path = path / "fault_score_trace.jsonl"
  bug_path = path / "bug_trace.jsonl"
  transition_path = fault_path if fault_path.exists() else bug_path
  stats = defaultdict(lambda: [0.0, 0])
  maxima = {}
  labels = []
  scores = []
  bug_types = set()
  bug_type_stats = defaultdict(lambda: {
      "count": 0, "scores": [], "first_step": None,
      "threshold_seen": 0, "threshold_detected": 0})
  actions = set()
  contexts = set()
  suspicious_contexts = set()
  suspicious_episode_contexts = defaultdict(set)
  context_available = False
  suspicious_hits = 0
  suspicious_repeats = 0
  first_bug_step = None
  first_bug_episode_count = None
  episodes_before_first_bug = set()
  bug_discovery_auc_sum = 0.0
  steps = 0

  def add(name, val):
    if val is not None and math.isfinite(val):
      stats[name][0] += val
      stats[name][1] += 1

  def mean(name):
    total, count = stats[name]
    return total / count if count else float("nan")

  for row in iter_jsonl(transition_path):
    steps += 1
    action = optional_value(row, "action", "log/requested_action")
    if action is not None and math.isfinite(action):
      actions.add(int(action))
    label_value = optional_value(
        row, "bug_triggered", "log/fault_applied", "log/fault_manifested")
    label = int((label_value or 0.0) > 0.5)
    score = optional_value(
        row, "fault_score", "fault_score_raw", "fault/fault_score",
        "log/ref_fault_score")
    labels.append(label)
    scores.append(score if score is not None else float("nan"))

    episode = optional_value(row, "episode_id", "episode_index")
    if first_bug_step is None and episode is not None and math.isfinite(episode):
      episodes_before_first_bug.add(int(episode))
    if label and first_bug_step is None:
      first_bug_step = steps
      first_bug_episode_count = len(episodes_before_first_bug)

    suspicious = optional_value(
        row, "suspicious_context", "log/suspicious_context")
    trigger = optional_value(row, "log/fault_trigger_context")
    violation = optional_value(
        row, "constraint_violation", "log/constraint_violation")
    add("trigger", 1.0 if trigger is not None and trigger > 0.5 else 0.0 if trigger is not None else None)
    add("suspicious", 1.0 if suspicious is not None and suspicious > 0.5 else 0.0 if suspicious is not None else None)
    add("novelty_trigger", optional_value(
        row, "novelty_bound_triggered", "log/novelty_bound_triggered"))
    add("constraint_active", optional_value(
        row, "constraint_active", "log/constraint_active"))
    add("constraint_violation", violation)
    add("constraint_feasible", 1.0 if violation is not None and violation <= 0.0 else 0.0 if violation is not None else None)
    if suspicious is not None:
      if label:
        add("threshold_bug", 1.0 if suspicious > 0.5 else 0.0)
      else:
        add("threshold_normal", 1.0 if suspicious > 0.5 else 0.0)

    scalar_fields = {
        "raw_reward": ("raw_reward", "reward"),
        "augmented_reward": ("augmented_reward", "log/augmented_reward"),
        "fault_delta": ("fault_score_delta", "log/fault_score_delta"),
        "latent_kl": ("latent_kl_surprise", "fault/latent_kl_surprise", "log/ref_latent_kl_surprise"),
        "reward_error": ("reward_prediction_error", "fault/reward_prediction_error", "log/ref_reward_prediction_error"),
        "fault_bonus": ("fault_reward_bonus", "log/fault_reward_bonus"),
        "fault_reward_score": ("fault_reward_score", "log/fault_reward_score"),
        "novelty_score": ("novelty_bound_score", "log/novelty_bound_score"),
        "constraint_lambda": ("constraint_lambda", "log/constraint_lambda"),
        "semantic_fault": ("log/semantic_fault_applied",),
        "fault_episode": ("log/fault_episode",),
    }
    for name, keys in scalar_fields.items():
      val = optional_value(row, *keys)
      add(name, val)
      if name == "constraint_lambda" and val is not None and math.isfinite(val):
        maxima[name] = max(maxima.get(name, val), val)

    context_keys = (
        "inventory_bucket", "nearby_tile", "achievement_stage", "nearby_mob")
    context_vals = [optional_value(row, key) for key in context_keys]
    if (not bool(value(row, "is_first")) and action is not None and
        all(val is not None and math.isfinite(val) for val in context_vals)):
      context_available = True
      context = (int(action), *(int(val) for val in context_vals))
      contexts.add(context)
      if suspicious is not None and suspicious > 0.5:
        suspicious_hits += 1
        episode_key = int(episode) if episode is not None else -1
        if context in suspicious_episode_contexts[episode_key]:
          suspicious_repeats += 1
        suspicious_episode_contexts[episode_key].add(context)
        suspicious_contexts.add(context)

    if label:
      bug_type = canonical_bug_type(row)
      if bug_type:
        bug_types.add(bug_type)
        group = bug_type_stats[bug_type]
        group["count"] += 1
        if score is not None and math.isfinite(score):
          group["scores"].append(score)
        if group["first_step"] is None:
          group["first_step"] = steps
        if suspicious is not None:
          group["threshold_seen"] += 1
          group["threshold_detected"] += int(suspicious > 0.5)
    bug_discovery_auc_sum += len(bug_types)

  labels = np.asarray(labels, np.int32)
  scores = np.asarray(scores, np.float64)
  score_mask = np.isfinite(scores)
  paired_labels = labels[score_mask]
  paired_scores = scores[score_mask]
  expected = EXPECTED_BUG_TYPES.get(info["split"], set())
  event_count = int(labels.sum())
  no_bug_episode_count = len(episodes_before_first_bug) + 1
  diversity_coverage = float(len(contexts)) if context_available else float("nan")
  diversity_suspicious = float(len(suspicious_contexts)) if context_available else float("nan")
  repeat_ratio = (
      suspicious_repeats / suspicious_hits
      if context_available and suspicious_hits else 0.0 if context_available else float("nan"))

  bug_rewards = []
  task_scores = []
  bug_trace_rows = 0
  for row in iter_jsonl(bug_path):
    bug_trace_rows += 1
    reward = optional_value(row, "reward")
    task_score = optional_value(row, "log/score")
    if reward is not None and math.isfinite(reward):
      bug_rewards.append(reward)
    if task_score is not None and math.isfinite(task_score):
      task_scores.append(task_score)

  episode_scores = []
  episode_lengths = []
  for row in iter_jsonl(path / "scores.jsonl"):
    score = optional_value(row, "episode/score")
    length = optional_value(row, "episode/length")
    if score is not None and math.isfinite(score):
      episode_scores.append(score)
    if length is not None and math.isfinite(length):
      episode_lengths.append(length)
  task_score = float(np.mean(episode_scores)) if episode_scores else float("nan")
  last_metric_step = 0
  for row in iter_jsonl(path / "metrics.jsonl"):
    step = optional_value(row, "step")
    if step is not None and math.isfinite(step):
      last_metric_step = max(last_metric_step, int(step))

  case_auroc = auroc(paired_labels, paired_scores)
  case_auprc = auprc(paired_labels, paired_scores)
  prevalence = event_count / steps if steps else float("nan")
  bug_scores = paired_scores[paired_labels == 1]
  normal_scores = paired_scores[paired_labels == 0]
  bug_score_mean = float(bug_scores.mean()) if bug_scores.size else float("nan")
  normal_score_mean = float(normal_scores.mean()) if normal_scores.size else float("nan")
  per_1k = steps / 1000.0
  out = {
      **info,
      "steps": steps,
      "bug_trace_rows": bug_trace_rows,
      "fault_trace_rows": steps if fault_path.exists() else 0,
      "reward_mean": float(np.mean(bug_rewards)) if bug_rewards else float("nan"),
      "raw_reward_mean": mean("raw_reward"),
      "augmented_reward_mean": mean("augmented_reward"),
      "task_score_mean": float(np.mean(task_scores)) if task_scores else float("nan"),
      "task_score_max": float(np.max(task_scores)) if task_scores else float("nan"),
      "episode_score_mean": task_score,
      "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
      "fault_applied_rate": prevalence,
      "fault_trigger_rate": mean("trigger"),
      "semantic_fault_rate": mean("semantic_fault"),
      "fault_episode_rate": mean("fault_episode"),
      "fault_score_mean": float(paired_scores.mean()) if paired_scores.size else float("nan"),
      "fault_score_p95": percentile(paired_scores, 95),
      "fault_score_delta_mean": mean("fault_delta"),
      "latent_kl_mean": mean("latent_kl"),
      "reward_error_mean": mean("reward_error"),
      "fault_reward_bonus_mean": mean("fault_bonus"),
      "fault_reward_score_mean": mean("fault_reward_score"),
      "novelty_bound_score_mean": mean("novelty_score"),
      "novelty_bound_trigger_rate": mean("novelty_trigger"),
      "constraint_active_rate": mean("constraint_active"),
      "constraint_lambda_mean": mean("constraint_lambda"),
      "constraint_lambda_max": maxima.get("constraint_lambda", float("nan")),
      "constraint_violation_mean": mean("constraint_violation"),
      "task_constraint_feasible_rate": mean("constraint_feasible"),
      "bug_event_count": event_count,
      "bug_events_per_10k": event_count * 10000.0 / max(steps, 1),
      "bug_prevalence": prevalence,
      "bug_found": int(first_bug_step is not None),
      "unique_bug_types": len(bug_types),
      "expected_bug_types": len(expected),
      "bug_type_coverage_fraction": len(bug_types & expected) / len(expected) if expected else float("nan"),
      "bug_discovery_auc": bug_discovery_auc_sum / max(steps, 1),
      "bug_discovery_auc_norm": (
          bug_discovery_auc_sum / max(steps, 1) / len(expected)
          if expected else float("nan")),
      "unexpected_bug_types": len(bug_types - expected) if expected else 0,
      "time_to_first_bug_steps": first_bug_step if first_bug_step is not None else steps + 1,
      "episodes_to_first_bug": first_bug_episode_count if first_bug_episode_count is not None else no_bug_episode_count,
      "semantic_context_coverage": diversity_coverage,
      "semantic_context_coverage_per_1k": diversity_coverage / per_1k if per_1k and math.isfinite(diversity_coverage) else float("nan"),
      "unique_suspicious_context_count": diversity_suspicious,
      "unique_suspicious_context_per_1k": diversity_suspicious / per_1k if per_1k and math.isfinite(diversity_suspicious) else float("nan"),
      "suspicious_context_rate": mean("suspicious"),
      "repeated_suspicious_context_ratio": repeat_ratio,
      "threshold_bug_recall": mean("threshold_bug"),
      "threshold_false_positive_rate": mean("threshold_normal"),
      "clean_false_positive_rate": mean("threshold_normal") if info["split"] == "clean" else float("nan"),
      "manifestation_given_trigger": event_count / stats["trigger"][0] if stats["trigger"][1] and stats["trigger"][0] else 0.0 if stats["trigger"][1] else float("nan"),
      "bug_per_task_score": prevalence / max(abs(task_score), 1e-8) if math.isfinite(task_score) else float("nan"),
      "auroc": case_auroc,
      "auroc_above_chance": case_auroc - 0.5 if math.isfinite(case_auroc) else float("nan"),
      "auprc": case_auprc,
      "auprc_lift": case_auprc / prevalence if prevalence > 0 else float("nan"),
      "precision_at_top1pct": precision_at(paired_labels, paired_scores, 0.01),
      "precision_at_top5pct": precision_at(paired_labels, paired_scores, 0.05),
      "fault_score_bug_mean": bug_score_mean,
      "fault_score_normal_mean": normal_score_mean,
      "fault_score_bug_normal_gap": bug_score_mean - normal_score_mean if math.isfinite(bug_score_mean) and math.isfinite(normal_score_mean) else float("nan"),
      "unique_actions": len(actions),
      "last_metric_step": last_metric_step,
  }

  type_rows = []
  for bug_type, group in sorted(bug_type_stats.items()):
    type_scores = np.asarray(group["scores"], np.float64)
    threshold_seen = group["threshold_seen"]
    type_rows.append({
        **info,
        "bug_type": bug_type,
        "event_count": group["count"],
        "events_per_10k": group["count"] * 10000.0 / max(steps, 1),
        "fault_score_mean": float(type_scores.mean()) if type_scores.size else float("nan"),
        "fault_score_median": float(np.median(type_scores)) if type_scores.size else float("nan"),
        "fault_score_p95": percentile(type_scores, 95),
        "first_step": group["first_step"],
        "threshold_detected_count": group["threshold_detected"] if threshold_seen else float("nan"),
        "threshold_recall": group["threshold_detected"] / threshold_seen if threshold_seen else float("nan"),
    })
  return out, type_rows


def summarize_case(
    path, root, bug_rows=None, fault_rows=None, metric_rows=None,
    score_rows=None):
  info = infer_case(path, root)
  bug_rows = read_jsonl(path / "bug_trace.jsonl") if bug_rows is None else bug_rows
  fault_rows = read_jsonl(path / "fault_score_trace.jsonl") if fault_rows is None else fault_rows
  metric_rows = read_jsonl(path / "metrics.jsonl") if metric_rows is None else metric_rows
  score_rows = read_jsonl(path / "scores.jsonl") if score_rows is None else score_rows

  rows = fault_rows or bug_rows
  labels = binary(rows, "bug_triggered", "log/fault_applied", "log/fault_manifested")
  scores = values(rows, "fault_score", "fault_score_raw", "fault/fault_score", "log/ref_fault_score")

  rewards = values(bug_rows, "reward") if bug_rows else values(rows, "raw_reward")
  raw_rewards = values(rows, "raw_reward", "reward")
  aug_rewards = values(rows, "augmented_reward", "log/augmented_reward")
  task_scores = values(bug_rows, "log/score") if bug_rows else np.asarray([], np.float64)
  episode_scores = values(score_rows, "episode/score") if score_rows else np.asarray([], np.float64)
  episode_lengths = values(score_rows, "episode/length") if score_rows else np.asarray([], np.float64)
  actions = set(int(value(row, "action", "log/requested_action")) for row in rows) if rows else set()
  diversity = context_diversity(rows)
  trigger = optional_binary(rows, "log/fault_trigger_context")
  suspicious = optional_binary(rows, "suspicious_context", "log/suspicious_context")
  novelty_trigger = optional_binary(rows, "novelty_bound_triggered", "log/novelty_bound_triggered")
  constraint_active = optional_binary(rows, "constraint_active", "log/constraint_active")
  constraint_violation = optional_values(rows, "constraint_violation", "log/constraint_violation")
  task_score = float(np.mean(episode_scores)) if episode_scores.size else float("nan")
  discovery = bug_discovery(rows, labels, info["split"])
  normal = labels == 0
  bug = labels == 1
  score_bug_mean = float(np.mean(scores[bug])) if scores.size and bug.any() else float("nan")
  score_normal_mean = float(np.mean(scores[normal])) if scores.size and normal.any() else float("nan")
  prevalence = float(labels.mean()) if labels.size else float("nan")
  case_auroc = auroc(labels, scores) if rows else float("nan")
  case_auprc = auprc(labels, scores) if rows else float("nan")
  suspicious_bug_recall = (
      float(suspicious[bug].mean()) if suspicious.size and bug.any() else float("nan"))
  suspicious_fpr = (
      float(suspicious[normal].mean()) if suspicious.size and normal.any() else float("nan"))
  steps_per_1k = len(rows) / 1000.0

  out = {
      **info,
      "steps": len(rows),
      "bug_trace_rows": len(bug_rows),
      "fault_trace_rows": len(fault_rows),
      "reward_mean": float(np.mean(rewards)) if rewards.size else float("nan"),
      "raw_reward_mean": float(np.mean(raw_rewards)) if raw_rewards.size else float("nan"),
      "augmented_reward_mean": float(np.mean(aug_rewards)) if aug_rewards.size else float("nan"),
      "task_score_mean": float(np.mean(task_scores)) if task_scores.size else float("nan"),
      "task_score_max": float(np.max(task_scores)) if task_scores.size else float("nan"),
      "episode_score_mean": task_score,
      "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths.size else float("nan"),
      "fault_applied_rate": float(binary(rows, "bug_triggered", "log/fault_applied").mean()) if rows else float("nan"),
      "fault_trigger_rate": float(binary(rows, "log/fault_trigger_context").mean()) if rows else float("nan"),
      "semantic_fault_rate": float(binary(rows, "log/semantic_fault_applied").mean()) if rows else float("nan"),
      "fault_episode_rate": float(binary(rows, "log/fault_episode").mean()) if rows else float("nan"),
      "fault_score_mean": float(np.mean(scores)) if scores.size else float("nan"),
      "fault_score_p95": percentile(scores, 95),
      "fault_score_delta_mean": float(np.mean(values(rows, "fault_score_delta", "log/fault_score_delta"))) if rows else float("nan"),
      "latent_kl_mean": float(np.mean(values(rows, "latent_kl_surprise", "fault/latent_kl_surprise", "log/ref_latent_kl_surprise"))) if rows else float("nan"),
      "reward_error_mean": float(np.mean(values(rows, "reward_prediction_error", "fault/reward_prediction_error", "log/ref_reward_prediction_error"))) if rows else float("nan"),
      "fault_reward_bonus_mean": float(np.mean(values(rows, "fault_reward_bonus", "log/fault_reward_bonus"))) if rows else float("nan"),
      "fault_reward_score_mean": float(np.mean(values(rows, "fault_reward_score", "log/fault_reward_score"))) if rows else float("nan"),
      "novelty_bound_score_mean": float(np.mean(values(rows, "novelty_bound_score", "log/novelty_bound_score"))) if rows else float("nan"),
      "novelty_bound_trigger_rate": float(novelty_trigger.mean()) if novelty_trigger.size else float("nan"),
      "constraint_active_rate": float(constraint_active.mean()) if constraint_active.size else float("nan"),
      "constraint_lambda_mean": float(np.mean(values(rows, "constraint_lambda", "log/constraint_lambda"))) if rows else float("nan"),
      "constraint_lambda_max": float(np.max(values(rows, "constraint_lambda", "log/constraint_lambda"))) if rows else float("nan"),
      "constraint_violation_mean": float(np.mean(constraint_violation)) if constraint_violation.size else float("nan"),
      "task_constraint_feasible_rate": float(np.mean(constraint_violation <= 0.0)) if constraint_violation.size else float("nan"),
      "bug_event_count": discovery["event_count"],
      "bug_events_per_10k": discovery["event_count"] * 10000.0 / max(len(rows), 1),
      "bug_prevalence": prevalence,
      "bug_found": discovery["found"],
      "unique_bug_types": len(discovery["types"]),
      "expected_bug_types": discovery["expected_count"],
      "bug_type_coverage_fraction": discovery["coverage"],
      "bug_discovery_auc": discovery["auc"],
      "bug_discovery_auc_norm": discovery["auc_norm"],
      "unexpected_bug_types": discovery["unexpected"],
      "time_to_first_bug_steps": discovery["first_step"],
      "episodes_to_first_bug": discovery["first_episode"],
      "semantic_context_coverage": float(diversity["coverage"]) if rows else float("nan"),
      "semantic_context_coverage_per_1k": (
          float(diversity["coverage"]) / steps_per_1k
          if rows and math.isfinite(diversity["coverage"]) else float("nan")),
      "unique_suspicious_context_count": float(diversity["unique_suspicious"]) if rows else float("nan"),
      "unique_suspicious_context_per_1k": (
          float(diversity["unique_suspicious"]) / steps_per_1k
          if rows and math.isfinite(diversity["unique_suspicious"]) else float("nan")),
      "suspicious_context_rate": float(suspicious.mean()) if suspicious.size else float("nan"),
      "repeated_suspicious_context_ratio": float(diversity["repeat_ratio"]) if rows else float("nan"),
      "threshold_bug_recall": suspicious_bug_recall,
      "threshold_false_positive_rate": suspicious_fpr,
      "clean_false_positive_rate": suspicious_fpr if info["split"] == "clean" else float("nan"),
      "manifestation_given_trigger": float(labels.sum() / max(trigger.sum(), 1)) if trigger.size else float("nan"),
      "bug_per_task_score": float(labels.mean() / max(abs(task_score), 1e-8)) if rows and math.isfinite(task_score) else float("nan"),
      "auroc": case_auroc,
      "auroc_above_chance": case_auroc - 0.5 if math.isfinite(case_auroc) else float("nan"),
      "auprc": case_auprc,
      "auprc_lift": case_auprc / prevalence if prevalence > 0 else float("nan"),
      "precision_at_top1pct": precision_at(labels, scores, 0.01) if rows else float("nan"),
      "precision_at_top5pct": precision_at(labels, scores, 0.05) if rows else float("nan"),
      "fault_score_bug_mean": score_bug_mean,
      "fault_score_normal_mean": score_normal_mean,
      "fault_score_bug_normal_gap": (
          score_bug_mean - score_normal_mean
          if math.isfinite(score_bug_mean) and math.isfinite(score_normal_mean)
          else float("nan")),
      "unique_actions": len(actions),
      "last_metric_step": max([int(value(row, "step")) for row in metric_rows], default=0),
  }
  return out


def discover_cases(root):
  paths = set()
  for trace in root.glob("seed_*/**/bug_trace.jsonl"):
    paths.add(trace.parent)
  for trace in root.glob("seed_*/**/fault_score_trace.jsonl"):
    paths.add(trace.parent)
  return sorted(paths)


def summarize_bug_types(path, root, rows=None):
  info = infer_case(path, root)
  rows = rows if rows is not None else (
      read_jsonl(path / "fault_score_trace.jsonl") or read_jsonl(
          path / "bug_trace.jsonl"))
  if not rows:
    return []
  labels = binary(rows, "bug_triggered", "log/fault_applied", "log/fault_manifested")
  scores = values(
      rows, "fault_score", "fault_score_raw", "fault/fault_score",
      "log/ref_fault_score")
  has_threshold = has_any(rows, "suspicious_context", "log/suspicious_context")
  suspicious = optional_binary(rows, "suspicious_context", "log/suspicious_context")
  groups = defaultdict(list)
  for index, (row, label) in enumerate(zip(rows, labels)):
    if not label:
      continue
    bug_type = str(row.get("bug_type", "")).strip()
    if bug_type.lower() in ("", "none", "null", "nan"):
      bug_type = "unknown"
    groups[bug_type].append(index)
  result = []
  for bug_type, indices in sorted(groups.items()):
    idx = np.asarray(indices, np.int64)
    row = {
        **info,
        "bug_type": bug_type,
        "event_count": len(indices),
        "events_per_10k": len(indices) * 10000.0 / len(rows),
        "fault_score_mean": float(scores[idx].mean()),
        "fault_score_median": float(np.median(scores[idx])),
        "fault_score_p95": percentile(scores[idx], 95),
        "first_step": int(idx[0]) + 1,
        "threshold_detected_count": int(suspicious[idx].sum()) if has_threshold else float("nan"),
        "threshold_recall": float(suspicious[idx].mean()) if has_threshold else float("nan"),
    }
    result.append(row)
  return result


def aggregate_bug_types(rows):
  groups = defaultdict(list)
  for row in rows:
    groups[(row["variant"], row["split"], row["bug_type"])].append(row)
  result = []
  metrics = (
      "event_count", "events_per_10k", "fault_score_mean",
      "fault_score_median", "fault_score_p95", "first_step",
      "threshold_recall")
  for (variant, split, bug_type), items in sorted(groups.items()):
    out = {
        "variant": variant,
        "split": split,
        "bug_type": bug_type,
        "num_seeds": len({row["seed"] for row in items}),
        "event_count_total": sum(int(row["event_count"]) for row in items),
    }
    for metric in metrics:
      vals = np.asarray(
          [safe_float(row.get(metric), float("nan")) for row in items],
          np.float64)
      vals = vals[np.isfinite(vals)]
      out[f"{metric}_mean"] = float(vals.mean()) if vals.size else float("nan")
      out[f"{metric}_std"] = (
          float(vals.std(ddof=1)) if vals.size > 1 else 0.0 if vals.size else float("nan"))
    result.append(out)
  return result


def aggregate(rows):
  groups = defaultdict(list)
  for row in rows:
    groups[(row["variant"], row["split"], row["phase"])].append(row)
  out = []
  for (variant, split, phase), items in sorted(groups.items()):
    base = {
        "variant": variant,
        "split": split,
        "phase": phase,
        "num_seeds": len({x["seed"] for x in items if x["seed"] != ""}),
    }
    for metric in METRICS:
      vals = np.asarray([safe_float(x.get(metric), float("nan")) for x in items], np.float64)
      vals = vals[np.isfinite(vals)]
      if vals.size:
        base[f"{metric}_mean"] = float(vals.mean())
        base[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        base[f"{metric}_sem"] = float(base[f"{metric}_std"] / math.sqrt(vals.size))
      else:
        base[f"{metric}_mean"] = float("nan")
        base[f"{metric}_std"] = float("nan")
        base[f"{metric}_sem"] = float("nan")
    out.append(base)
  return out


def add_deltas(agg_rows, baseline):
  by_split = {
      row["split"]: row for row in agg_rows
      if row["variant"] == baseline and row["phase"] == "eval"
  }
  for row in agg_rows:
    base = by_split.get(row["split"])
    if not base:
      continue
    for metric in ("task_score_mean", "episode_score_mean", "fault_applied_rate", "auroc", "auprc"):
      key = f"{metric}_mean"
      row[f"{metric}_baseline"] = base.get(key, float("nan"))
      row[f"{metric}_delta"] = row.get(key, float("nan")) - base.get(key, float("nan"))
    task = row.get("episode_score_mean_mean")
    base_task = base.get("episode_score_mean_mean")
    if base_task and math.isfinite(base_task) and abs(base_task) > 1e-8:
      row["task_retention"] = task / base_task
    row["objective_score"] = (
        safe_float(row.get("fault_applied_rate_delta")) * 100.0
        + safe_float(row.get("auroc_delta"))
        + safe_float(row.get("auprc_delta")) * 2.0
        + safe_float(row.get("episode_score_mean_delta")) * 0.1
    )


def pareto_front(agg_rows):
  rows = [
      row for row in agg_rows
      if row["phase"] == "eval" and row["split"] in ("seen", "holdout", "sparse")
      and row["variant"] != "reference"
  ]
  fronts = []
  for split in sorted({row["split"] for row in rows}):
    sub = [row for row in rows if row["split"] == split]
    for row in sub:
      obj = np.asarray([
          safe_float(row.get("episode_score_mean_mean")),
          safe_float(row.get("fault_applied_rate_mean")),
          safe_float(row.get("auroc_mean")),
          safe_float(row.get("auprc_mean")),
      ], np.float64)
      dominated = False
      for other in sub:
        if other is row:
          continue
        other_obj = np.asarray([
            safe_float(other.get("episode_score_mean_mean")),
            safe_float(other.get("fault_applied_rate_mean")),
            safe_float(other.get("auroc_mean")),
            safe_float(other.get("auprc_mean")),
        ], np.float64)
        if np.all(other_obj >= obj) and np.any(other_obj > obj):
          dominated = True
          break
      if not dominated:
        fronts.append(row)
  return fronts


def setup_style():
  plt.rcParams.update({
      "figure.dpi": 140,
      "savefig.dpi": 220,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "axes.grid": True,
      "grid.alpha": 0.25,
      "font.size": 10,
  })


def savefig(fig, outdir, stem, formats):
  paths = []
  for fmt in formats:
    path = outdir / f"{stem}.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    paths.append(path)
  plt.close(fig)
  return paths


def plot_metric_grid(agg_rows, outdir, formats, error_bars="sem"):
  rows = [r for r in agg_rows if r["phase"] == "eval" and r["split"] in ("clean", "seen", "holdout", "sparse")]
  if not rows:
    return []
  variants = sorted({r["variant"] for r in rows})
  splits = ["clean", "seen", "holdout", "sparse"]
  fig, axes = plt.subplots(len(PLOT_METRICS), len(splits), figsize=(4.2 * len(splits), 12), squeeze=False)
  for i, (metric, title) in enumerate(PLOT_METRICS):
    for j, split in enumerate(splits):
      ax = axes[i][j]
      sub = [r for r in rows if r["split"] == split]
      sub = sorted(sub, key=lambda x: variants.index(x["variant"]))
      labels = [r["variant"] for r in sub]
      y = [safe_float(r.get(f"{metric}_mean"), float("nan")) for r in sub]
      err = error_values(sub, metric, error_bars)
      ax.bar(np.arange(len(labels)), y, yerr=err, capsize=3)
      if metric == "auroc":
        ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1)
      ax.set_title(f"{title} / {split}")
      ax.set_xticks(np.arange(len(labels)))
      ax.set_xticklabels(labels, rotation=35, ha="right")
      if j == 0:
        ax.set_ylabel(title)
      if i == 0 and j == len(splits) - 1 and error_bars != "none":
        ax.text(
            1.0, 1.12, f"Error bars: {error_bars.upper()}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
  fig.tight_layout()
  return savefig(fig, outdir, "craftax_multiseed_metric_grid", formats)


def plot_tradeoffs(agg_rows, outdir, formats):
  rows = [r for r in agg_rows if r["phase"] == "eval" and r["split"] in ("seen", "holdout", "sparse")]
  if not rows:
    return []
  fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  plots = [
      ("episode_score_mean_mean", "fault_applied_rate_mean", "Episode score", "Bug rate"),
      ("episode_score_mean_mean", "auroc_mean", "Episode score", "AUROC"),
      ("episode_score_mean_mean", "auprc_mean", "Episode score", "AUPRC"),
  ]
  markers = {"seen": "o", "holdout": "s", "sparse": "^"}
  variants = sorted({r["variant"] for r in rows})
  cmap = plt.get_cmap("tab10")
  colors = {variant: cmap(i % 10) for i, variant in enumerate(variants)}
  for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, plots):
    for row in rows:
      x = safe_float(row.get(xkey), float("nan"))
      y = safe_float(row.get(ykey), float("nan"))
      if not math.isfinite(x) or not math.isfinite(y):
        continue
      ax.scatter(
          x, y, s=80, color=colors[row["variant"]],
          marker=markers.get(row["split"], "o"), edgecolor="white", linewidth=0.8)
      ax.annotate(row["variant"].replace("_beta", "\nbeta"), (x, y), xytext=(4, 4),
                  textcoords="offset points", fontsize=7)
    if ykey == "auroc_mean":
      ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{xlabel} vs {ylabel}")
  fig.tight_layout()
  return savefig(fig, outdir, "craftax_multiseed_tradeoffs", formats)


def plot_context_metrics(agg_rows, outdir, formats, error_bars="sem"):
  rows = [
      r for r in agg_rows
      if r["phase"] == "eval" and
      r["split"] in ("clean", "seen", "holdout", "sparse")]
  if not rows:
    return []
  metrics = [
      ("semantic_context_coverage_per_1k", "Semantic contexts / 1k steps"),
      ("unique_suspicious_context_per_1k", "Unique suspicious contexts / 1k"),
      ("suspicious_context_rate", "Suspicious context rate"),
      ("repeated_suspicious_context_ratio", "Repeated suspicious ratio"),
      ("manifestation_given_trigger", "Manifestation given trigger"),
      ("bug_per_task_score", "Bug rate per task score"),
  ]
  variants = sorted({r["variant"] for r in rows})
  splits = ["clean", "seen", "holdout", "sparse"]
  fig, axes = plt.subplots(
      len(metrics), len(splits), figsize=(4.2 * len(splits), 17),
      squeeze=False)
  for i, (metric, title) in enumerate(metrics):
    for j, split in enumerate(splits):
      ax = axes[i][j]
      sub = sorted(
          (r for r in rows if r["split"] == split),
          key=lambda x: variants.index(x["variant"]))
      labels = [r["variant"] for r in sub]
      y = [safe_float(r.get(f"{metric}_mean"), float("nan")) for r in sub]
      err = error_values(sub, metric, error_bars)
      ax.bar(np.arange(len(labels)), y, yerr=err, capsize=3)
      ax.set_title(f"{title} / {split}")
      ax.set_xticks(np.arange(len(labels)))
      ax.set_xticklabels(labels, rotation=35, ha="right")
      if j == 0:
        ax.set_ylabel(title)
  fig.tight_layout()
  return savefig(fig, outdir, "craftax_contextual_testing_metrics", formats)


def plot_bar_dashboard(
    agg_rows, outdir, formats, metrics, stem, splits=None,
    error_bars="sem"):
  splits = splits or ["clean", "seen", "holdout", "sparse"]
  rows = [
      row for row in agg_rows
      if row["phase"] == "eval" and row["split"] in splits]
  if not rows:
    return []
  variants = sorted({row["variant"] for row in rows})
  fig, axes = plt.subplots(
      len(metrics), len(splits),
      figsize=(4.2 * len(splits), max(3.2 * len(metrics), 4.0)),
      squeeze=False)
  for i, (metric, title) in enumerate(metrics):
    for j, split in enumerate(splits):
      ax = axes[i][j]
      sub = sorted(
          (row for row in rows if row["split"] == split),
          key=lambda row: variants.index(row["variant"]))
      labels = [row["variant"] for row in sub]
      means = [safe_float(row.get(f"{metric}_mean"), float("nan")) for row in sub]
      errors = error_values(sub, f"{metric}_mean", error_bars)
      ax.bar(np.arange(len(labels)), means, yerr=errors, capsize=3)
      if metric == "auroc":
        ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1)
      ax.set_title(f"{title} / {split}")
      ax.set_xticks(np.arange(len(labels)))
      ax.set_xticklabels(labels, rotation=35, ha="right")
      if j == 0:
        ax.set_ylabel(title)
      if i == 0 and j == len(splits) - 1 and error_bars != "none":
        ax.text(
            1.0, 1.12, f"Error bars: {error_bars.upper()}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
  fig.tight_layout()
  return savefig(fig, outdir, stem, formats)


def plot_generalization_gaps(agg_rows, outdir, formats):
  indexed = {
      (row["variant"], row["split"]): row for row in agg_rows
      if row["phase"] == "eval" and row["split"] in ("seen", "holdout")
  }
  variants = sorted({variant for variant, _ in indexed})
  metrics = [
      ("episode_score_mean_mean", "Episode score"),
      ("fault_applied_rate_mean", "Bug rate"),
      ("unique_bug_types_mean", "Unique bug types"),
      ("auroc_mean", "AUROC"),
      ("auprc_lift_mean", "AUPRC lift"),
  ]
  if not variants:
    return []
  fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4.6), squeeze=False)
  for ax, (metric, title) in zip(axes[0], metrics):
    deltas = []
    for variant in variants:
      seen = indexed.get((variant, "seen"), {})
      holdout = indexed.get((variant, "holdout"), {})
      deltas.append(
          safe_float(holdout.get(metric), float("nan")) -
          safe_float(seen.get(metric), float("nan")))
    ax.bar(np.arange(len(variants)), deltas)
    ax.axhline(0.0, color="#444444", linewidth=1)
    ax.set_title(f"Holdout - seen\n{title}")
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(variants, rotation=35, ha="right")
  fig.tight_layout()
  return savefig(fig, outdir, "craftax_seen_holdout_generalization_gaps", formats)


def plot_bug_type_heatmaps(rows, outdir, formats):
  rows = [row for row in rows if row["split"] in ("seen", "holdout", "sparse")]
  if not rows:
    return []
  variants = sorted({row["variant"] for row in rows})
  columns = sorted({(row["split"], row["bug_type"]) for row in rows})
  matrices = []
  for metric in ("events_per_10k_mean", "threshold_recall_mean"):
    matrix = np.full((len(variants), len(columns)), np.nan)
    index = {
        (row["variant"], row["split"], row["bug_type"]): row for row in rows}
    for i, variant in enumerate(variants):
      for j, (split, bug_type) in enumerate(columns):
        row = index.get((variant, split, bug_type))
        if row:
          matrix[i, j] = safe_float(row.get(metric), float("nan"))
    matrices.append(matrix)
  fig, axes = plt.subplots(2, 1, figsize=(max(16, 0.65 * len(columns)), 8.5))
  for ax, matrix, title, cmap in zip(
      axes, matrices,
      ("Bug events per 10k steps", "Threshold recall by bug type"),
      ("viridis", "magma")):
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(
        [f"{split}\n{bug_type}" for split, bug_type in columns],
        rotation=55, ha="right", fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
  fig.tight_layout()
  return savefig(fig, outdir, "craftax_bug_type_heatmaps", formats)


def write_markdown(path, agg_rows, fronts):
  lines = [
      "# Craftax Multi-Seed Fault Results",
      "",
      "Higher task score means better game competence; higher bug rate/AUROC/AUPRC means stronger bug-seeking or detection signal.",
      "",
      "## Pareto Candidates",
      "",
  ]
  if fronts:
    for row in fronts:
      lines.append(
          f"- `{row['split']}` / `{row['variant']}`: "
          f"score={safe_float(row.get('episode_score_mean_mean')):.4f}, "
          f"bug_rate={safe_float(row.get('fault_applied_rate_mean')):.5f}, "
          f"auroc={safe_float(row.get('auroc_mean')):.4f}, "
          f"auprc={safe_float(row.get('auprc_mean')):.4f}")
  else:
    lines.append("- No Pareto rows found.")
  lines.extend(["", "## Best By Split", ""])
  for split in ("seen", "holdout", "sparse"):
    sub = [r for r in agg_rows if r["phase"] == "eval" and r["split"] == split]
    if not sub:
      continue
    best_bug = max(sub, key=lambda r: safe_float(r.get("fault_applied_rate_mean"), -1))
    best_task = max(sub, key=lambda r: safe_float(r.get("episode_score_mean_mean"), -1))
    best_score = max(sub, key=lambda r: safe_float(r.get("objective_score"), -1))
    lines.extend([
        f"### {split}",
        "",
        f"- Best task: `{best_task['variant']}` ({safe_float(best_task.get('episode_score_mean_mean')):.4f})",
        f"- Best bug rate: `{best_bug['variant']}` ({safe_float(best_bug.get('fault_applied_rate_mean')):.5f})",
        f"- Best combined score: `{best_score['variant']}` ({safe_float(best_score.get('objective_score')):.4f})",
        "",
    ])
  path.write_text("\n".join(lines), encoding="utf-8")


def write_metric_definitions(path):
  lines = [
      "# Metric Definitions",
      "",
      "## Coverage safeguards",
      "",
      "- `unique_bug_types` is recomputed as the set of non-empty `bug_type` values on rows where `bug_triggered=1`.",
      "- `bug_type_coverage_fraction` is observed expected bug types divided by the configured expected types (seen=8, holdout/sparse=7).",
      "- `unique_bug_count_cumulative` is intentionally ignored because it counts repeated bug events, not unique bug types.",
      "- `unique_tile_coverage_cumulative` is intentionally ignored because the Craftax wrapper does not populate that Crafter-specific metric.",
      "- `semantic_context_coverage` is the number of unique `(action, inventory_bucket, nearby_tile, achievement_stage, nearby_mob)` tuples.",
      "- `semantic_context_coverage_per_1k` normalizes this count by trace length so different evaluation budgets can be compared.",
      "",
      "## Detection and discovery",
      "",
      "- `bug_events_per_10k` is the number of applied bug events per 10,000 evaluated transitions.",
      "- `bug_discovery_auc` is the mean cumulative unique bug-type count over the evaluation horizon; it rewards finding diverse bugs early rather than only by the final step.",
      "- `bug_discovery_auc_norm` divides `bug_discovery_auc` by the configured expected bug-type count for the split.",
      "- `time_to_first_bug_steps` is the first applied bug row index (1-based). Runs with no bug are right-censored at `evaluation_steps + 1`.",
      "- `threshold_bug_recall` is the fraction of bug rows marked `suspicious_context`.",
      "- `threshold_false_positive_rate` is the fraction of normal rows marked `suspicious_context`; `clean_false_positive_rate` reports it only on clean splits.",
      "- `auprc_lift` is AUPRC divided by bug prevalence. A value above 1 is better than the prevalence baseline.",
      "- `fault_score_bug_normal_gap` is mean fault score on bug rows minus mean fault score on normal rows.",
      "",
      "All rate denominators are transitions in the corresponding trace. Error bars are standard errors across seeds.",
  ]
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
  args = parse_args()
  root = Path(args.root).expanduser()
  outdir = Path(args.outdir).expanduser() if args.outdir else root / "analysis"
  outdir.mkdir(parents=True, exist_ok=True)

  cases = discover_cases(root)
  if args.eval_only:
    cases = [path for path in cases if infer_case(path, root)["phase"] == "eval"]
  if args.seeds is not None:
    seeds = set(args.seeds)
    cases = [path for path in cases if infer_case(path, root)["seed"] in seeds]
  rows = []
  bug_type_rows = []
  for path in cases:
    summary, type_rows = summarize_case_stream(path, root)
    rows.append(summary)
    bug_type_rows.extend(type_rows)
  bug_type_agg = aggregate_bug_types(bug_type_rows)
  agg_rows = aggregate(rows)
  add_deltas(agg_rows, args.baseline)
  fronts = pareto_front(agg_rows)

  per_fields = ["seed", "variant", "split", "phase", "path"] + METRICS + [
      "bug_trace_rows", "fault_trace_rows", "raw_reward_mean", "task_score_max", "last_metric_step"]
  agg_fields = ["variant", "split", "phase", "num_seeds"]
  for metric in METRICS:
    agg_fields += [f"{metric}_mean", f"{metric}_std", f"{metric}_sem"]
  agg_fields += [
      "task_score_mean_delta", "episode_score_mean_delta",
      "fault_applied_rate_delta", "auroc_delta", "auprc_delta",
      "task_retention", "objective_score"]

  write_csv(outdir / "per_run_metrics.csv", rows, per_fields)
  write_csv(outdir / "aggregate_metrics.csv", agg_rows, agg_fields)
  write_csv(outdir / "pareto_front.csv", fronts, agg_fields)
  bug_type_fields = [
      "seed", "variant", "split", "phase", "path", "bug_type",
      "event_count", "events_per_10k", "fault_score_mean",
      "fault_score_median", "fault_score_p95", "first_step",
      "threshold_detected_count", "threshold_recall"]
  bug_type_agg_fields = [
      "variant", "split", "bug_type", "num_seeds", "event_count_total"]
  for metric in (
      "event_count", "events_per_10k", "fault_score_mean",
      "fault_score_median", "fault_score_p95", "first_step",
      "threshold_recall"):
    bug_type_agg_fields += [f"{metric}_mean", f"{metric}_std"]
  write_csv(outdir / "bug_type_per_run.csv", bug_type_rows, bug_type_fields)
  write_csv(outdir / "bug_type_aggregate.csv", bug_type_agg, bug_type_agg_fields)
  write_json(outdir / "analysis_summary.json", {
      "root": str(root),
      "num_cases": len(rows),
      "num_aggregate_rows": len(agg_rows),
      "baseline": args.baseline,
      "seeds": args.seeds,
      "pareto_rows": [
          {"variant": r["variant"], "split": r["split"], "objective_score": r.get("objective_score")}
          for r in fronts
      ],
  })

  setup_style()
  formats = [x.strip() for x in args.formats.split(",") if x.strip()]
  written = []
  written += plot_metric_grid(agg_rows, outdir, formats, args.error_bars)
  written += plot_tradeoffs(agg_rows, outdir, formats)
  written += plot_context_metrics(agg_rows, outdir, formats, args.error_bars)
  written += plot_bar_dashboard(
      agg_rows, outdir, formats, [
          ("unique_bug_types", "Unique bug types"),
          ("bug_type_coverage_fraction", "Expected type coverage"),
          ("bug_discovery_auc_norm", "Bug discovery AUC"),
          ("time_to_first_bug_steps", "Steps to first bug"),
          ("bug_events_per_10k", "Bug events / 10k steps"),
      ], "craftax_bug_discovery_metrics", error_bars=args.error_bars)
  written += plot_bar_dashboard(
      agg_rows, outdir, formats, [
          ("auprc_lift", "AUPRC lift over prevalence"),
          ("precision_at_top1pct", "Precision at top 1%"),
          ("threshold_bug_recall", "Threshold bug recall"),
          ("threshold_false_positive_rate", "Threshold false-positive rate"),
          ("fault_score_bug_normal_gap", "Bug-normal score gap"),
      ], "craftax_detection_quality_metrics", error_bars=args.error_bars)
  written += plot_bar_dashboard(
      agg_rows, outdir, formats, [
          ("constraint_lambda_mean", "Constraint lambda"),
          ("constraint_violation_mean", "Mean constraint violation"),
          ("task_constraint_feasible_rate", "Constraint feasible rate"),
          ("constraint_active_rate", "Constraint active rate"),
      ], "craftax_constraint_metrics", error_bars=args.error_bars)
  written += plot_generalization_gaps(agg_rows, outdir, formats)
  written += plot_bug_type_heatmaps(bug_type_agg, outdir, formats)
  write_markdown(outdir / "interpretation.md", agg_rows, fronts)
  write_metric_definitions(outdir / "metric_definitions.md")

  print(f"Wrote analysis to: {outdir}")
  print(f"- {outdir / 'per_run_metrics.csv'}")
  print(f"- {outdir / 'aggregate_metrics.csv'}")
  print(f"- {outdir / 'pareto_front.csv'}")
  print(f"- {outdir / 'bug_type_per_run.csv'}")
  print(f"- {outdir / 'bug_type_aggregate.csv'}")
  print(f"- {outdir / 'interpretation.md'}")
  print(f"- {outdir / 'metric_definitions.md'}")
  for path in written:
    print(f"- {path}")


if __name__ == "__main__":
  main()

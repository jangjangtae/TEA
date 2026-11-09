#!/usr/bin/env python3
"""Create a compact five-method Craftax main-results figure.

This combines the current main baselines and reward-design ablations before
the RND baseline finishes:

  TaskOnly, ScratchDreamer, DenseSurprise, ExcessDelta, ContextualExcess.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ABLATION = (
    "/home/railab/logdir/craftax_aaai_excess_ablation_20260709_161930/"
    "analysis/milestone_1000000/per_run_metrics.csv")
DEFAULT_SCRATCH = (
    "/home/railab/logdir/craftax_bugonly_totalbudget_20260710_155846/"
    "full/analysis/milestone_2100000/per_run_metrics.csv")

SPLITS = ["seen", "holdout", "sparse"]

METHODS = [
    ("TaskOnly", "Task-only", "#7b61b3"),
    ("ScratchDreamer", "ScratchDreamer", "#8f8f8f"),
    ("DenseSurprise", "Dense surprise", "#d98c1f"),
    ("ExcessDelta", "ExcessDelta (ours)", "#2f7ed8"),
    ("ContextualExcess", "Contextual excess", "#45a05a"),
]

VARIANT_TO_METHOD = {
    "taskonly": "TaskOnly",
    "dense_beta02": "DenseSurprise",
    "excess_delta_p95_beta02": "ExcessDelta",
    "contextual_excess_delta_beta02": "ContextualExcess",
    "bugonly_from_scratch": "ScratchDreamer",
}

METRICS = [
    ("episode_score_mean", "Task Episode Return", "Return", False),
    ("bug_events_per_10k", "Bug Events", "Events / 10k steps", False),
    ("bug_type_coverage_fraction", "Bug-Type Coverage", "Coverage", False),
    ("time_to_first_bug_steps", "Time to First Bug", "Steps", True),
]

EXTENDED_METRICS = [
    ("episode_score_mean", "Task Episode Return", "Return", False),
    ("bug_events_per_10k", "Bug Events", "Events / 10k steps", False),
    ("bug_type_coverage_fraction", "Bug-Type Coverage", "Coverage", False),
    ("bug_discovery_auc_norm", "Discovery AUC", "Norm. cumulative discovery", False),
    ("time_to_first_bug_steps", "Time to First Bug", "Steps", True),
    ("auroc", "Fault-Score Ranking", "AUROC", False),
]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ablation-csv", default=DEFAULT_ABLATION)
  parser.add_argument("--scratch-csv", default=DEFAULT_SCRATCH)
  parser.add_argument("--outdir", required=True)
  parser.add_argument("--formats", default="png,pdf")
  parser.add_argument("--title", default="")
  parser.add_argument("--legend-cols", type=int, default=5)
  parser.add_argument(
      "--extended", action="store_true",
      help="Use a 2x3 panel layout with discovery AUC and fault-score AUROC.")
  return parser.parse_args()


def load_data(ablation_csv, scratch_csv):
  ablation = pd.read_csv(ablation_csv)
  scratch = pd.read_csv(scratch_csv)
  data = pd.concat([ablation, scratch], ignore_index=True)
  data = data[data["phase"].eq("eval") & data["split"].isin(SPLITS)].copy()
  data = data[data["variant"].isin(VARIANT_TO_METHOD)].copy()
  data["method"] = data["variant"].map(VARIANT_TO_METHOD)
  data["split"] = pd.Categorical(data["split"], categories=SPLITS, ordered=True)
  return data


def summarize(data, metric):
  rows = []
  for (method, split), group in data.groupby(["method", "split"], observed=True):
    values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
    rows.append({
        "method": method,
        "split": split,
        "mean": float(values.mean()) if values.size else np.nan,
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "count": int(values.size),
    })
  return pd.DataFrame(rows)


def draw_metric(ax, data, metric, title, ylabel, lower_is_better):
  summary = summarize(data, metric)
  centers = np.arange(len(SPLITS), dtype=np.float64)
  width = 0.135
  offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width

  for idx, (method, label, color) in enumerate(METHODS):
    means, stds = [], []
    for split in SPLITS:
      row = summary[
          summary["method"].eq(method) & summary["split"].astype(str).eq(split)]
      means.append(float(row.iloc[0]["mean"]) if not row.empty else np.nan)
      stds.append(float(row.iloc[0]["std"]) if not row.empty else 0.0)

    xs = centers + offsets[idx]
    ax.bar(
        xs, means, width=width, yerr=stds, capsize=2.5,
        color=color, edgecolor="white", linewidth=0.7, label=label)

    for split_idx, split in enumerate(SPLITS):
      vals = pd.to_numeric(
          data[data["method"].eq(method) & data["split"].astype(str).eq(split)][metric],
          errors="coerce").dropna().to_numpy()
      if vals.size == 0:
        continue
      jitter = np.linspace(-width * 0.24, width * 0.24, vals.size)
      ax.scatter(
          np.full(vals.size, xs[split_idx]) + jitter, vals,
          s=15, color=color, edgecolor="white", linewidth=0.4, zorder=3)

  ax.set_title(title)
  ax.set_xticks(centers)
  ax.set_xticklabels([split.capitalize() for split in SPLITS])
  ax.set_ylabel(ylabel)
  ax.grid(axis="y", alpha=0.25)
  if metric != "episode_score_mean":
    ax.set_ylim(bottom=0)
  if metric == "auroc":
    ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1)
  if lower_is_better:
    ax.text(
        0.03, 0.93, "Lower is better",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, color="#555",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0))


def savefig(fig, outdir, stem, formats):
  paths = []
  for fmt in formats:
    path = outdir / f"{stem}.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    paths.append(path)
  plt.close(fig)
  return paths


def plot(data, outdir, formats, title, legend_cols, extended=False):
  metrics = EXTENDED_METRICS if extended else METRICS
  if extended:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 6.8), squeeze=False)
    stem = "craftax_main_five_methods_extended"
  else:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.9), squeeze=False)
    stem = "craftax_main_five_methods"
  for ax, (metric, panel_title, ylabel, lower) in zip(axes.ravel(), metrics):
    draw_metric(ax, data, metric, panel_title, ylabel, lower)

  handles, labels = axes[0][0].get_legend_handles_labels()
  fig.legend(
      handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
      ncol=legend_cols, frameon=False)
  if title:
    fig.suptitle(title, y=1.035, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
  else:
    fig.tight_layout(rect=(0, 0, 1, 0.94))
  return savefig(fig, outdir, stem, formats)


def main():
  args = parse_args()
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

  data = load_data(args.ablation_csv, args.scratch_csv)
  data.to_csv(outdir / "craftax_main_five_methods_per_seed.csv", index=False)

  summaries = []
  all_metrics = list(dict.fromkeys([m[0] for m in METRICS + EXTENDED_METRICS]))
  for metric in all_metrics:
    part = summarize(data, metric)
    part.insert(0, "metric", metric)
    summaries.append(part)
  pd.concat(summaries, ignore_index=True).to_csv(
      outdir / "craftax_main_five_methods_summary.csv", index=False)

  paths = plot(data, outdir, formats, args.title, args.legend_cols, args.extended)
  print("Wrote:")
  for path in paths:
    print(path)
  print(outdir / "craftax_main_five_methods_per_seed.csv")
  print(outdir / "craftax_main_five_methods_summary.csv")


if __name__ == "__main__":
  main()

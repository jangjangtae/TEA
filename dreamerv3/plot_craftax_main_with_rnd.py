#!/usr/bin/env python3
"""Create the Craftax main-results figure including the RND baseline."""

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
DEFAULT_RND = (
    "/home/railab/logdir/craftax_rnd_baseline_full_20260714_130703/"
    "full/analysis/milestone_1000000/per_run_metrics.csv")
DEFAULT_CLEANEVAL = ""

SPLITS = ["seen", "holdout", "sparse"]

METHODS = [
    ("CleanEval", "No-adapt clean", "#BBBBBB"),
    # Okabe-Ito inspired, colorblind-friendly palette. Keep the proposed
    # method as the most saturated blue and reserve gray for scratch training.
    ("TaskOnly", "Task-only", "#CC79A7"),
    ("ScratchDreamer", "ScratchDreamer", "#666666"),
    ("DreamerRND", "Dreamer+RND", "#E69F00"),
    ("DenseSurprise", "Dense surprise", "#D55E00"),
    ("ExcessDelta", "FATE (ours)", "#0072B2"),
    ("ContextualExcess", "Contextual excess", "#009E73"),
]

VARIANT_TO_METHOD = {
    "reference": "CleanEval",
    "taskonly": "TaskOnly",
    "bugonly_from_scratch": "ScratchDreamer",
    "rnd_beta005": "DreamerRND",
    "dense_beta02": "DenseSurprise",
    "excess_delta_p95_beta02": "ExcessDelta",
    "contextual_excess_delta_beta02": "ContextualExcess",
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
  parser.add_argument("--rnd-csv", default=DEFAULT_RND)
  parser.add_argument(
      "--cleaneval-csv", default=DEFAULT_CLEANEVAL,
      help="Optional CleanEval per_run_metrics.csv with variant=reference.")
  parser.add_argument("--outdir", required=True)
  parser.add_argument("--formats", default="png,pdf")
  parser.add_argument("--legend-cols", type=int, default=6)
  parser.add_argument(
      "--error-bars", choices=("sem", "std", "none"), default="sem",
      help="Error bars over seeds. SEM is cleaner for paper figures.")
  parser.add_argument("--extended", action="store_true")
  return parser.parse_args()


def load_csvs(spec):
  frames = []
  for item in str(spec).split(","):
    item = item.strip()
    if not item:
      continue
    path = Path(item)
    if not path.exists():
      raise FileNotFoundError(path)
    frames.append(pd.read_csv(path))
  if not frames:
    return pd.DataFrame()
  return pd.concat(frames, ignore_index=True)


def load_data(args):
  frames = [
      load_csvs(args.ablation_csv),
      load_csvs(args.scratch_csv),
      load_csvs(args.rnd_csv),
  ]
  if args.cleaneval_csv:
    frames.append(load_csvs(args.cleaneval_csv))
  data = pd.concat(frames, ignore_index=True)
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
        "sem": float(values.std(ddof=1) / np.sqrt(values.size))
        if values.size > 1 else 0.0,
        "count": int(values.size),
    })
  return pd.DataFrame(rows)


def draw_metric(ax, data, metric, title, ylabel, lower_is_better, error_bars):
  summary = summarize(data, metric)
  centers = np.arange(len(SPLITS), dtype=np.float64)
  width = 0.115
  offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width

  for idx, (method, label, color) in enumerate(METHODS):
    means, errs = [], []
    for split in SPLITS:
      row = summary[
          summary["method"].eq(method) & summary["split"].astype(str).eq(split)]
      means.append(float(row.iloc[0]["mean"]) if not row.empty else np.nan)
      if row.empty or error_bars == "none":
        errs.append(0.0)
      else:
        errs.append(float(row.iloc[0][error_bars]))

    xs = centers + offsets[idx]
    ax.bar(
        xs, means, width=width, yerr=errs, capsize=2.2,
        color=color, edgecolor="white", linewidth=0.7, label=label)

    for split_idx, split in enumerate(SPLITS):
      vals = pd.to_numeric(
          data[data["method"].eq(method) & data["split"].astype(str).eq(split)][metric],
          errors="coerce").dropna().to_numpy()
      if vals.size == 0:
        continue
      jitter = np.linspace(-width * 0.25, width * 0.25, vals.size)
      ax.scatter(
          np.full(vals.size, xs[split_idx]) + jitter, vals,
          s=14, color=color, edgecolor="white", linewidth=0.35, zorder=3)

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


def plot(data, outdir, formats, legend_cols, extended, error_bars):
  metrics = EXTENDED_METRICS if extended else METRICS
  if extended:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.0), squeeze=False)
    stem = "craftax_main_with_rnd_extended"
  else:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.0), squeeze=False)
    stem = "craftax_main_with_rnd"

  for ax, (metric, title, ylabel, lower) in zip(axes.ravel(), metrics):
    draw_metric(ax, data, metric, title, ylabel, lower, error_bars)

  handles, labels = axes[0][0].get_legend_handles_labels()
  fig.legend(
      handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
      ncol=legend_cols, frameon=False)
  fig.tight_layout(rect=(0, 0, 1, 0.935))
  return savefig(fig, outdir, stem, formats)


def main():
  args = parse_args()
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

  data = load_data(args)
  data.to_csv(outdir / "craftax_main_with_rnd_per_seed.csv", index=False)

  summaries = []
  all_metrics = list(dict.fromkeys([m[0] for m in METRICS + EXTENDED_METRICS]))
  for metric in all_metrics:
    part = summarize(data, metric)
    part.insert(0, "metric", metric)
    summaries.append(part)
  pd.concat(summaries, ignore_index=True).to_csv(
      outdir / "craftax_main_with_rnd_summary.csv", index=False)

  paths = plot(
      data, outdir, formats, args.legend_cols, args.extended, args.error_bars)
  print("Wrote:")
  for path in paths:
    print(path)
  print(outdir / "craftax_main_with_rnd_per_seed.csv")
  print(outdir / "craftax_main_with_rnd_summary.csv")


if __name__ == "__main__":
  main()

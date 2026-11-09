#!/usr/bin/env python3
"""Plot ScratchDreamer vs clean-prior two-model adaptation.

This figure is meant to support the architectural choice: instead of training
one Dreamer directly in the faulty environment, keep a clean reference world
model and adapt a separate active Dreamer with calibrated excess surprise.
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

METHODS_FOCUSED = [
    ("ScratchDreamer", "ScratchDreamer", "#9a9a9a"),
    ("ExcessDelta", "ExcessDelta (ours)", "#2f7ed8"),
]

METHODS_WITH_CONTROL = [
    ("TaskOnly", "Task-only", "#7b61b3"),
    ("ScratchDreamer", "ScratchDreamer", "#9a9a9a"),
    ("ExcessDelta", "ExcessDelta (ours)", "#2f7ed8"),
]

METRICS = [
    ("episode_score_mean", "Task Episode Return", "Return", False),
    ("bug_events_per_10k", "Bug Events", "Events / 10k steps", False),
    ("bug_type_coverage_fraction", "Bug-Type Coverage", "Coverage", False),
    ("time_to_first_bug_steps", "Time to First Bug", "Steps", True),
]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ablation-csv", default=DEFAULT_ABLATION)
  parser.add_argument("--scratch-csv", default=DEFAULT_SCRATCH)
  parser.add_argument("--outdir", required=True)
  parser.add_argument("--formats", default="png,pdf")
  parser.add_argument("--paper-style", action="store_true")
  return parser.parse_args()


def load_rows(ablation_csv, scratch_csv):
  ablation = pd.read_csv(ablation_csv)
  scratch = pd.read_csv(scratch_csv)

  parts = []
  task = ablation[ablation["variant"].eq("taskonly")].copy()
  task["method"] = "TaskOnly"
  parts.append(task)

  ours = ablation[ablation["variant"].eq("excess_delta_p95_beta02")].copy()
  ours["method"] = "ExcessDelta"
  parts.append(ours)

  single = scratch[scratch["variant"].eq("bugonly_from_scratch")].copy()
  single["method"] = "ScratchDreamer"
  parts.append(single)

  data = pd.concat(parts, ignore_index=True)
  data = data[data["phase"].eq("eval") & data["split"].isin(SPLITS)].copy()
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


def draw_panel(ax, data, methods, metric, title, ylabel, lower_is_better):
  summary = summarize(data, metric)
  centers = np.arange(len(SPLITS), dtype=np.float64)
  width = min(0.26, 0.75 / len(methods))
  offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

  for idx, (method, label, color) in enumerate(methods):
    means, stds = [], []
    for split in SPLITS:
      row = summary[
          summary["method"].eq(method) & summary["split"].astype(str).eq(split)]
      if row.empty:
        means.append(np.nan)
        stds.append(0.0)
      else:
        means.append(float(row.iloc[0]["mean"]))
        stds.append(float(row.iloc[0]["std"]))
    xs = centers + offsets[idx]
    ax.bar(
        xs, means, width=width, yerr=stds, capsize=3,
        color=color, edgecolor="white", linewidth=0.8, label=label)

    for split_idx, split in enumerate(SPLITS):
      values = pd.to_numeric(
          data[data["method"].eq(method) & data["split"].astype(str).eq(split)][metric],
          errors="coerce").dropna().to_numpy()
      if not values.size:
        continue
      jitter = np.linspace(-width * 0.22, width * 0.22, values.size)
      ax.scatter(
          np.full(values.size, xs[split_idx]) + jitter, values,
          s=18, color=color, edgecolor="white", linewidth=0.45, zorder=3)

  ax.set_title(title)
  ax.set_xticks(centers)
  ax.set_xticklabels([split.capitalize() for split in SPLITS])
  ax.set_ylabel(ylabel)
  ax.grid(axis="y", alpha=0.25)
  if metric in ("bug_events_per_10k", "bug_type_coverage_fraction", "time_to_first_bug_steps"):
    ax.set_ylim(bottom=0)
  if lower_is_better:
    ax.text(
        0.03, 0.93, "Lower is better",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, color="#555",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0))


def plot(data, methods, outdir, stem, formats, paper_style=False):
  fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.8), squeeze=False)
  for ax, (metric, title, ylabel, lower) in zip(axes.ravel(), METRICS):
    draw_panel(ax, data, methods, metric, title, ylabel, lower)

  handles, labels = axes[0][0].get_legend_handles_labels()
  fig.legend(
      handles, labels, loc="upper center",
      bbox_to_anchor=(0.5, 0.99 if paper_style else 0.965),
      ncol=len(methods), frameon=False)
  if not paper_style:
    fig.suptitle(
        "Single Dreamer vs Clean-Prior Adaptation (3 seeds)",
        y=1.005, fontsize=13)
    fig.text(
        0.5, 0.012,
        "ScratchDreamer: random init in faulty env for 2.1M steps. "
        "ExcessDelta: clean pretraining + 1.0M fault adaptation with frozen clean reference.",
        ha="center", va="bottom", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
  else:
    fig.tight_layout(rect=(0, 0, 1, 0.95))

  paths = []
  for fmt in formats:
    path = outdir / f"{stem}.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    paths.append(path)
  plt.close(fig)
  return paths


def main():
  args = parse_args()
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

  data = load_rows(args.ablation_csv, args.scratch_csv)
  data.to_csv(outdir / "single_vs_two_model_per_seed.csv", index=False)

  summary_parts = []
  for metric, _, _, _ in METRICS:
    part = summarize(data, metric)
    part.insert(0, "metric", metric)
    summary_parts.append(part)
  pd.concat(summary_parts, ignore_index=True).to_csv(
      outdir / "single_vs_two_model_summary.csv", index=False)

  paths = []
  paths.extend(plot(
      data, METHODS_FOCUSED, outdir,
      "single_vs_two_model_core", formats, args.paper_style))
  paths.extend(plot(
      data, METHODS_WITH_CONTROL, outdir,
      "single_vs_two_model_with_taskonly", formats, args.paper_style))

  print("Wrote:")
  for path in paths:
    print(path)
  print(outdir / "single_vs_two_model_per_seed.csv")
  print(outdir / "single_vs_two_model_summary.csv")


if __name__ == "__main__":
  main()

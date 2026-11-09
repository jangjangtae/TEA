#!/usr/bin/env python3
"""Create compact Craftax figures for paper main text.

This script consumes the analysis outputs produced by
analyze_craftax_multiseed.py and analyze_craftax_milestones.py.
It intentionally keeps only a small set of metrics that support the main
claim: task competence, bug discovery, discovery efficiency, and detection
quality.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_VARIANTS = [
    "taskonly",
    "dense_beta02",
    "excess_delta_p95_beta02",
    "contextual_excess_delta_beta02",
]

LABELS = {
    "taskonly": "Task-only",
    "dense_beta02": "Dense surprise",
    "excess_delta_p95_beta02": "Excess-delta (ours)",
    "contextual_excess_delta_beta02": "Contextual excess",
    "rnd_beta005": "Dreamer+RND",
    "rnd_beta01": "Dreamer+RND x2",
    "rnd_beta02": "Dreamer+RND x4",
}

COLORS = {
    "taskonly": "#7b61b3",
    "dense_beta02": "#9a9a9a",
    "excess_delta_p95_beta02": "#2f7ed8",
    "contextual_excess_delta_beta02": "#45a05a",
    "rnd_beta005": "#d98c1f",
    "rnd_beta01": "#c26a1a",
    "rnd_beta02": "#a94e15",
}

SPLITS = ["seen", "holdout", "sparse"]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--analysis-root", required=True)
  parser.add_argument("--milestone", type=int, default=1000000)
  parser.add_argument("--outdir", default="")
  parser.add_argument("--formats", default="png,pdf")
  parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
  parser.add_argument(
      "--curve-variants", nargs="*",
      default=["taskonly", "excess_delta_p95_beta02",
               "contextual_excess_delta_beta02"])
  parser.add_argument(
      "--no-seed-dots", action="store_true",
      help="Hide per-seed dots on the final outcome bar plot.")
  return parser.parse_args()


def read_csv(path):
  with Path(path).open(newline="") as f:
    return list(csv.DictReader(f))


def number(value, default=float("nan")):
  try:
    out = float(value)
  except Exception:
    return default
  return out


def finite(value):
  return math.isfinite(value)


def metric(row, key):
  return number(row.get(key, "nan"))


def savefig(fig, outdir, stem, formats):
  paths = []
  for fmt in formats:
    path = outdir / f"{stem}.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    paths.append(path)
  plt.close(fig)
  return paths


def final_row_index(rows):
  return {
      (row["variant"], row["split"]): row
      for row in rows
      if row.get("phase") == "eval" and row.get("split") in SPLITS
  }


def per_run_index(rows):
  index = {}
  for row in rows:
    if row.get("phase") != "eval" or row.get("split") not in SPLITS:
      continue
    index.setdefault((row["variant"], row["split"]), []).append(row)
  return index


def per_run_metric_key(metric_key):
  if metric_key.endswith("_mean_mean"):
    return metric_key[:-len("_mean_mean")] + "_mean"
  if metric_key.endswith("_mean"):
    return metric_key[:-len("_mean")]
  return metric_key


def grouped_bars(
    ax, index, variants, metric_key, std_key, title, ylabel,
    per_index=None, lower_is_better=False, ylim_bottom=None,
    show_seed_dots=True):
  width = 0.18
  centers = np.arange(len(SPLITS))
  offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2.0) * width
  for i, variant in enumerate(variants):
    ys, es = [], []
    for split in SPLITS:
      row = index.get((variant, split), {})
      ys.append(metric(row, metric_key))
      es.append(metric(row, std_key))
    ax.bar(
        centers + offsets[i], ys, width=width,
        yerr=es, capsize=2.5,
        label=LABELS.get(variant, variant),
        color=COLORS.get(variant, None),
        edgecolor="white", linewidth=0.7)
    if show_seed_dots and per_index:
      dot_key = per_run_metric_key(metric_key)
      for split_i, split in enumerate(SPLITS):
        seed_rows = per_index.get((variant, split), [])
        vals = [metric(row, dot_key) for row in seed_rows]
        vals = [v for v in vals if finite(v)]
        if not vals:
          continue
        jitter = np.linspace(-width * 0.22, width * 0.22, len(vals))
        xs = centers[split_i] + offsets[i] + jitter
        ax.scatter(
            xs, vals, s=16, marker="o",
            facecolor=COLORS.get(variant, "#333333"),
            edgecolor="white", linewidth=0.45, alpha=0.9, zorder=3)
  ax.set_title(title)
  ax.set_xticks(centers)
  ax.set_xticklabels([s.capitalize() for s in SPLITS])
  ax.set_ylabel(ylabel)
  ax.grid(axis="y", alpha=0.25)
  if ylim_bottom is not None:
    ax.set_ylim(bottom=ylim_bottom)
  if lower_is_better:
    ax.text(
        0.04, 0.92, "Lower is better",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, color="#555555",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0))


def plot_final_summary(
    rows, per_rows, outdir, formats, variants, show_seed_dots=True):
  index = final_row_index(rows)
  per_index = per_run_index(per_rows)
  fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.8), squeeze=False)
  grouped_bars(
      axes[0][0], index, variants,
      "episode_score_mean_mean", "episode_score_mean_std",
      "Game Competence", "Task episode return",
      per_index=per_index, show_seed_dots=show_seed_dots)
  grouped_bars(
      axes[0][1], index, variants,
      "bug_events_per_10k_mean", "bug_events_per_10k_std",
      "Bug Events", "Events / 10k steps",
      per_index=per_index, show_seed_dots=show_seed_dots,
      ylim_bottom=0.0)
  grouped_bars(
      axes[1][0], index, variants,
      "bug_type_coverage_fraction_mean", "bug_type_coverage_fraction_std",
      "Bug-Type Coverage", "Bug-type coverage",
      per_index=per_index, show_seed_dots=show_seed_dots,
      ylim_bottom=0.0)
  grouped_bars(
      axes[1][1], index, variants,
      "time_to_first_bug_steps_mean", "time_to_first_bug_steps_std",
      "Time to First Bug", "Steps to first bug",
      per_index=per_index, lower_is_better=True,
      ylim_bottom=0.0, show_seed_dots=show_seed_dots)
  handles, labels = axes[0][0].get_legend_handles_labels()
  fig.legend(
      handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
      ncol=max(1, min(len(variants), 4)), frameon=False)
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  return savefig(fig, outdir, "paper_core_final_outcomes", formats)


def plot_detection_summary(rows, outdir, formats, variants):
  index = final_row_index(rows)
  fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.4), squeeze=False)
  grouped_bars(
      axes[0][0], index, variants,
      "auroc_mean", "auroc_std",
      "Fault Score Ranking", "AUROC", ylim_bottom=0.0)
  axes[0][0].axhline(0.5, color="#555555", linestyle=":", linewidth=1)
  grouped_bars(
      axes[0][1], index, variants,
      "auprc_mean", "auprc_std",
      "Fault Score Precision", "AUPRC", ylim_bottom=0.0)
  axes[0][1].legend(
      loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
  fig.suptitle("Fault-Score Detection Quality", y=1.05)
  fig.tight_layout()
  return savefig(fig, outdir, "paper_core_detection_quality", formats)


def plot_learning_curves(rows, outdir, formats, variants):
  rows = [
      r for r in rows
      if r.get("phase") == "eval" and r.get("split") in SPLITS and
      r.get("variant") in variants]
  if not rows:
    return []
  metrics = [
      ("episode_score_mean_mean", "Task episode return", False),
      ("bug_events_per_10k_mean", "Bug events / 10k", False),
      ("bug_type_coverage_fraction_mean", "Bug-type coverage", False),
      ("time_to_first_bug_steps_mean", "Steps to first bug", True),
  ]
  fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), squeeze=False)
  for ax, (key, title, lower) in zip(axes.ravel(), metrics):
    for variant in variants:
      xs, ys = [], []
      for milestone in sorted({int(number(r["milestone"], 0)) for r in rows}):
        vals = [
            metric(r, key)
            for r in rows
            if r["variant"] == variant and int(number(r["milestone"], 0)) == milestone
        ]
        vals = [v for v in vals if finite(v)]
        if vals:
          xs.append(milestone / 1000.0)
          ys.append(float(np.mean(vals)))
      if xs:
        ax.plot(
            xs, ys, marker="o", linewidth=2,
            color=COLORS.get(variant),
            label=LABELS.get(variant, variant))
    ax.set_title(title)
    ax.set_xlabel("Adaptation steps (thousands)")
    ax.grid(alpha=0.25)
    if lower:
      ax.set_ylim(bottom=0.0)
  axes[0][1].legend(
      loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
  fig.suptitle("Learning Dynamics Across Milestones", y=1.02)
  fig.tight_layout()
  return savefig(fig, outdir, "paper_core_milestone_curves", formats)


def write_summary(rows, outdir, variants):
  index = final_row_index(rows)
  lines = [
      "# Paper Core Figure Summary",
      "",
      "Recommended main-text figures:",
      "",
      "- `paper_core_final_outcomes`: competence, bug rate, type coverage, and time-to-first-bug.",
      "- Final outcome bars show mean +/- standard deviation; dots show individual seeds.",
      "- `paper_core_milestone_curves`: whether adaptation keeps improving after 200k/400k/600k/800k.",
      "- `paper_core_detection_quality`: optional main text or appendix if space is tight.",
      "",
      "Final-step values:",
      "",
  ]
  for split in SPLITS:
    lines.extend([
        f"## {split}",
        "",
        "| Method | Task episode return | Bug events / 10k | Bug-type coverage | Steps to first bug | AUROC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in variants:
      row = index.get((variant, split), {})
      lines.append(
          f"| {LABELS.get(variant, variant)} | "
          f"{metric(row, 'episode_score_mean_mean'):.3f} | "
          f"{metric(row, 'bug_events_per_10k_mean'):.3f} | "
          f"{metric(row, 'bug_type_coverage_fraction_mean'):.3f} | "
          f"{metric(row, 'time_to_first_bug_steps_mean'):.1f} | "
          f"{metric(row, 'auroc_mean'):.3f} | "
          f"{metric(row, 'auprc_mean'):.4f} |")
    lines.append("")
  (outdir / "paper_core_summary.md").write_text(
      "\n".join(lines), encoding="utf-8")


def main():
  args = parse_args()
  analysis_root = Path(args.analysis_root)
  outdir = Path(args.outdir) if args.outdir else analysis_root / "paper_figures"
  outdir.mkdir(parents=True, exist_ok=True)
  formats = [x.strip() for x in args.formats.split(",") if x.strip()]

  final_csv = analysis_root / f"milestone_{args.milestone}" / "aggregate_metrics.csv"
  final_per_run_csv = analysis_root / f"milestone_{args.milestone}" / "per_run_metrics.csv"
  milestone_csv = analysis_root / "milestones" / "milestone_metrics.csv"
  final_rows = read_csv(final_csv)
  final_per_rows = read_csv(final_per_run_csv) if final_per_run_csv.exists() else []
  milestone_rows = read_csv(milestone_csv)

  written = []
  written += plot_final_summary(
      final_rows, final_per_rows, outdir, formats, args.variants,
      show_seed_dots=not args.no_seed_dots)
  written += plot_detection_summary(final_rows, outdir, formats, args.variants)
  written += plot_learning_curves(
      milestone_rows, outdir, formats, args.curve_variants)
  write_summary(final_rows, outdir, args.variants)
  written.append(outdir / "paper_core_summary.md")

  print("Wrote paper core figures:")
  for path in written:
    print(f"- {path}")


if __name__ == "__main__":
  main()

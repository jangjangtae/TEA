#!/usr/bin/env python3
"""Combine Craftax milestone analyses into learning-curve artifacts."""

import argparse
import csv
import math
import os
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


METRICS = (
    ("episode_score_mean_mean", "Episode score"),
    ("fault_applied_rate_mean", "Bug manifestation rate"),
    ("auroc_mean", "Fault-score AUROC"),
    ("auprc_mean", "Fault-score AUPRC"),
    ("auprc_lift_mean", "AUPRC lift over prevalence"),
    ("unique_bug_types_mean", "Unique bug types"),
    ("bug_type_coverage_fraction_mean", "Expected bug-type coverage"),
    ("bug_discovery_auc_norm_mean", "Bug discovery AUC"),
    ("time_to_first_bug_steps_mean", "Steps to first bug"),
    ("bug_events_per_10k_mean", "Bug events per 10k steps"),
    ("semantic_context_coverage_per_1k_mean", "Semantic contexts per 1k steps"),
    ("threshold_bug_recall_mean", "Threshold bug recall"),
    ("threshold_false_positive_rate_mean", "Threshold false-positive rate"),
    ("constraint_lambda_mean_mean", "Constraint lambda"),
    ("task_constraint_feasible_rate_mean", "Constraint feasible rate"),
)
SPLITS = ("clean", "seen", "holdout", "sparse")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", required=True)
  parser.add_argument("--milestones", type=int, nargs="+", required=True)
  parser.add_argument(
      "--analysis-root", default="",
      help="Directory containing milestone_<step>/aggregate_metrics.csv. "
      "Defaults to <root>/analysis.")
  parser.add_argument(
      "--error-bars", choices=("std", "none"), default="none",
      help="Draw milestone curve error bars from aggregate seed std columns.")
  parser.add_argument("--outdir", default="")
  return parser.parse_args()


def number(value):
  try:
    value = float(value)
  except (TypeError, ValueError):
    return float("nan")
  return value if math.isfinite(value) else float("nan")


def std_key(metric):
  if metric.endswith("_mean_mean"):
    return metric[:-len("_mean_mean")] + "_mean_std"
  if metric.endswith("_mean"):
    return metric[:-len("_mean")] + "_std"
  return metric + "_std"


def read_rows(analysis_root, milestones):
  rows = []
  for milestone in milestones:
    path = analysis_root / f"milestone_{milestone}" / "aggregate_metrics.csv"
    if not path.exists():
      raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
      for row in csv.DictReader(f):
        if row.get("phase") != "eval":
          continue
        row["milestone"] = milestone
        rows.append(row)
  return rows


def write_csv(path, rows, fields):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def plot_curves(rows, outdir, milestones, error_bars="none"):
  variants = sorted({row["variant"] for row in rows if row["variant"] != "reference"})
  colors = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(variants)}
  plt.rcParams.update({
      "figure.dpi": 140,
      "savefig.dpi": 220,
      "axes.grid": True,
      "grid.alpha": 0.25,
      "axes.spines.top": False,
      "axes.spines.right": False,
  })
  for metric, title in METRICS:
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(17, 4.5), sharex=True)
    for ax, split in zip(axes, SPLITS):
      for variant in variants:
        points = {
            int(row["milestone"]): number(row.get(metric))
            for row in rows
            if row["split"] == split and row["variant"] == variant
        }
        xs = [step / 1000 for step in milestones if math.isfinite(points.get(step, float("nan")))]
        ys = [points[step] for step in milestones if math.isfinite(points.get(step, float("nan")))]
        if xs:
          if error_bars == "std":
            keyed = {
                int(row["milestone"]): row
                for row in rows
                if row["split"] == split and row["variant"] == variant
            }
            yerr = [
                number(keyed[step].get(std_key(metric)))
                for step in milestones
                if step in keyed and math.isfinite(points.get(step, float("nan")))
            ]
            ax.errorbar(
                xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=3,
                color=colors[variant], label=variant)
          else:
            ax.plot(xs, ys, marker="o", linewidth=2, color=colors[variant], label=variant)
      if metric == "auroc_mean":
        ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1)
      ax.set_title(split)
      ax.set_xlabel("Adaptation steps (thousands)")
      if ax is axes[0]:
        ax.set_ylabel(title)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
      fig.legend(handles, labels, loc="upper center", ncol=max(1, min(4, len(labels))))
    fig.suptitle(f"{title} across adaptation milestones", y=1.05)
    if error_bars == "std":
      fig.text(0.995, 0.995, "Error bars: SD across seeds", ha="right", va="top", fontsize=9)
    fig.tight_layout()
    stem = metric.replace("_mean_mean", "").replace("_mean", "")
    for ext in ("png", "pdf"):
      fig.savefig(outdir / f"milestone_{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(rows, outdir, milestones, error_bars="none"):
  selected = (
      ("episode_score_mean_mean", "Episode score"),
      ("unique_bug_types_mean", "Unique bug types"),
      ("bug_discovery_auc_norm_mean", "Bug discovery AUC"),
      ("time_to_first_bug_steps_mean", "Steps to first bug"),
      ("auroc_mean", "AUROC"),
      ("auprc_lift_mean", "AUPRC lift"),
      ("semantic_context_coverage_per_1k_mean", "Contexts / 1k"),
  )
  variants = sorted({row["variant"] for row in rows if row["variant"] != "reference"})
  splits = ("seen", "holdout", "sparse")
  colors = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(variants)}
  fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
  for ax, (metric, title) in zip(axes.flat, selected):
    for split, linestyle in zip(splits, ("-", "--", ":")):
      for variant in variants:
        points = {
            int(row["milestone"]): number(row.get(metric))
            for row in rows
            if row["split"] == split and row["variant"] == variant
        }
        xs = [step / 1000 for step in milestones if math.isfinite(points.get(step, float("nan")))]
        ys = [points[step] for step in milestones if math.isfinite(points.get(step, float("nan")))]
        if xs:
          if error_bars == "std":
            keyed = {
                int(row["milestone"]): row
                for row in rows
                if row["split"] == split and row["variant"] == variant
            }
            yerr = [
                number(keyed[step].get(std_key(metric)))
                for step in milestones
                if step in keyed and math.isfinite(points.get(step, float("nan")))
            ]
            ax.errorbar(
                xs, ys, yerr=yerr, marker="o", linewidth=1.8,
                linestyle=linestyle, capsize=2, color=colors[variant],
                label=f"{variant} / {split}")
          else:
            ax.plot(
                xs, ys, marker="o", linewidth=1.8, linestyle=linestyle,
                color=colors[variant], label=f"{variant} / {split}")
    if metric == "auroc_mean":
      ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Adaptation steps (thousands)")
  handles, labels = axes[0, 0].get_legend_handles_labels()
  if handles:
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
  fig.suptitle("Craftax tester learning dashboard", y=1.03)
  if error_bars == "std":
    fig.text(0.995, 0.995, "Error bars: SD across seeds", ha="right", va="top", fontsize=9)
  fig.tight_layout()
  for ext in ("png", "pdf"):
    fig.savefig(outdir / f"milestone_testing_dashboard.{ext}", bbox_inches="tight")
  plt.close(fig)


def interval_deltas(rows, milestones):
  if len(milestones) < 2:
    return []
  previous, final = milestones[-2:]
  indexed = {
      (int(row["milestone"]), row["variant"], row["split"]): row
      for row in rows
  }
  result = []
  cases = sorted({(row["variant"], row["split"]) for row in rows})
  for variant, split in cases:
    before = indexed.get((previous, variant, split))
    after = indexed.get((final, variant, split))
    if not before or not after:
      continue
    item = {
        "variant": variant,
        "split": split,
        "from_step": previous,
        "to_step": final,
    }
    for metric, _ in METRICS:
      item[f"{metric}_delta"] = number(after.get(metric)) - number(before.get(metric))
    result.append(item)
  return result


def write_summary(path, deltas, milestones):
  previous, final = milestones[-2:] if len(milestones) > 1 else (milestones[0], milestones[0])
  lines = [
      "# Craftax Milestone Summary",
      "",
      f"Final interval: {previous:,} -> {final:,} adaptation steps.",
      "Use this interval to judge whether extending beyond the penultimate milestone is still productive.",
      "",
      "| Variant | Split | Task delta | Bug-rate delta | Type delta | Discovery AUC delta | TTFB delta | AUROC delta | AUPRC delta |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in deltas:
    lines.append(
        f"| {row['variant']} | {row['split']} | "
        f"{row['episode_score_mean_mean_delta']:+.4f} | "
        f"{row['fault_applied_rate_mean_delta']:+.6f} | "
        f"{row['unique_bug_types_mean_delta']:+.3f} | "
        f"{row['bug_discovery_auc_norm_mean_delta']:+.3f} | "
        f"{row['time_to_first_bug_steps_mean_delta']:+.1f} | "
        f"{row['auroc_mean_delta']:+.4f} | "
        f"{row['auprc_mean_delta']:+.5f} |")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
  args = parse_args()
  root = Path(args.root).expanduser()
  analysis_root = (
      Path(args.analysis_root).expanduser() if args.analysis_root
      else root / "analysis")
  outdir = Path(args.outdir).expanduser() if args.outdir else analysis_root / "milestones"
  outdir.mkdir(parents=True, exist_ok=True)
  milestones = sorted(set(args.milestones))
  rows = read_rows(analysis_root, milestones)

  fields = ["milestone", "variant", "split", "phase", "num_seeds"]
  fields += [metric for metric, _ in METRICS]
  write_csv(outdir / "milestone_metrics.csv", rows, fields)
  plot_curves(rows, outdir, milestones, args.error_bars)
  plot_dashboard(rows, outdir, milestones, args.error_bars)

  deltas = interval_deltas(rows, milestones)
  delta_fields = ["variant", "split", "from_step", "to_step"]
  delta_fields += [f"{metric}_delta" for metric, _ in METRICS]
  write_csv(outdir / "last_interval_deltas.csv", deltas, delta_fields)
  write_summary(outdir / "summary.md", deltas, milestones)
  print(f"Wrote milestone analysis: {outdir}")


if __name__ == "__main__":
  main()

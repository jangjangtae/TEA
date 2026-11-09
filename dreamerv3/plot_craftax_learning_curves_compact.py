#!/usr/bin/env python3
"""Compact learning curves for Craftax reward-design variants.

The plot averages each seed across seen/holdout/sparse evaluations at each
milestone, then shows mean +/- std over seeds. This keeps the learning-curve
figure readable enough for appendix or a small main-paper panel.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = "/home/railab/logdir/craftax_aaai_excess_ablation_20260709_161930"
SPLITS = ["seen", "holdout", "sparse"]
METHODS = [
    ("taskonly", "Task-only", "#7b61b3"),
    ("dense_beta02", "Dense surprise", "#d98c1f"),
    ("excess_delta_p95_beta02", "ExcessDelta (ours)", "#2f7ed8"),
    ("contextual_excess_delta_beta02", "Contextual excess", "#45a05a"),
]
METRICS = [
    ("episode_score_mean", "Task Episode Return", "Return", False),
    ("bug_events_per_10k", "Bug Events", "Events / 10k steps", False),
    ("bug_type_coverage_fraction", "Bug-Type Coverage", "Coverage", False),
    ("bug_discovery_auc_norm", "Discovery AUC", "Norm. cumulative discovery", False),
]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", default=DEFAULT_ROOT)
  parser.add_argument("--outdir", required=True)
  parser.add_argument("--formats", default="png,pdf")
  return parser.parse_args()


def load_milestones(root):
  root = Path(root)
  parts = []
  for path in sorted((root / "analysis").glob("milestone_*/per_run_metrics.csv")):
    try:
      milestone = int(path.parent.name.split("_", 1)[1])
    except Exception:
      continue
    data = pd.read_csv(path)
    data["milestone"] = milestone
    parts.append(data)
  if not parts:
    raise FileNotFoundError(f"No milestone per_run_metrics.csv files under {root}")
  data = pd.concat(parts, ignore_index=True)
  data = data[
      data["phase"].eq("eval") &
      data["split"].isin(SPLITS) &
      data["variant"].isin([m[0] for m in METHODS])
  ].copy()
  return data


def aggregate_seed_split(data):
  rows = []
  for metric, _, _, _ in METRICS:
    values = data.copy()
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    per_seed = (
        values.groupby(["milestone", "variant", "seed"], observed=True)[metric]
        .mean()
        .reset_index(name="value"))
    for (milestone, variant), group in per_seed.groupby(
        ["milestone", "variant"], observed=True):
      vals = group["value"].dropna().to_numpy()
      rows.append({
          "metric": metric,
          "milestone": int(milestone),
          "variant": variant,
          "mean": float(vals.mean()) if vals.size else np.nan,
          "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
          "num_seeds": int(vals.size),
      })
  return pd.DataFrame(rows)


def plot(summary, outdir, formats):
  fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), squeeze=False)
  for ax, (metric, title, ylabel, lower) in zip(axes.ravel(), METRICS):
    subset = summary[summary["metric"].eq(metric)]
    for variant, label, color in METHODS:
      rows = subset[subset["variant"].eq(variant)].sort_values("milestone")
      if rows.empty:
        continue
      xs = rows["milestone"].to_numpy(dtype=np.float64) / 1000.0
      ys = rows["mean"].to_numpy(dtype=np.float64)
      es = rows["std"].to_numpy(dtype=np.float64)
      ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)
      ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.14, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Adaptation steps (thousands)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if not lower:
      ax.set_ylim(bottom=0)
  handles, labels = axes[0][0].get_legend_handles_labels()
  fig.legend(
      handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
      ncol=4, frameon=False)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  paths = []
  for fmt in formats:
    path = outdir / f"craftax_learning_curves_compact.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    paths.append(path)
  plt.close(fig)
  return paths


def main():
  args = parse_args()
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
  data = load_milestones(args.root)
  data.to_csv(outdir / "craftax_learning_curves_per_run.csv", index=False)
  summary = aggregate_seed_split(data)
  summary.to_csv(outdir / "craftax_learning_curves_summary.csv", index=False)
  paths = plot(summary, outdir, formats)
  print("Wrote:")
  for path in paths:
    print(path)
  print(outdir / "craftax_learning_curves_summary.csv")


if __name__ == "__main__":
  main()

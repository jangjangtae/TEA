#!/usr/bin/env python3
"""Plot task-performance retention against the clean reference policy."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPLITS = ("clean", "seen", "holdout", "sparse")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", required=True, help="Adaptation run root.")
  parser.add_argument("--analysis-root", required=True)
  parser.add_argument("--milestones", type=int, nargs="+", required=True)
  parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
  parser.add_argument("--outdir", default="")
  return parser.parse_args()


def number(value):
  try:
    value = float(value)
  except (TypeError, ValueError):
    return float("nan")
  return value if math.isfinite(value) else float("nan")


def mean_std(values):
  values = np.asarray([x for x in values if math.isfinite(x)], np.float64)
  if values.size == 0:
    return float("nan"), float("nan")
  mean = float(values.mean())
  std = float(values.std(ddof=1)) if values.size > 1 else 0.0
  return mean, std


def read_scores(path):
  values = []
  score_path = path / "scores.jsonl"
  if not score_path.exists():
    return float("nan")
  with score_path.open(encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      row = json.loads(line)
      if "episode/score" in row:
        values.append(number(row["episode/score"]))
  mean, _ = mean_std(values)
  return mean


def read_csv(path):
  with path.open(newline="", encoding="utf-8") as f:
    return list(csv.DictReader(f))


def write_csv(path, rows, fields):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def reference_scores(root, seeds):
  rows = []
  scores = {}
  for seed in seeds:
    for split in SPLITS:
      score = read_scores(root / f"seed_{seed}" / f"base_{split}_eval")
      scores[(seed, split)] = score
      rows.append({"seed": seed, "split": split, "reference_score": score})
  return scores, rows


def collect_retention(analysis_root, ref_scores, milestones, seeds):
  rows = []
  seed_set = set(seeds)
  for milestone in milestones:
    path = analysis_root / f"milestone_{milestone}" / "per_run_metrics.csv"
    for row in read_csv(path):
      if row.get("phase") != "eval":
        continue
      seed = int(row["seed"])
      if seed not in seed_set:
        continue
      split = row["split"]
      variant = row["variant"]
      if split not in SPLITS or variant == "reference":
        continue
      score = number(row.get("episode_score_mean"))
      ref = ref_scores.get((seed, split), float("nan"))
      retention = score / ref if math.isfinite(score) and ref > 0 else float("nan")
      rows.append({
          "milestone": milestone,
          "seed": seed,
          "split": split,
          "variant": variant,
          "episode_score": score,
          "reference_score": ref,
          "task_retention": retention,
          "task_retention_pct": retention * 100.0,
      })
  return rows


def aggregate(rows):
  groups = defaultdict(list)
  for row in rows:
    groups[(row["milestone"], row["split"], row["variant"])].append(row)
  out = []
  for (milestone, split, variant), items in sorted(groups.items()):
    score_mean, score_std = mean_std([number(x["episode_score"]) for x in items])
    ref_mean, ref_std = mean_std([number(x["reference_score"]) for x in items])
    ret_mean, ret_std = mean_std([number(x["task_retention_pct"]) for x in items])
    out.append({
        "milestone": milestone,
        "split": split,
        "variant": variant,
        "num_seeds": len({x["seed"] for x in items}),
        "episode_score_mean": score_mean,
        "episode_score_std": score_std,
        "reference_score_mean": ref_mean,
        "reference_score_std": ref_std,
        "task_retention_pct_mean": ret_mean,
        "task_retention_pct_std": ret_std,
    })
  return out


def style():
  plt.rcParams.update({
      "figure.dpi": 140,
      "savefig.dpi": 220,
      "axes.grid": True,
      "grid.alpha": 0.25,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "font.size": 10,
  })


def plot_final_bars(rows, outdir, final_milestone):
  rows = [r for r in rows if int(r["milestone"]) == final_milestone]
  variants = sorted({r["variant"] for r in rows})
  fig, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
  for ax, split in zip(axes.flat, SPLITS):
    sub = sorted(
        [r for r in rows if r["split"] == split],
        key=lambda r: variants.index(r["variant"]))
    xs = np.arange(len(sub))
    means = [number(r["task_retention_pct_mean"]) for r in sub]
    stds = [number(r["task_retention_pct_std"]) for r in sub]
    ax.bar(xs, means, yerr=stds, capsize=4)
    ax.axhline(100.0, color="#444444", linestyle=":", linewidth=1.2)
    ax.set_title(f"{split}: task retention vs clean reference")
    ax.set_ylabel("Reference task score retained (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["variant"] for r in sub], rotation=30, ha="right")
    ax.text(
        0.99, 0.98, "Error bars: SD across seeds",
        transform=ax.transAxes, ha="right", va="top", fontsize=8)
  fig.suptitle(f"Task Performance Retention at {final_milestone:,} Steps", y=1.02)
  fig.tight_layout()
  fig.savefig(outdir / "task_retention_final_vs_reference.png", bbox_inches="tight")
  plt.close(fig)


def plot_curves(rows, outdir, milestones):
  variants = sorted({r["variant"] for r in rows})
  colors = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(variants)}
  fig, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
  for ax, split in zip(axes.flat, SPLITS):
    for variant in variants:
      points = {
          int(r["milestone"]): r for r in rows
          if r["split"] == split and r["variant"] == variant
      }
      xs = [m / 1000 for m in milestones if m in points]
      ys = [number(points[m]["task_retention_pct_mean"]) for m in milestones if m in points]
      yerr = [number(points[m]["task_retention_pct_std"]) for m in milestones if m in points]
      if xs:
        ax.errorbar(
            xs, ys, yerr=yerr, marker="o", linewidth=1.8, capsize=3,
            color=colors[variant], label=variant)
    ax.axhline(100.0, color="#444444", linestyle=":", linewidth=1.2)
    ax.set_title(split)
    ax.set_xlabel("Adaptation steps (thousands)")
    ax.set_ylabel("Reference task score retained (%)")
  handles, labels = axes[0, 0].get_legend_handles_labels()
  if handles:
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
  fig.suptitle("Task Performance Retention Across Milestones", y=1.03)
  fig.text(0.995, 0.995, "Error bars: SD across seeds", ha="right", va="top", fontsize=8)
  fig.tight_layout()
  fig.savefig(outdir / "task_retention_milestones_vs_reference.png", bbox_inches="tight")
  plt.close(fig)


def write_summary(path, rows, final_milestone):
  final = [r for r in rows if int(r["milestone"]) == final_milestone]
  lines = [
      "# Task Retention Summary",
      "",
      "Retention is episode score divided by the clean pretrained reference score for the same seed and split.",
      "",
      "| Split | Variant | Retention (%) | Task score | Reference score |",
      "|---|---|---:|---:|---:|",
  ]
  for split in SPLITS:
    for row in sorted([r for r in final if r["split"] == split], key=lambda x: x["variant"]):
      lines.append(
          f"| {split} | `{row['variant']}` | "
          f"{number(row['task_retention_pct_mean']):.1f} +/- {number(row['task_retention_pct_std']):.1f} | "
          f"{number(row['episode_score_mean']):.3f} +/- {number(row['episode_score_std']):.3f} | "
          f"{number(row['reference_score_mean']):.3f} +/- {number(row['reference_score_std']):.3f} |")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
  args = parse_args()
  root = Path(args.root).expanduser()
  analysis_root = Path(args.analysis_root).expanduser()
  outdir = Path(args.outdir).expanduser() if args.outdir else analysis_root / "task_retention"
  outdir.mkdir(parents=True, exist_ok=True)
  milestones = sorted(set(args.milestones))

  refs, ref_rows = reference_scores(root, args.seeds)
  rows = collect_retention(analysis_root, refs, milestones, args.seeds)
  agg = aggregate(rows)

  write_csv(outdir / "reference_task_scores.csv", ref_rows, [
      "seed", "split", "reference_score"])
  write_csv(outdir / "task_retention_per_seed.csv", rows, [
      "milestone", "seed", "split", "variant", "episode_score",
      "reference_score", "task_retention", "task_retention_pct"])
  write_csv(outdir / "task_retention_aggregate.csv", agg, [
      "milestone", "split", "variant", "num_seeds",
      "episode_score_mean", "episode_score_std",
      "reference_score_mean", "reference_score_std",
      "task_retention_pct_mean", "task_retention_pct_std"])

  style()
  final_milestone = max(milestones)
  plot_final_bars(agg, outdir, final_milestone)
  plot_curves(agg, outdir, milestones)
  write_summary(outdir / "task_retention_summary.md", agg, final_milestone)
  print(f"Wrote task retention analysis: {outdir}")


if __name__ == "__main__":
  main()

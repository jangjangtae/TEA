#!/usr/bin/env python3
"""Plot and score fault-objective ablations against the task-only baseline.

The generic analyzer writes broad CSV summaries. This script turns those CSVs
into compact figures and tables for the threshold/delta objective experiments.
It does not rerun any environment code; it only reads existing analysis files.
"""

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_DIR = (
    "/home/railab/logdir/fault_objective_weekend_20260619_142823/analysis")
DEFAULT_OBJECTIVE_SUITE = "fault_objective_weekend_20260619_142823"
DEFAULT_BASELINE_SUITE = "fault_profile_long_20260618_134300"
DEFAULT_BASELINE_RUN = "02_eval_task_only_fault_logging"
DEFAULT_DENSE_RUN = "04_eval_fault_adapt_beta005"

SPLITS = ["seen", "holdout"]
METRIC_SPECS = [
    ("episode_task_score_mean", "Task score", "higher"),
    ("step_fault_applied_rate", "Bug manifestation rate", "higher"),
    ("step_auroc", "Fault-score AUROC", "higher"),
    ("step_auprc", "Fault-score AUPRC", "higher"),
    ("clean_false_alarm_episode_rate", "Clean false-alarm episode rate", "lower"),
]

RUN_LABELS = {
    "02_eval_task_only_fault_logging": "Task-only",
    "04_eval_fault_adapt_beta005": "Dense beta=.05",
    "02_eval_threshold_p95_beta01_action": "Threshold p95 beta=.10",
    "04_eval_threshold_p99_beta02_action": "Threshold p99 beta=.20",
    "06_eval_excess_p95_beta01_action": "Excess p95 beta=.10",
    "08_eval_delta_p95_beta01_action": "Delta p95 beta=.10",
    "10_eval_excess_delta_p95_beta01_action": "Excess-delta p95 beta=.10",
}

RUN_COLORS = {
    "02_eval_task_only_fault_logging": "#4C78A8",
    "04_eval_fault_adapt_beta005": "#F28E2B",
    "02_eval_threshold_p95_beta01_action": "#59A14F",
    "04_eval_threshold_p99_beta02_action": "#8CD17D",
    "06_eval_excess_p95_beta01_action": "#B6992D",
    "08_eval_delta_p95_beta01_action": "#B07AA1",
    "10_eval_excess_delta_p95_beta01_action": "#E15759",
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
  parser.add_argument("--outdir", default="")
  parser.add_argument("--objective-suite", default=DEFAULT_OBJECTIVE_SUITE)
  parser.add_argument("--baseline-suite", default=DEFAULT_BASELINE_SUITE)
  parser.add_argument("--baseline-run", default=DEFAULT_BASELINE_RUN)
  parser.add_argument("--dense-run", default=DEFAULT_DENSE_RUN)
  parser.add_argument("--formats", default="png,pdf")
  return parser.parse_args()


def setup_style():
  plt.rcParams.update({
      "figure.dpi": 140,
      "savefig.dpi": 220,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "axes.grid": True,
      "grid.alpha": 0.25,
      "grid.linewidth": 0.7,
      "axes.titleweight": "bold",
      "axes.labelsize": 10,
      "xtick.labelsize": 8,
      "ytick.labelsize": 9,
      "legend.frameon": False,
      "font.size": 10,
  })


def load_summary(analysis_dir):
  path = Path(analysis_dir) / "summary_metrics.csv"
  if not path.exists():
    raise FileNotFoundError(f"Missing summary metrics: {path}")
  df = pd.read_csv(path)
  for col in df.columns:
    if col in {"suite", "run", "split", "raw_eval_name", "root", "eval_dir", "summary_path"}:
      continue
    df[col] = pd.to_numeric(df[col], errors="coerce")
  return df


def label_run(run):
  return RUN_LABELS.get(str(run), str(run))


def select_rows(df, args):
  baseline = df[
      (df["suite"] == args.baseline_suite)
      & (df["run"] == args.baseline_run)
      & (df["split"].isin(SPLITS))
  ].copy()
  dense = df[
      (df["suite"] == args.baseline_suite)
      & (df["run"] == args.dense_run)
      & (df["split"].isin(SPLITS))
  ].copy()
  objective = df[
      (df["suite"] == args.objective_suite)
      & (df["split"].isin(SPLITS))
      & (df["run"].astype(str).str.startswith(("02_eval_", "04_eval_", "06_eval_", "08_eval_", "10_eval_")))
  ].copy()
  rows = pd.concat([baseline, dense, objective], ignore_index=True)
  rows["run_label"] = rows["run"].map(label_run)
  rows["is_baseline"] = (
      (rows["suite"] == args.baseline_suite) & (rows["run"] == args.baseline_run))
  rows["is_dense"] = (
      (rows["suite"] == args.baseline_suite) & (rows["run"] == args.dense_run))
  return rows


def compute_scorecard(rows, args):
  baselines = rows[rows["is_baseline"]].set_index("split")
  out = []
  for _, row in rows.iterrows():
    split = row["split"]
    if split not in baselines.index:
      continue
    base = baselines.loc[split]
    item = {
        "split": split,
        "suite": row["suite"],
        "run": row["run"],
        "run_label": row["run_label"],
        "is_baseline": bool(row["is_baseline"]),
        "is_dense": bool(row["is_dense"]),
    }
    for metric, _, _ in METRIC_SPECS:
      value = row.get(metric, np.nan)
      base_value = base.get(metric, np.nan)
      item[metric] = value
      item[f"{metric}_baseline"] = base_value
      item[f"{metric}_delta"] = value - base_value
      if metric == "episode_task_score_mean" and base_value and not math.isnan(base_value):
        item["task_retention"] = value / base_value
    out.append(item)
  score = pd.DataFrame(out)
  if score.empty:
    return score
  score["objective_score"] = (
      score["step_fault_applied_rate_delta"].fillna(0.0) * 100.0
      + score["step_auroc_delta"].fillna(0.0)
      + score["step_auprc_delta"].fillna(0.0) * 2.0
      - score["clean_false_alarm_episode_rate_delta"].fillna(0.0) * 0.2
      - score["episode_task_score_mean_delta"].clip(upper=0).abs().fillna(0.0) * 0.1
  )
  return score


def savefig(fig, outdir, stem, formats):
  paths = []
  for fmt in formats:
    path = outdir / f"{stem}.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    paths.append(path)
  plt.close(fig)
  return paths


def plot_metric_grid(rows, outdir, formats):
  run_order = [
      DEFAULT_BASELINE_RUN,
      DEFAULT_DENSE_RUN,
      "02_eval_threshold_p95_beta01_action",
      "04_eval_threshold_p99_beta02_action",
      "06_eval_excess_p95_beta01_action",
      "08_eval_delta_p95_beta01_action",
      "10_eval_excess_delta_p95_beta01_action",
  ]
  fig, axes = plt.subplots(len(METRIC_SPECS), len(SPLITS), figsize=(15, 14))
  for i, (metric, title, _) in enumerate(METRIC_SPECS):
    for j, split in enumerate(SPLITS):
      ax = axes[i, j]
      sub = rows[rows["split"] == split].copy()
      sub["order"] = sub["run"].map({name: idx for idx, name in enumerate(run_order)})
      sub = sub.sort_values("order")
      labels = [label_run(x) for x in sub["run"]]
      colors = [RUN_COLORS.get(str(x), "#777777") for x in sub["run"]]
      values = sub[metric].astype(float).to_numpy()
      ax.bar(np.arange(len(sub)), values, color=colors)
      base = sub[sub["is_baseline"]]
      if len(base) and pd.notna(base[metric].iloc[0]):
        ax.axhline(float(base[metric].iloc[0]), color="#222222", linestyle="--", linewidth=1.0)
      if metric == "step_auroc":
        ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1.0)
      ax.set_title(f"{title} / {split}")
      ax.set_xticks(np.arange(len(labels)))
      ax.set_xticklabels(labels, rotation=35, ha="right")
      if j == 0:
        ax.set_ylabel(title)
  fig.suptitle("Fault objective ablation metrics", y=1.01)
  fig.tight_layout()
  return savefig(fig, outdir, "objective_metric_grid", formats)


def plot_tradeoffs(rows, outdir, formats):
  fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
  markers = {"seen": "o", "holdout": "s"}
  plots = [
      ("episode_task_score_mean", "step_fault_applied_rate", "Task score", "Bug manifestation rate"),
      ("episode_task_score_mean", "step_auroc", "Task score", "Fault-score AUROC"),
      ("episode_task_score_mean", "step_auprc", "Task score", "Fault-score AUPRC"),
  ]
  for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, plots):
    for _, row in rows.iterrows():
      color = RUN_COLORS.get(str(row["run"]), "#777777")
      marker = markers.get(str(row["split"]), "o")
      ax.scatter(row[xcol], row[ycol], color=color, marker=marker, s=80,
                 edgecolor="white", linewidth=0.8)
      short = {
          "Task-only": "Task",
          "Dense beta=.05": "Dense",
          "Threshold p95 beta=.10": "T95",
          "Threshold p99 beta=.20": "T99",
          "Excess p95 beta=.10": "Ex95",
          "Delta p95 beta=.10": "D95",
          "Excess-delta p95 beta=.10": "XD95",
      }.get(row["run_label"], row["run_label"])
      ax.annotate(short, (row[xcol], row[ycol]), xytext=(4, 4),
                  textcoords="offset points", fontsize=8)
    if ycol == "step_auroc":
      ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{xlabel} vs {ylabel}")
  handles = [
      plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=label_run(run), markersize=8)
      for run, color in RUN_COLORS.items()
      if run in set(rows["run"])
  ]
  split_handles = [
      plt.Line2D([0], [0], marker=marker, linestyle="", color="#333333", label=split, markersize=8)
      for split, marker in markers.items()
  ]
  fig.legend(handles=handles + split_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
  fig.tight_layout()
  return savefig(fig, outdir, "objective_tradeoffs", formats)


def plot_deltas(score, outdir, formats):
  objective = score[(~score["is_baseline"]) & (~score["is_dense"])].copy()
  if objective.empty:
    return []
  metrics = [
      ("step_fault_applied_rate_delta", "Bug rate delta"),
      ("step_auroc_delta", "AUROC delta"),
      ("step_auprc_delta", "AUPRC delta"),
      ("episode_task_score_mean_delta", "Task score delta"),
  ]
  fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 12), sharex=True)
  xlabels = []
  for split in SPLITS:
    for run in objective[objective["split"] == split]["run_label"]:
      xlabels.append(f"{run}\n{split}")
  x = np.arange(len(xlabels))
  for ax, (metric, title) in zip(axes, metrics):
    vals = []
    colors = []
    for split in SPLITS:
      sub = objective[objective["split"] == split]
      for _, row in sub.iterrows():
        vals.append(row[metric])
        colors.append(RUN_COLORS.get(row["run"], "#777777"))
    ax.bar(x, vals, color=colors)
    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_title(f"{title} vs task-only baseline")
    ax.set_ylabel("Delta")
  axes[-1].set_xticks(x)
  axes[-1].set_xticklabels(xlabels, rotation=35, ha="right")
  fig.tight_layout()
  return savefig(fig, outdir, "objective_deltas_vs_taskonly", formats)


def write_markdown(path, score):
  objective = score[~score["is_baseline"]].copy()
  lines = [
      "# Fault Objective Result Notes",
      "",
      "Generated from `summary_metrics.csv`; task-only is the comparison baseline.",
      "",
      "## Best Rows",
      "",
  ]
  for split in SPLITS:
    sub = objective[objective["split"] == split].copy()
    if sub.empty:
      continue
    best_task = sub.sort_values("episode_task_score_mean", ascending=False).iloc[0]
    best_bug = sub.sort_values("step_fault_applied_rate", ascending=False).iloc[0]
    best_auroc = sub.sort_values("step_auroc", ascending=False).iloc[0]
    best_score = sub.sort_values("objective_score", ascending=False).iloc[0]
    lines.extend([
        f"### {split}",
        "",
        f"- Best task score: {best_task['run_label']} ({best_task['episode_task_score_mean']:.4f}).",
        f"- Best bug manifestation rate: {best_bug['run_label']} ({best_bug['step_fault_applied_rate']:.5f}).",
        f"- Best AUROC: {best_auroc['run_label']} ({best_auroc['step_auroc']:.4f}).",
        f"- Best combined score: {best_score['run_label']} ({best_score['objective_score']:.4f}).",
        "",
    ])
  keep = [
      "split", "run_label", "episode_task_score_mean", "task_retention",
      "step_fault_applied_rate", "step_fault_applied_rate_delta",
      "step_auroc", "step_auroc_delta", "step_auprc", "step_auprc_delta",
      "clean_false_alarm_episode_rate", "clean_false_alarm_episode_rate_delta",
      "objective_score",
  ]
  lines.extend(["## Scorecard", "", markdown_table(score[keep]), ""])
  path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df):
  cols = list(df.columns)
  out = [
      "| " + " | ".join(cols) + " |",
      "| " + " | ".join(["---"] * len(cols)) + " |",
  ]
  for _, row in df.iterrows():
    vals = []
    for col in cols:
      value = row[col]
      if isinstance(value, float):
        vals.append("" if math.isnan(value) else f"{value:.5f}")
      else:
        vals.append(str(value))
    out.append("| " + " | ".join(vals) + " |")
  return "\n".join(out)


def main():
  args = parse_args()
  setup_style()
  formats = [x.strip() for x in args.formats.split(",") if x.strip()]
  analysis_dir = Path(args.analysis_dir)
  outdir = Path(args.outdir) if args.outdir else analysis_dir / "objective_figures"
  outdir.mkdir(parents=True, exist_ok=True)

  summary = load_summary(analysis_dir)
  rows = select_rows(summary, args)
  if rows.empty:
    raise SystemExit("No matching rows found. Check --analysis-dir and suite names.")
  score = compute_scorecard(rows, args)

  rows.to_csv(outdir / "objective_selected_rows.csv", index=False)
  score.to_csv(outdir / "objective_scorecard.csv", index=False)
  write_markdown(outdir / "objective_interpretation.md", score)

  written = []
  written += plot_metric_grid(rows, outdir, formats)
  written += plot_tradeoffs(rows, outdir, formats)
  written += plot_deltas(score, outdir, formats)

  print(f"Wrote: {outdir}")
  print(f"- {outdir / 'objective_selected_rows.csv'}")
  print(f"- {outdir / 'objective_scorecard.csv'}")
  print(f"- {outdir / 'objective_interpretation.md'}")
  for path in written:
    print(f"- {path}")


if __name__ == "__main__":
  main()

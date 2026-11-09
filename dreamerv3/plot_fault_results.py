#!/usr/bin/env python3
"""Generate figures and compact interpretation tables for fault-score runs."""

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


SPLIT_ORDER = ["seen", "holdout", "semantic_holdout"]
METHOD_ORDER = [
    "task_only",
    "gated_beta005",
    "gated_beta01",
    "ungated_beta005",
    "oracle_tester",
]
METHOD_LABELS = {
    "task_only": "Task-only",
    "gated_beta005": "Gated beta=.05",
    "gated_beta01": "Gated beta=.10",
    "ungated_beta005": "Ungated beta=.05",
    "oracle_tester": "Manual bug reward",
}
METHOD_COLORS = {
    "task_only": "#4C78A8",
    "gated_beta005": "#59A14F",
    "gated_beta01": "#8CD17D",
    "ungated_beta005": "#F28E2B",
    "oracle_tester": "#E15759",
}
SPLIT_LABELS = {
    "seen": "Seen bugs",
    "holdout": "Held-out low-level",
    "semantic_holdout": "Semantic holdout",
}
MARKERS = {
    "seen": "o",
    "holdout": "s",
    "semantic_holdout": "^",
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--analysis-dir",
      default="/home/railab/logdir/fault_weekend_20260612_134024/analysis_long_resume",
      help="Directory containing summary_metrics.csv and related analysis CSVs.",
  )
  parser.add_argument(
      "--outdir",
      default=None,
      help="Output directory. Defaults to <analysis-dir>/figures.",
  )
  parser.add_argument(
      "--focus-suite",
      default="fault_weekend_20260612_134024",
      help="Suite to emphasize in the main plots.",
  )
  parser.add_argument(
      "--formats",
      default="png,pdf",
      help="Comma-separated figure formats.",
  )
  return parser.parse_args()


def setup_style():
  plt.rcParams.update({
      "figure.figsize": (10, 5.6),
      "figure.dpi": 140,
      "savefig.dpi": 220,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "axes.grid": True,
      "grid.alpha": 0.24,
      "grid.linewidth": 0.7,
      "axes.titleweight": "bold",
      "axes.labelsize": 10,
      "xtick.labelsize": 9,
      "ytick.labelsize": 9,
      "legend.frameon": False,
      "font.size": 10,
  })


def load_csv(path):
  path = Path(path)
  if not path.exists():
    return pd.DataFrame()
  return pd.read_csv(path)


def coerce_numeric(df):
  if df.empty:
    return df
  for col in df.columns:
    if col in {"suite", "run", "split", "condition", "raw_eval_name", "root", "eval_dir", "summary_path"}:
      continue
    converted = pd.to_numeric(df[col], errors="coerce")
    if converted.notna().any() or df[col].isna().all():
      df[col] = converted
  return df


def method_from_run(run):
  run = str(run)
  if "ungated_beta005" in run or "beta0.05_ungated" in run:
    return "ungated_beta005"
  if "gated_beta005" in run or "beta0.05" in run or "beta005" in run:
    return "gated_beta005"
  if "gated_beta01" in run or "beta0.1" in run or "beta01" in run:
    return "gated_beta01"
  if "oracle" in run or "tester_reward" in run:
    return "oracle_tester"
  if "task_only" in run:
    return "task_only"
  if "reference" in run:
    return "reference"
  return run


def add_method(df):
  if df.empty or "run" not in df:
    return df
  df = df.copy()
  if "raw_eval_name" in df:
    source = df["raw_eval_name"].fillna("").astype(str) + " " + df["run"].fillna("").astype(str)
  else:
    source = df["run"].fillna("").astype(str)
  df["method"] = source.map(method_from_run)
  df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
  return df


def focus_mask(df, suite):
  if df.empty:
    return pd.Series(False, index=df.index)
  mask = pd.Series(False, index=df.index)
  if "suite" in df:
    mask = mask | (df["suite"].astype(str) == suite)
  if "root" in df:
    root_names = df["root"].astype(str).map(lambda x: Path(x).name)
    mask = mask | (root_names == suite)
  return mask


def savefig(fig, outdir, stem, formats):
  paths = []
  for fmt in formats:
    path = outdir / f"{stem}.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    paths.append(path)
  plt.close(fig)
  return paths


def focus_frame(summary, suite):
  df = summary[(summary["suite"] == suite) & (summary["split"].isin(SPLIT_ORDER))].copy()
  df = df[df["method"].isin(METHOD_ORDER)].copy()
  df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
  df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
  return df.sort_values(["split", "method"])


def grouped_bars(ax, df, metric, ylabel, title, ylim=None, baseline=None):
  splits = SPLIT_ORDER
  methods = METHOD_ORDER
  width = 0.15
  x = np.arange(len(splits))
  for i, method in enumerate(methods):
    vals = []
    for split in splits:
      rows = df[(df["split"] == split) & (df["method"] == method)]
      vals.append(float(rows[metric].iloc[0]) if len(rows) and pd.notna(rows[metric].iloc[0]) else np.nan)
    offset = (i - (len(methods) - 1) / 2) * width
    ax.bar(
        x + offset,
        vals,
        width=width,
        color=METHOD_COLORS.get(method, "#777777"),
        label=METHOD_LABELS.get(method, method),
    )
  ax.set_xticks(x)
  ax.set_xticklabels([SPLIT_LABELS[x] for x in splits], rotation=12, ha="right")
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  if baseline is not None:
    ax.axhline(baseline, color="#333333", linewidth=1, linestyle="--", alpha=0.7)
  if ylim is not None:
    ax.set_ylim(*ylim)


def plot_long_eval_bars(df, outdir, formats):
  fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
  grouped_bars(axes[0], df, "step_auroc", "AUROC", "Fault-score AUROC", ylim=(0.45, 0.72), baseline=0.5)
  grouped_bars(axes[1], df, "step_auprc", "AUPRC", "Fault-score AUPRC", ylim=(0.0, max(0.045, df["step_auprc"].max() * 1.18)))
  grouped_bars(axes[2], df, "episode_task_score_mean", "Episode task score", "Task performance")
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
  fig.suptitle("Long-eval comparison: signal quality and task competence", y=1.03)
  fig.tight_layout()
  return savefig(fig, outdir, "01_long_eval_signal_and_task", formats)


def plot_task_tradeoffs(df, outdir, formats):
  fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
  for _, row in df.iterrows():
    method = str(row["method"])
    split = str(row["split"])
    color = METHOD_COLORS.get(method, "#777777")
    marker = MARKERS.get(split, "o")
    label = f"{METHOD_LABELS.get(method, method)} / {SPLIT_LABELS.get(split, split)}"
    axes[0].scatter(
        row["episode_task_score_mean"], row["step_fault_applied_rate"],
        s=70, color=color, marker=marker, edgecolor="white", linewidth=0.7)
    axes[1].scatter(
        row["episode_task_score_mean"], row["step_auroc"],
        s=70, color=color, marker=marker, edgecolor="white", linewidth=0.7)
    short = {
        "task_only": "Task",
        "gated_beta005": "G.05",
        "gated_beta01": "G.10",
        "ungated_beta005": "U.05",
        "oracle_tester": "BugR",
    }.get(method, method)
    axes[0].annotate(short, (row["episode_task_score_mean"], row["step_fault_applied_rate"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].annotate(short, (row["episode_task_score_mean"], row["step_auroc"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=8)

  axes[0].set_xlabel("Episode task score")
  axes[0].set_ylabel("Fault step rate")
  axes[0].set_title("Task competence vs bug exposure")
  axes[1].set_xlabel("Episode task score")
  axes[1].set_ylabel("Fault-score AUROC")
  axes[1].set_title("Task competence vs fault-score separability")
  axes[1].axhline(0.5, color="#333333", linewidth=1, linestyle="--", alpha=0.7)

  method_handles = [
      plt.Line2D([0], [0], marker="o", linestyle="", color=METHOD_COLORS[m],
                 label=METHOD_LABELS[m], markersize=8)
      for m in METHOD_ORDER
  ]
  split_handles = [
      plt.Line2D([0], [0], marker=MARKERS[s], linestyle="", color="#333333",
                 label=SPLIT_LABELS[s], markersize=8)
      for s in SPLIT_ORDER
  ]
  fig.legend(handles=method_handles + split_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
  fig.tight_layout()
  return savefig(fig, outdir, "02_task_tradeoff_scatter", formats)


def plot_detection_false_alarm(df, outdir, formats):
  fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
  for ax, split in zip(axes, SPLIT_ORDER):
    sub = df[df["split"] == split]
    for _, row in sub.iterrows():
      method = str(row["method"])
      ax.scatter(
          row["clean_false_alarm_episode_rate"], row["episode_detection_rate"],
          color=METHOD_COLORS.get(method, "#777777"), s=85,
          edgecolor="white", linewidth=0.7)
      ax.annotate(METHOD_LABELS.get(method, method).replace(" beta=", "\n"),
                  (row["clean_false_alarm_episode_rate"], row["episode_detection_rate"]),
                  xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_title(SPLIT_LABELS[split])
    ax.set_xlabel("Clean false-alarm episode rate")
    ax.set_xlim(0.78, 0.93)
    ax.set_ylim(0.15, 0.78)
  axes[0].set_ylabel("Episode detection rate")
  fig.suptitle("Thresholded detection is noisy: detection rises with high false alarms", y=1.03)
  fig.tight_layout()
  return savefig(fig, outdir, "03_detection_vs_false_alarm", formats)


def plot_trace_precision(trace, outdir, formats, suite):
  df = trace[focus_mask(trace, suite) & (trace["split"].isin(SPLIT_ORDER))].copy()
  if "label" in df:
    df = df[df["label"].fillna("fault_manifested").astype(str) == "fault_manifested"]
  df = add_method(df)
  df = df[df["method"].isin(METHOD_ORDER)]
  if df.empty:
    return []
  metrics = [
      ("precision_at_top_0.001", "Top 0.1%"),
      ("precision_at_top_0.01", "Top 1%"),
      ("precision_at_top_0.05", "Top 5%"),
  ]
  fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
  width = 0.15
  x = np.arange(len(SPLIT_ORDER))
  for ax, (metric, title) in zip(axes, metrics):
    for i, method in enumerate(METHOD_ORDER):
      vals = []
      for split in SPLIT_ORDER:
        rows = df[(df["split"] == split) & (df["method"] == method)]
        vals.append(float(rows[metric].iloc[0]) if len(rows) and pd.notna(rows[metric].iloc[0]) else np.nan)
      ax.bar(x + (i - 2) * width, vals, width=width, color=METHOD_COLORS.get(method, "#777777"))
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER], rotation=12, ha="right")
    ax.set_ylim(0, max(0.12, np.nanmax(df[[m[0] for m in metrics]].values) * 1.2))
    ax.set_ylabel("Fault precision in highest fault-score steps")
  handles = [
      plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
      for m in METHOD_ORDER
  ]
  fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
  fig.suptitle("Ranking quality: how bug-dense are top fault-score steps?", y=1.03)
  fig.tight_layout()
  return savefig(fig, outdir, "04_trace_topk_precision", formats)


def plot_context_conditioned(context, outdir, formats, suite):
  df = context[focus_mask(context, suite) & (context["split"] == "semantic_holdout")].copy()
  if "label" in df:
    df = df[df["label"].fillna("fault_manifested").astype(str) == "fault_manifested"]
  df = add_method(df)
  df = df[df["method"].isin(METHOD_ORDER)]
  df = df[df["condition"].isin(["all", "semantic_context"])]
  if df.empty:
    return []

  fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  metrics = [
      ("auroc", "AUROC", (0.45, 0.8)),
      ("auprc", "AUPRC", None),
      ("precision_at_top_0.01", "Precision@Top1%", (0.0, 1.05)),
  ]
  width = 0.15
  methods = [m for m in METHOD_ORDER if m in set(df["method"])]
  x = np.arange(len(methods))
  for ax, (metric, ylabel, ylim) in zip(axes, metrics):
    for j, cond in enumerate(["all", "semantic_context"]):
      vals = []
      for method in methods:
        rows = df[(df["method"] == method) & (df["condition"] == cond)]
        vals.append(float(rows[metric].iloc[0]) if len(rows) and pd.notna(rows[metric].iloc[0]) else np.nan)
      ax.bar(x + (j - 0.5) * width, vals, width=width,
             color="#9ecae1" if cond == "all" else "#fdae6b",
             label="All semantic-holdout steps" if cond == "all" else "Only semantic trigger contexts")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    if metric == "auroc":
      ax.axhline(0.5, color="#333333", linewidth=1, linestyle="--", alpha=0.7)
    if ylim:
      ax.set_ylim(*ylim)
  axes[0].legend(loc="upper left")
  fig.suptitle("Context-conditioned fault score: signal appears inside semantic trigger contexts", y=1.03)
  fig.tight_layout()
  return savefig(fig, outdir, "05_context_conditioned_semantic_signal", formats)


def plot_context_breakdown(context, outdir, formats, suite):
  df = context[focus_mask(context, suite) & (context["split"] == "semantic_holdout")].copy()
  if "label" in df:
    df = df[df["label"].fillna("fault_manifested").astype(str) == "fault_manifested"]
  df = add_method(df)
  df = df[df["method"].isin(METHOD_ORDER)]
  df = df[df["condition"].str.startswith("ctx_", na=False)]
  df = df[(df["n_steps"] >= 50) & np.isfinite(df["score_mean_all"])]
  if df.empty:
    return []

  agg = df.groupby("condition", as_index=False).agg(
      n_steps=("n_steps", "sum"),
      fault_step_rate=("fault_step_rate", "mean"),
      score_mean_fault=("score_mean_fault", "mean"),
      score_mean_normal=("score_mean_normal", "mean"),
      auroc=("auroc", "mean"),
  )
  agg["score_gap"] = agg["score_mean_fault"] - agg["score_mean_normal"]
  agg = agg.sort_values("score_gap", ascending=False)

  labels = [x.replace("ctx_", "").replace("_", "\n") for x in agg["condition"]]
  fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
  axes[0].bar(np.arange(len(agg)), agg["score_gap"], color="#76B7B2")
  axes[0].set_xticks(np.arange(len(agg)))
  axes[0].set_xticklabels(labels, rotation=0, ha="center")
  axes[0].set_ylabel("Fault minus normal mean score")
  axes[0].set_title("Fault-score gap by semantic context")
  axes[1].bar(np.arange(len(agg)), agg["fault_step_rate"], color="#EDC948")
  axes[1].set_xticks(np.arange(len(agg)))
  axes[1].set_xticklabels(labels, rotation=0, ha="center")
  axes[1].set_ylabel("Fault rate inside context")
  axes[1].set_title("How often the semantic trigger becomes a bug")
  fig.tight_layout()
  paths = savefig(fig, outdir, "06_semantic_context_breakdown", formats)
  agg.to_csv(outdir / "semantic_context_breakdown.csv", index=False)
  return paths


def plot_event_window(events, outdir, formats, suite):
  df = events[focus_mask(events, suite) & (events["split"].isin(SPLIT_ORDER))].copy()
  if "event_label" in df:
    df = df[df["event_label"].fillna("fault_manifested").astype(str) == "fault_manifested"]
  df = add_method(df)
  df = df[df["method"].isin(METHOD_ORDER)]
  if df.empty:
    return []

  fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
  for ax, split in zip(axes, SPLIT_ORDER):
    sub = df[df["split"] == split]
    for method in METHOD_ORDER:
      g = sub[sub["method"] == method].sort_values("relative_step")
      if g.empty:
        continue
      baseline = g[(g["relative_step"] >= -20) & (g["relative_step"] <= -5)]["mean_fault_score"].mean()
      vals = g["mean_fault_score"] - baseline if np.isfinite(baseline) else g["mean_fault_score"]
      ax.plot(g["relative_step"], vals, label=METHOD_LABELS[method],
              color=METHOD_COLORS.get(method, "#777777"), linewidth=2)
    ax.axvline(0, color="#333333", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.axhline(0, color="#999999", linewidth=0.8, alpha=0.7)
    ax.set_title(SPLIT_LABELS[split])
    ax.set_xlabel("Steps from bug event")
  axes[0].set_ylabel("Mean fault score delta vs pre-event baseline")
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
  fig.suptitle("Event-window view: does fault score rise around actual bug events?", y=1.03)
  fig.tight_layout()
  return savefig(fig, outdir, "07_event_window_fault_score_delta", formats)


def plot_pareto(summary, outdir, formats, suite):
  df = focus_frame(summary, suite)
  if df.empty:
    return []
  fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
  for ax, split in zip(axes, SPLIT_ORDER):
    sub = df[df["split"] == split]
    for _, row in sub.iterrows():
      method = str(row["method"])
      ax.scatter(
          row["episode_task_score_mean"], row["step_auprc"],
          s=90, color=METHOD_COLORS.get(method, "#777777"),
          edgecolor="white", linewidth=0.7)
      ax.annotate(METHOD_LABELS.get(method, method).replace(" beta=", "\n"),
                  (row["episode_task_score_mean"], row["step_auprc"]),
                  xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_title(SPLIT_LABELS[split])
    ax.set_xlabel("Episode task score")
    ax.set_ylabel("AUPRC")
  fig.suptitle("Multi-objective view: keep task performance while improving fault ranking", y=1.03)
  fig.tight_layout()
  return savefig(fig, outdir, "08_task_vs_auprc_pareto_view", formats)


def write_tables_and_findings(summary, context, trace, outdir, suite):
  focus = focus_frame(summary, suite).copy()
  if focus.empty:
    return

  base_scores = (
      focus[focus["method"] == "task_only"][["split", "episode_task_score_mean"]]
      .rename(columns={"episode_task_score_mean": "task_only_score"})
  )
  focus = focus.merge(base_scores, on="split", how="left")
  focus["task_retention_vs_task_only"] = (
      focus["episode_task_score_mean"] / focus["task_only_score"]
  )
  focus["competent_95pct_task"] = focus["task_retention_vs_task_only"] >= 0.95

  table_cols = [
      "run", "method", "split", "step_auroc", "step_auprc",
      "episode_detection_rate", "step_fault_applied_rate",
      "episode_task_score_mean", "task_retention_vs_task_only",
      "clean_false_alarm_episode_rate",
  ]
  focus[table_cols].to_csv(outdir / "focus_long_eval_table.csv", index=False)

  best_rows = []
  for split, sub in focus.groupby("split", observed=True):
    for metric, direction in [
        ("step_auroc", "max"),
        ("step_auprc", "max"),
        ("episode_task_score_mean", "max"),
        ("step_fault_applied_rate", "max"),
    ]:
      s = sub.dropna(subset=[metric])
      if s.empty:
        continue
      idx = s[metric].idxmax() if direction == "max" else s[metric].idxmin()
      row = s.loc[idx]
      best_rows.append({
          "split": split,
          "metric": metric,
          "best_method": row["method"],
          "best_run": row["run"],
          "value": row[metric],
          "task_score": row["episode_task_score_mean"],
          "task_retention_vs_task_only": row["task_retention_vs_task_only"],
      })
  best = pd.DataFrame(best_rows)
  best.to_csv(outdir / "focus_best_by_split.csv", index=False)

  ctx_focus = pd.DataFrame()
  if not context.empty:
    ctx_focus = context[
        focus_mask(context, suite) &
        (context["split"] == "semantic_holdout")
    ].copy()
    ctx_focus = add_method(ctx_focus)
    ctx_focus = ctx_focus[ctx_focus["method"].isin(METHOD_ORDER)]
    ctx_focus.to_csv(outdir / "focus_context_semantic_holdout.csv", index=False)

  tr_focus = pd.DataFrame()
  if not trace.empty:
    tr_focus = trace[focus_mask(trace, suite) & (trace["split"].isin(SPLIT_ORDER))].copy()
    tr_focus = add_method(tr_focus)
    tr_focus = tr_focus[tr_focus["method"].isin(METHOD_ORDER)]
    tr_focus.to_csv(outdir / "focus_trace_ranking_table.csv", index=False)

  lines = []
  lines.append("# Fault Result Plot Summary")
  lines.append("")
  lines.append(f"Focus suite: `{suite}`")
  lines.append("")
  lines.append("## Key Long-Eval Findings")
  lines.append("")
  for split in SPLIT_ORDER:
    sub = focus[focus["split"] == split]
    if sub.empty:
      continue
    task = sub[sub["method"] == "task_only"]
    task_auroc = float(task["step_auroc"].iloc[0]) if len(task) else math.nan
    task_score = float(task["episode_task_score_mean"].iloc[0]) if len(task) else math.nan
    best_auroc = sub.loc[sub["step_auroc"].idxmax()]
    best_auprc = sub.loc[sub["step_auprc"].idxmax()]
    best_task = sub.loc[sub["episode_task_score_mean"].idxmax()]
    competent = sub[sub["competent_95pct_task"]]
    best_competent = None
    if len(competent):
      best_competent = competent.loc[competent["step_auroc"].idxmax()]
    lines.append(
        f"- **{SPLIT_LABELS[split]}**: best AUROC is "
        f"`{METHOD_LABELS.get(best_auroc['method'], best_auroc['method'])}` "
        f"({best_auroc['step_auroc']:.4f}); task-only AUROC is {task_auroc:.4f} "
        f"with task score {task_score:.2f}."
    )
    lines.append(
        f"  Best AUPRC is `{METHOD_LABELS.get(best_auprc['method'], best_auprc['method'])}` "
        f"({best_auprc['step_auprc']:.4f}); best task score is "
        f"`{METHOD_LABELS.get(best_task['method'], best_task['method'])}` "
        f"({best_task['episode_task_score_mean']:.2f})."
    )
    if best_competent is not None:
      lines.append(
          f"  Under >=95% task retention, best AUROC is "
          f"`{METHOD_LABELS.get(best_competent['method'], best_competent['method'])}` "
          f"({best_competent['step_auroc']:.4f}, retention "
          f"{100 * best_competent['task_retention_vs_task_only']:.1f}%)."
      )
  lines.append("")
  lines.append("## Interpretation")
  lines.append("")
  lines.append(
      "- Low-level seen bugs retain a usable latent-surprise signal, but task-only remains a very strong baseline."
  )
  lines.append(
      "- Low-level holdout signal is modest; manual bug reward can improve AUROC slightly but sacrifices task competence."
  )
  lines.append(
      "- Semantic holdout is weak when measured over all steps, but context-conditioned rows should be read separately because semantic bugs are sparse and only meaningful inside trigger contexts."
  )
  lines.append(
      "- The high clean false-alarm episode rate means thresholded alarms are not yet suitable as a detector by themselves; ranking and context-conditioned analysis are more defensible."
  )
  lines.append("")
  lines.append("## Generated Files")
  lines.append("")
  for path in sorted(outdir.glob("*")):
    if path.suffix.lower() in {".png", ".pdf", ".csv"}:
      lines.append(f"- `{path.name}`")
  (outdir / "plot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
  args = parse_args()
  setup_style()
  analysis_dir = Path(args.analysis_dir).expanduser()
  outdir = Path(args.outdir).expanduser() if args.outdir else analysis_dir / "figures"
  outdir.mkdir(parents=True, exist_ok=True)
  formats = [x.strip() for x in args.formats.split(",") if x.strip()]

  summary = add_method(coerce_numeric(load_csv(analysis_dir / "summary_metrics.csv")))
  context = add_method(coerce_numeric(load_csv(analysis_dir / "context_conditioned_metrics.csv")))
  trace = add_method(coerce_numeric(load_csv(analysis_dir / "trace_ranking_metrics.csv")))
  events = add_method(coerce_numeric(load_csv(analysis_dir / "event_window_fault_score.csv")))

  focus = focus_frame(summary, args.focus_suite)
  if focus.empty:
    raise SystemExit(f"No focus rows found for suite: {args.focus_suite}")

  generated = []
  generated += plot_long_eval_bars(focus, outdir, formats)
  generated += plot_task_tradeoffs(focus, outdir, formats)
  generated += plot_detection_false_alarm(focus, outdir, formats)
  generated += plot_trace_precision(trace, outdir, formats, args.focus_suite)
  generated += plot_context_conditioned(context, outdir, formats, args.focus_suite)
  generated += plot_context_breakdown(context, outdir, formats, args.focus_suite)
  generated += plot_event_window(events, outdir, formats, args.focus_suite)
  generated += plot_pareto(summary, outdir, formats, args.focus_suite)

  write_tables_and_findings(summary, context, trace, outdir, args.focus_suite)

  print(f"Wrote figures and tables to: {outdir}")
  for path in generated:
    print(path)
  print(outdir / "plot_summary.md")


if __name__ == "__main__":
  main()

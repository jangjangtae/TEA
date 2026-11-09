#!/usr/bin/env python3
"""Write a compact report for balanced Craftax three-seed analyses."""

import argparse
import csv
import math
from pathlib import Path


CORE_SPLITS = ("seen", "holdout", "sparse")
CORE_METRICS = (
    "episode_score_mean_mean",
    "fault_applied_rate_mean",
    "bug_events_per_10k_mean",
    "unique_bug_types_mean",
    "bug_type_coverage_fraction_mean",
    "bug_discovery_auc_norm_mean",
    "time_to_first_bug_steps_mean",
    "auroc_mean",
    "auprc_mean",
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--analysis-root", required=True)
  parser.add_argument("--milestone", type=int, default=1000000)
  parser.add_argument("--baseline", default="taskonly")
  parser.add_argument("--outdir", default="")
  return parser.parse_args()


def number(value):
  try:
    value = float(value)
  except (TypeError, ValueError):
    return float("nan")
  return value if math.isfinite(value) else float("nan")


def fmt(value, digits=4):
  value = number(value)
  if not math.isfinite(value):
    return "nan"
  return f"{value:.{digits}f}"


def fmt_pm(row, metric, digits=3):
  mean = number(row.get(metric))
  std = number(row.get(std_key(metric)))
  if not math.isfinite(mean):
    return "nan"
  if not math.isfinite(std):
    return fmt(mean, digits)
  return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def std_key(metric):
  if metric.endswith("_mean_mean"):
    return metric[:-len("_mean_mean")] + "_mean_std"
  if metric.endswith("_mean"):
    return metric[:-len("_mean")] + "_std"
  return metric + "_std"


def compact_metric_name(metric):
  if metric.endswith("_mean_mean"):
    return metric[:-len("_mean_mean")]
  if metric.endswith("_mean"):
    return metric[:-len("_mean")]
  return metric


def read_csv(path):
  with path.open(newline="", encoding="utf-8") as f:
    return list(csv.DictReader(f))


def write_csv(path, rows, fields):
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def best(rows, split, metric, larger=True):
  candidates = [
      row for row in rows
      if row.get("phase") == "eval"
      and row.get("split") == split
      and row.get("variant") != "reference"
      and math.isfinite(number(row.get(metric)))
  ]
  if not candidates:
    return None
  return sorted(candidates, key=lambda row: number(row.get(metric)), reverse=larger)[0]


def final_key_rows(rows, baseline):
  indexed = {
      (row["variant"], row["split"]): row for row in rows
      if row.get("phase") == "eval" and row.get("variant") != "reference"
  }
  result = []
  for split in CORE_SPLITS:
    base = indexed.get((baseline, split), {})
    for (variant, row_split), row in sorted(indexed.items()):
      if row_split != split:
        continue
      item = {
          "split": split,
          "variant": variant,
          "num_seeds": row.get("num_seeds", ""),
      }
      for metric in CORE_METRICS:
        name = compact_metric_name(metric)
        item[f"{name}_mean"] = row.get(metric, "")
        item[f"{name}_std"] = row.get(std_key(metric), "")
        item[f"{name}_mean_pm_std"] = fmt_pm(row, metric)
        base_value = number(base.get(metric))
        value = number(row.get(metric))
        item[f"{name}_delta_vs_{baseline}"] = (
            value - base_value
            if math.isfinite(value) and math.isfinite(base_value)
            else float("nan"))
      result.append(item)
  return result


def milestone_best_rows(rows):
  result = []
  for split in CORE_SPLITS:
    for milestone in sorted({int(row["milestone"]) for row in rows}):
      subset = [
          row for row in rows
          if int(row["milestone"]) == milestone
          and row.get("split") == split
          and row.get("phase") == "eval"
          and row.get("variant") != "reference"
      ]
      if not subset:
        continue
      by_bug = sorted(
          subset, key=lambda row: number(row.get("fault_applied_rate_mean")),
          reverse=True)[0]
      by_task = sorted(
          subset, key=lambda row: number(row.get("episode_score_mean_mean")),
          reverse=True)[0]
      by_ttfb = sorted(
          [row for row in subset if math.isfinite(number(row.get("time_to_first_bug_steps_mean")))],
          key=lambda row: number(row.get("time_to_first_bug_steps_mean")))
      result.append({
          "split": split,
          "milestone": milestone,
          "best_bug_variant": by_bug["variant"],
          "best_bug_rate": by_bug.get("fault_applied_rate_mean", ""),
          "best_task_variant": by_task["variant"],
          "best_task_score": by_task.get("episode_score_mean_mean", ""),
          "fastest_first_bug_variant": by_ttfb[0]["variant"] if by_ttfb else "",
          "fastest_first_bug_steps": by_ttfb[0].get("time_to_first_bug_steps_mean", "") if by_ttfb else "",
      })
  return result


def write_report(path, final_rows, milestone_rows, baseline):
  lines = [
      "# Craftax Three-Seed Analysis",
      "",
      f"Balanced seeds: 0, 1, 2. Baseline: `{baseline}`.",
      "",
      "## Final 1M-Step Highlights",
      "",
  ]
  for split in CORE_SPLITS:
    bug = best(final_rows, split, "fault_applied_rate_mean", True)
    task = best(final_rows, split, "episode_score_mean_mean", True)
    auroc = best(final_rows, split, "auroc_mean", True)
    ttfb = best(final_rows, split, "time_to_first_bug_steps_mean", False)
    lines.extend([
        f"### {split}",
        "",
        (
            f"- Highest bug rate: `{bug['variant']}` "
            f"({fmt(bug.get('fault_applied_rate_mean'), 6)}, "
            f"{fmt(bug.get('bug_events_per_10k_mean'), 2)} events/10k)."
            if bug else "- Highest bug rate: unavailable."
        ),
        (
            f"- Best task score: `{task['variant']}` "
            f"({fmt(task.get('episode_score_mean_mean'), 3)})."
            if task else "- Best task score: unavailable."
        ),
        (
            f"- Fastest first bug: `{ttfb['variant']}` "
            f"({fmt(ttfb.get('time_to_first_bug_steps_mean'), 1)} steps)."
            if ttfb else "- Fastest first bug: unavailable."
        ),
        (
            f"- Best AUROC: `{auroc['variant']}` "
            f"({fmt(auroc.get('auroc_mean'), 3)})."
            if auroc else "- Best AUROC: unavailable."
        ),
        "",
    ])
  lines.extend([
      "## Final 1M-Step Mean +/- Std",
      "",
      "| Split | Variant | Task score | Bug events / 10k | Coverage | Discovery AUC | First bug steps | AUROC |",
      "|---|---|---:|---:|---:|---:|---:|---:|",
  ])
  for split in CORE_SPLITS:
    rows = [
        row for row in final_rows
        if row.get("phase") == "eval"
        and row.get("split") == split
        and row.get("variant") != "reference"
    ]
    for row in sorted(rows, key=lambda item: item["variant"]):
      lines.append(
          f"| {split} | `{row['variant']}` | "
          f"{fmt_pm(row, 'episode_score_mean_mean', 3)} | "
          f"{fmt_pm(row, 'bug_events_per_10k_mean', 2)} | "
          f"{fmt_pm(row, 'bug_type_coverage_fraction_mean', 2)} | "
          f"{fmt_pm(row, 'bug_discovery_auc_norm_mean', 3)} | "
          f"{fmt_pm(row, 'time_to_first_bug_steps_mean', 1)} | "
          f"{fmt_pm(row, 'auroc_mean', 3)} |")
  lines.extend([
      "",
      "## Reading",
      "",
      "- `contextual_excess_delta_beta02` is the clearest bug-seeking policy: it raises bug rate, expected bug-type coverage, and first-bug speed on seen/holdout/sparse while also keeping high task score.",
      "- `excess_delta_p95_beta02` is more conservative: task score remains good and AUROC can improve on seen/holdout, but it does not consistently raise bug event rate.",
      "- CRL/KL-bound variants are excluded from the AAAI main story; the deadline-focused comparison is the excess-delta method and its calibration/reward ablations.",
      "- Detection metrics and policy metrics diverge: the best bug-seeking variant is not necessarily the best AUROC variant. This supports reporting both tester behavior and fault-score separability.",
      "",
      "## Generated Tables",
      "",
      "- `three_seed_key_metrics.csv`: final 1M metrics and deltas versus task-only.",
      "- `three_seed_milestone_best.csv`: best variant per split and milestone.",
      "",
  ])
  path.write_text("\n".join(lines), encoding="utf-8")


def main():
  args = parse_args()
  analysis_root = Path(args.analysis_root).expanduser()
  outdir = Path(args.outdir).expanduser() if args.outdir else analysis_root
  outdir.mkdir(parents=True, exist_ok=True)

  final_path = analysis_root / f"milestone_{args.milestone}" / "aggregate_metrics.csv"
  milestone_path = analysis_root / "milestones" / "milestone_metrics.csv"
  final_rows = read_csv(final_path)
  milestone_rows = read_csv(milestone_path)

  key_rows = final_key_rows(final_rows, args.baseline)
  key_fields = ["split", "variant", "num_seeds"]
  for metric in CORE_METRICS:
    name = compact_metric_name(metric)
    key_fields += [
        f"{name}_mean", f"{name}_std", f"{name}_mean_pm_std",
        f"{name}_delta_vs_{args.baseline}"]
  write_csv(outdir / "three_seed_key_metrics.csv", key_rows, key_fields)

  best_rows = milestone_best_rows(milestone_rows)
  write_csv(outdir / "three_seed_milestone_best.csv", best_rows, [
      "split", "milestone", "best_bug_variant", "best_bug_rate",
      "best_task_variant", "best_task_score",
      "fastest_first_bug_variant", "fastest_first_bug_steps"])
  write_report(outdir / "three_seed_report.md", final_rows, milestone_rows, args.baseline)
  print(f"Wrote {outdir / 'three_seed_report.md'}")
  print(f"Wrote {outdir / 'three_seed_key_metrics.csv'}")
  print(f"Wrote {outdir / 'three_seed_milestone_best.csv'}")


if __name__ == "__main__":
  main()

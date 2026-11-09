#!/usr/bin/env python3
"""Analyze MiniGrid multi-seed fault-seeking runs.

This reuses the generic trace aggregation from the Craftax analyzer, but
overrides the expected fault taxonomy and bug-id mapping for MiniGrid.
"""

from dreamerv3 import analyze_craftax_multiseed as base


MINIGRID_BUG_TYPE_NAMES = {
    0: "none",
    1: "broken_door",
    2: "heavy_key",
    3: "action_flip",
    4: "teleport",
    5: "door_gone",
    6: "lava_gap",
}


def minigrid_bug_type(row):
  try:
    bug_id = int(row.get("bug_id", row.get("log/bug_type_id", 0)))
  except Exception:
    bug_id = 0
  bug_type = MINIGRID_BUG_TYPE_NAMES.get(bug_id, "")
  if bug_type and bug_type != "none":
    return bug_type
  bug_type = str(row.get("bug_type", "")).strip()
  if bug_type.lower() in ("", "none", "null", "nan", "unknown"):
    return ""
  return bug_type


base.EXPECTED_BUG_TYPES = {
    "seen": {"broken_door", "heavy_key", "action_flip"},
    "holdout": {"teleport", "door_gone", "lava_gap"},
    "sparse": {
        "broken_door", "heavy_key", "action_flip",
        "teleport", "door_gone", "lava_gap",
    },
}
base.canonical_bug_type = minigrid_bug_type


if __name__ == "__main__":
  base.main()

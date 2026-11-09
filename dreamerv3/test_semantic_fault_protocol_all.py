#!/usr/bin/env python3
"""Protocol smoke tests for all configured Crafter semantic fault subtypes."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from dreamerv3 import test_crafter_hud_red_semantic7 as hud


SEMANTIC_SUBTYPES = [
    'tool_collect_desync_on_upgrade',
    'upgrade_branch_inconsistent_collect_behavior',
    'collect_result_delayed_after_tool_upgrade',
    'craft_result_missing_on_retry',
    'craft_output_delayed_on_retry',
    'recipe_retry_requires_revisit',
    'recipe_precondition_mischeck_on_retry',
    'station_place_ghost_on_relocate',
    'station_state_partial_reset_after_relocate',
    'station_usable_flag_broken_after_relocate',
    'station_second_use_inconsistent_after_placement',
    'achievement_unlock_missing_after_valid_progress',
    'achievement_unlock_missing_after_reconfirm',
    'progress_confirmation_requires_revisit',
    'delayed_inventory_desync_after_station_use',
]


def trigger_progress_immediate(raw, recorder):
  hud.set_material(raw, hud.rel(raw, 'left'), 'table')
  hud.set_inventory(raw, wood=1)
  recorder.step('make_wood_pickaxe', 'make_wood_pickaxe_valid_progress')


def trigger_station_second_use(raw, recorder):
  hud.set_inventory(raw, wood=5)
  hud.set_material(raw, hud.rel(raw, 'right'), 'grass')
  hud.set_facing(raw, 'right')
  recorder.step('place_table', 'place_table_arm_second_use')
  raw._player.inventory['wood'] = 4
  recorder.step('make_wood_pickaxe', 'first_station_use')
  raw._player.inventory['wood'] = 4
  recorder.step('make_wood_pickaxe', 'second_station_use_inconsistent')


def trigger_delayed_inventory(raw, recorder, expect_manifest):
  hud.set_material(raw, hud.rel(raw, 'left'), 'table')
  hud.set_inventory(raw, wood=1)
  recorder.step('make_wood_pickaxe', 'make_wood_pickaxe_schedule_delayed_desync')
  for index in range(1, 12):
    recorder.step('noop', f'wait_for_delayed_desync_{index}')
    if expect_manifest and any(r.get('fault_manifested', 0) for r in recorder.records):
      return


def run_trigger(subtype, raw, recorder, expect_manifest):
  if subtype in (
      'tool_collect_desync_on_upgrade',
      'upgrade_branch_inconsistent_collect_behavior',
      'collect_result_delayed_after_tool_upgrade'):
    hud.trigger_upgrade_collect(raw, recorder)
  elif subtype in (
      'craft_result_missing_on_retry',
      'craft_output_delayed_on_retry',
      'recipe_retry_requires_revisit',
      'recipe_precondition_mischeck_on_retry'):
    hud.trigger_retry_craft(raw, recorder, subtype)
  elif subtype == 'station_place_ghost_on_relocate':
    hud.trigger_station_place_ghost(raw, recorder)
  elif subtype in (
      'station_state_partial_reset_after_relocate',
      'station_usable_flag_broken_after_relocate'):
    hud.trigger_station_usable_broken(raw, recorder)
  elif subtype == 'station_second_use_inconsistent_after_placement':
    trigger_station_second_use(raw, recorder)
  elif subtype == 'achievement_unlock_missing_after_valid_progress':
    trigger_progress_immediate(raw, recorder)
  elif subtype in (
      'achievement_unlock_missing_after_reconfirm',
      'progress_confirmation_requires_revisit'):
    hud.trigger_progress_reconfirm(raw, recorder)
  elif subtype == 'delayed_inventory_desync_after_station_use':
    trigger_delayed_inventory(raw, recorder, expect_manifest)
  else:
    raise KeyError(subtype)


def summarize_records(records):
  return {
      'fault_exists_episode': int(any(r.get('fault_exists_episode', 0) for r in records)),
      'fault_trigger_context': int(any(r.get('fault_trigger_context', 0) for r in records)),
      'semantic_trigger_context': int(any(r.get('semantic_trigger_context', 0) for r in records)),
      'fault_manifested': int(any(r.get('fault_manifested', 0) for r in records)),
      'semantic_fault_applied': int(any(r.get('semantic_fault_applied', 0) for r in records)),
      'max_manifest_prob': float(max(
          [float(r.get('fault_manifest_prob', 0.0)) for r in records] or [0.0])),
  }


def run_case(root, subtype, seed, manifest_prob):
  os.environ['CRAFTER_SEMANTIC_FAULT_MANIFEST_PROB'] = str(manifest_prob)
  env, raw, recorder = hud.setup_env_for_case(root / subtype, subtype, seed)
  expect_manifest = manifest_prob > 0.999
  run_trigger(subtype, raw, recorder, expect_manifest=expect_manifest)
  summary = summarize_records(recorder.records)

  assert summary['fault_exists_episode'] == 1, (subtype, summary)
  assert summary['fault_trigger_context'] == 1, (subtype, summary)
  assert summary['semantic_trigger_context'] == 1, (subtype, summary)
  if expect_manifest:
    assert summary['fault_manifested'] == 1, (subtype, summary)
    assert summary['semantic_fault_applied'] == 1, (subtype, summary)
  else:
    assert summary['fault_manifested'] == 0, (subtype, summary)
    assert summary['semantic_fault_applied'] == 0, (subtype, summary)
  return summary


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--outdir', default='/tmp/crafter_semantic_protocol_all')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--manifest-prob', type=float, default=1.0)
  args = parser.parse_args()

  root = Path(args.outdir).expanduser().resolve()
  root.mkdir(parents=True, exist_ok=True)

  results = {}
  for index, subtype in enumerate(SEMANTIC_SUBTYPES):
    summary = run_case(root, subtype, args.seed + index, args.manifest_prob)
    results[subtype] = summary
    print(f'PASS: {subtype} {summary}')

  report = {
      'manifest_prob': float(args.manifest_prob),
      'cases': results,
  }
  with (root / 'semantic_protocol_summary.json').open('w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
  print(f'PASS: all {len(results)} semantic protocol checks passed')
  print(f'Output directory: {root}')


if __name__ == '__main__':
  main()

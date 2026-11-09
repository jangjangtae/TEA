#!/usr/bin/env python3
"""Generate real Crafter HUD frames for the 7 semantic holdout bugs."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


SEMANTIC7 = [
    'upgrade_branch_inconsistent_collect_behavior',
    'craft_result_missing_on_retry',
    'station_place_ghost_on_relocate',
    'achievement_unlock_missing_after_reconfirm',
    'station_usable_flag_broken_after_relocate',
    'recipe_precondition_mischeck_on_retry',
    'delayed_inventory_desync_after_station_use',
]


OFFSETS = {
    'left': np.array((-1, 0)),
    'right': np.array((1, 0)),
    'up': np.array((0, -1)),
    'down': np.array((0, 1)),
}


def configure_env(outdir, subtype):
  os.environ['CRAFTER_OUTPUT_DIR'] = str(outdir)
  os.environ['CRAFTER_RECORD_GIFS'] = '1'
  os.environ['CRAFTER_FAULT'] = '0'
  os.environ['CRAFTER_FAULT_SAMPLER'] = '0'
  os.environ['CRAFTER_TESTER_REWARD'] = '0'
  os.environ['CRAFTER_USE_RND'] = '0'
  os.environ['CRAFTER_SEMANTIC_FAULT_SAMPLER'] = '1'
  os.environ['CRAFTER_SEMANTIC_FAULT_PROFILE'] = 'eval_holdout'
  os.environ['CRAFTER_SEMANTIC_FAULT_EP_PROB'] = '1.0'
  os.environ['CRAFTER_SEMANTIC_SUBTYPES'] = subtype
  os.environ['CRAFTER_SEMANTIC_FAULT_VERBOSE'] = '0'
  os.environ['CRAFTER_SEMANTIC_RETRY_GAP'] = '40'


def action_index(env, name):
  return env._env.action_names.index(name)


def rgb_mean(frame):
  return tuple(float(frame[..., channel].mean()) for channel in range(3))


def is_red_enough(frame):
  red, green, blue = rgb_mean(frame)
  return red >= 170 and red >= green * 2.0 and red >= blue * 2.0


def save_png(frame, path):
  from PIL import Image
  path.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(frame).save(path)


def save_gif(frames, path):
  try:
    import imageio.v2 as imageio
  except ImportError:
    return None
  path.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(str(path), frames, fps=2)
  return path


def clear_obj(raw, pos):
  _, obj = raw._world[tuple(pos)]
  if obj is not None and obj is not raw._player:
    raw._world.remove(obj)


def set_material(raw, pos, material):
  pos = tuple(np.array(pos, dtype=int))
  clear_obj(raw, pos)
  raw._world[pos] = material


def prepare_local_area(raw):
  player = raw._player
  player.inventory['health'] = 9
  player.inventory['food'] = 9
  player.inventory['drink'] = 9
  player.inventory['energy'] = 9
  raw._last_health = player.health

  px, py = [int(x) for x in player.pos]
  for dx in range(-4, 5):
    for dy in range(-4, 5):
      pos = (px + dx, py + dy)
      if tuple(pos) == tuple(player.pos):
        continue
      set_material(raw, pos, 'grass')


def set_inventory(raw, **items):
  for key in raw._player.inventory:
    if key not in ('health', 'food', 'drink', 'energy'):
      raw._player.inventory[key] = 0
  raw._player.inventory['health'] = 9
  raw._player.inventory['food'] = 9
  raw._player.inventory['drink'] = 9
  raw._player.inventory['energy'] = 9
  for key, value in items.items():
    raw._player.inventory[key] = int(value)


def set_facing(raw, direction):
  raw._player.facing = OFFSETS[direction].copy()


def rel(raw, direction, distance=1):
  return tuple((np.array(raw._player.pos) + OFFSETS[direction] * distance).astype(int))


def move_player(raw, direction, distance=3):
  target = rel(raw, direction, distance)
  set_material(raw, target, 'grass')
  raw._world.move(raw._player, np.array(target))


class CaseRecorder:

  def __init__(self, env, outdir):
    self.env = env
    self.outdir = outdir
    self.records = []

  def reset(self):
    obs = self.env.step({'reset': True, 'action': np.int32(0)})
    self._capture('reset', obs)
    return obs

  def step(self, action_name, label=None):
    obs = self.env.step({
        'reset': False,
        'action': np.int32(action_index(self.env, action_name)),
    })
    self._capture(label or action_name, obs)
    return obs

  def _capture(self, label, obs):
    frame = self.env._episode_frames[-1]
    fault_applied = int(obs.get('log/fault_applied', 0))
    semantic_applied = int(obs.get('log/semantic_fault_applied', 0))
    record = {
        'index': len(self.records),
        'label': label,
        'fault_applied': fault_applied,
        'fault_exists_episode': int(obs.get('log/fault_exists_episode', 0)),
        'fault_trigger_context': int(obs.get('log/fault_trigger_context', 0)),
        'fault_manifested': int(obs.get('log/fault_manifested', fault_applied)),
        'fault_manifest_prob': float(obs.get('log/fault_manifest_prob', 0.0)),
        'semantic_fault_applied': semantic_applied,
        'semantic_trigger_context': int(obs.get('log/semantic_trigger_context', 0)),
        'mean_rgb': rgb_mean(frame),
        'frame': frame,
    }
    self.records.append(record)

  def save(self, subtype):
    frames_dir = self.outdir / 'frames'
    for record in self.records:
      path = frames_dir / f"{record['index']:03d}_{record['label']}.png"
      save_png(record['frame'], path)
      record['path'] = str(path)
      record.pop('frame')

    first_bug = next(
        (record for record in self.records if record['semantic_fault_applied']),
        None)
    if first_bug is None:
      raise AssertionError(f'{subtype}: no semantic fault frame was recorded')

    first_bug_frame = np.array(
        __import__('PIL.Image').Image.open(first_bug['path']))
    if not is_red_enough(first_bug_frame):
      raise AssertionError(
          f"{subtype}: first bug frame is not red enough: "
          f"mean RGB={rgb_mean(first_bug_frame)}")

    bug_path = self.outdir / f'{subtype}_bug_frame.png'
    save_png(first_bug_frame, bug_path)
    gif_path = save_gif(
        [np.array(__import__('PIL.Image').Image.open(record['path']))
         for record in self.records],
        self.outdir / f'{subtype}.gif')

    manifest = {
        'subtype': subtype,
        'bug_frame': str(bug_path),
        'gif': str(gif_path) if gif_path is not None else None,
        'first_bug_index': int(first_bug['index']),
        'frames': self.records,
    }
    with (self.outdir / 'manifest.json').open('w', encoding='utf-8') as f:
      json.dump(manifest, f, indent=2)
    return manifest


def setup_env_for_case(outdir, subtype, seed):
  configure_env(outdir, subtype)
  from embodied.envs.crafter import Crafter
  env = Crafter(
      task='reward',
      size=(64, 64),
      logs=False,
      logdir=str(outdir),
      seed=seed,
  )
  recorder = CaseRecorder(env, outdir)
  recorder.reset()
  raw = env._env
  prepare_local_area(raw)
  spec = raw._semantic_fault_spec
  if not spec or spec.get('type') != subtype:
    raise AssertionError(f'{subtype}: unexpected semantic spec: {spec}')
  return env, raw, recorder


def trigger_upgrade_collect(raw, recorder):
  set_material(raw, rel(raw, 'left'), 'table')
  set_inventory(raw, wood=1)
  recorder.step('make_wood_pickaxe', 'make_wood_pickaxe_arm_collect_bug')
  set_material(raw, rel(raw, 'right'), 'stone')
  set_facing(raw, 'right')
  recorder.step('do', 'collect_stone_after_pickaxe_upgrade')


def trigger_retry_craft(raw, recorder, subtype):
  set_material(raw, rel(raw, 'left'), 'table')
  set_inventory(raw, wood=2)
  recorder.step('make_wood_pickaxe', 'first_make_wood_pickaxe')
  recorder.step('make_wood_pickaxe', f'{subtype}_retry_make_wood_pickaxe')


def trigger_station_place_ghost(raw, recorder):
  set_inventory(raw, wood=4)
  set_material(raw, rel(raw, 'right'), 'grass')
  set_facing(raw, 'right')
  recorder.step('place_table', 'first_place_table')
  set_material(raw, rel(raw, 'left'), 'grass')
  set_facing(raw, 'left')
  recorder.step('place_table', 'second_place_table_ghost')


def trigger_progress_reconfirm(raw, recorder):
  set_material(raw, rel(raw, 'left'), 'table')
  set_inventory(raw, wood=1)
  recorder.step('make_wood_pickaxe', 'make_wood_pickaxe_progress_removed')
  move_player(raw, 'down', distance=3)
  set_facing(raw, 'right')
  set_material(raw, rel(raw, 'right'), 'grass')
  recorder.step('do', 'reconfirm_progress_after_revisit')


def trigger_station_usable_broken(raw, recorder):
  set_inventory(raw, wood=5)
  set_material(raw, rel(raw, 'right'), 'grass')
  set_facing(raw, 'right')
  recorder.step('place_table', 'first_place_table')
  set_material(raw, rel(raw, 'left'), 'grass')
  set_facing(raw, 'left')
  recorder.step('place_table', 'second_place_table_broken')
  raw._player.inventory['wood'] = 1
  recorder.step('make_wood_pickaxe', 'make_with_broken_relocated_table')


def trigger_delayed_inventory(raw, recorder):
  set_material(raw, rel(raw, 'left'), 'table')
  set_inventory(raw, wood=1)
  recorder.step('make_wood_pickaxe', 'make_wood_pickaxe_schedule_delayed_desync')
  for index in range(1, 12):
    obs = recorder.step('noop', f'wait_for_delayed_desync_{index}')
    if int(obs.get('log/semantic_fault_applied', 0)):
      return
  raise AssertionError('delayed inventory desync did not fire within 11 noop steps')


def run_case(root, subtype, seed):
  case_dir = root / subtype
  env, raw, recorder = setup_env_for_case(case_dir, subtype, seed)

  if subtype == 'upgrade_branch_inconsistent_collect_behavior':
    trigger_upgrade_collect(raw, recorder)
  elif subtype == 'craft_result_missing_on_retry':
    trigger_retry_craft(raw, recorder, subtype)
  elif subtype == 'station_place_ghost_on_relocate':
    trigger_station_place_ghost(raw, recorder)
  elif subtype == 'achievement_unlock_missing_after_reconfirm':
    trigger_progress_reconfirm(raw, recorder)
  elif subtype == 'station_usable_flag_broken_after_relocate':
    trigger_station_usable_broken(raw, recorder)
  elif subtype == 'recipe_precondition_mischeck_on_retry':
    trigger_retry_craft(raw, recorder, subtype)
  elif subtype == 'delayed_inventory_desync_after_station_use':
    trigger_delayed_inventory(raw, recorder)
  else:
    raise KeyError(subtype)

  return recorder.save(subtype)


def save_contact_sheet(manifests, path):
  from PIL import Image, ImageDraw
  thumbs = []
  for manifest in manifests:
    image = Image.open(manifest['bug_frame']).convert('RGB').resize((256, 256))
    thumbs.append((manifest['subtype'], image))

  width = 2 * 360
  height = 4 * 300
  sheet = Image.new('RGB', (width, height), (20, 20, 20))
  draw = ImageDraw.Draw(sheet)
  for idx, (name, image) in enumerate(thumbs):
    col = idx % 2
    row = idx // 2
    x = col * 360
    y = row * 300
    sheet.paste(image, (x, y))
    draw.text((x + 8, y + 260), name, fill=(255, 255, 255))
  path.parent.mkdir(parents=True, exist_ok=True)
  sheet.save(path)
  return path


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--outdir',
      default='debug/crafter_hud_red_semantic7',
      help='Directory for generated semantic 7 PNG/GIF files.')
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()

  root = Path(args.outdir).expanduser().resolve()
  root.mkdir(parents=True, exist_ok=True)

  manifests = []
  for index, subtype in enumerate(SEMANTIC7):
    manifest = run_case(root, subtype, seed=args.seed + index)
    manifests.append(manifest)
    print(
        f"PASS: {subtype} "
        f"bug_frame={manifest['bug_frame']} gif={manifest['gif']}")

  contact_sheet = save_contact_sheet(
      manifests, root / 'semantic7_contact_sheet.png')
  summary = {
      'outdir': str(root),
      'contact_sheet': str(contact_sheet),
      'cases': manifests,
  }
  with (root / 'summary.json').open('w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

  print('PASS: generated all 7 semantic holdout bug visual checks.')
  print(f'Output directory: {root}')
  print(f'Contact sheet: {contact_sheet}')


if __name__ == '__main__':
  main()

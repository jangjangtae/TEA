#!/usr/bin/env python3
"""Smoke test for Crafter HUD red bug frames.

This avoids running a full environment rollout. It calls Crafter._record_hud_frame
directly with synthetic frames and verifies that bug frames become visibly red.
"""

import argparse
from pathlib import Path

import numpy as np

from embodied.envs.crafter import Crafter


def make_image(size=64):
  xs = np.linspace(0, 255, size, dtype=np.uint8)
  ys = np.linspace(0, 255, size, dtype=np.uint8)
  grid_x, grid_y = np.meshgrid(xs, ys)
  blue = np.full_like(grid_x, 80)
  return np.stack([grid_x, grid_y, blue], axis=-1)


def make_env_stub():
  env = Crafter.__new__(Crafter)
  env._episode_frames = []
  env._length = 0
  env._fault_spec = {'type': 'target_dummy_fault'}
  return env


def save_frame(frame, path):
  path.parent.mkdir(parents=True, exist_ok=True)
  try:
    from PIL import Image
    Image.fromarray(frame).save(path)
    return path
  except ImportError:
    npy_path = path.with_suffix('.npy')
    np.save(npy_path, frame)
    return npy_path


def rgb_mean(frame):
  return tuple(float(frame[..., channel].mean()) for channel in range(3))


def record_case(env, name, info, outdir):
  env._length += 1
  env._record_hud_frame(make_image(), requested_action=0, info=info)
  frame = env._episode_frames[-1]
  path = save_frame(frame, outdir / f'{name}.png')
  return frame, path


def maybe_save_gif(frames, outdir):
  try:
    import imageio.v2 as imageio
    gif_path = outdir / 'hud_red_frame_test.gif'
    imageio.mimsave(str(gif_path), frames, fps=1)
    return gif_path
  except ImportError:
    return None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--outdir',
      default='/tmp/crafter_hud_red_frame_test',
      help='Directory for generated PNG/GIF files.')
  args = parser.parse_args()

  outdir = Path(args.outdir).expanduser()
  env = make_env_stub()
  inventory = {
      'wood': 1,
      'stone': 2,
      'coal': 0,
      'iron': 0,
      'table': 1,
      'furnace': 0,
      'wood_pickaxe': 1,
  }

  cases = {
      'normal_frame': {
          'fault_applied': 0,
          'semantic_fault_applied': 0,
          'fault_type': 'none',
          'inventory': inventory,
      },
      'fault_frame': {
          'fault_applied': 1,
          'semantic_fault_applied': 0,
          'fault_type': 'dummy_action_fault',
          'inventory': inventory,
      },
      'semantic_fault_frame': {
          'fault_applied': 1,
          'semantic_fault_applied': 1,
          'fault_type': 'dummy_semantic_fault',
          'semantic_fault_type': 'dummy_semantic_fault',
          'inventory': inventory,
      },
  }

  outputs = {}
  for name, info in cases.items():
    frame, path = record_case(env, name, info, outdir)
    outputs[name] = (frame, path, rgb_mean(frame))

  gif_path = maybe_save_gif([item[0] for item in outputs.values()], outdir)

  normal_mean = outputs['normal_frame'][2]
  fault_mean = outputs['fault_frame'][2]
  semantic_mean = outputs['semantic_fault_frame'][2]

  for name, mean in (
      ('fault_frame', fault_mean),
      ('semantic_fault_frame', semantic_mean),
  ):
    red, green, blue = mean
    if red < 170 or red < green * 2.0 or red < blue * 2.0:
      raise AssertionError(
          f'{name} is not red enough: mean RGB={mean}')

  print('PASS: HUD bug frames are visibly red.')
  print(f'Output directory: {outdir}')
  for name, (_, path, mean) in outputs.items():
    print(f'{name}: {path} mean_rgb={mean}')
  if gif_path is not None:
    print(f'gif: {gif_path}')
  else:
    print('gif: skipped because imageio is not installed')
  print(f'normal_mean_rgb={normal_mean}')


if __name__ == '__main__':
  main()

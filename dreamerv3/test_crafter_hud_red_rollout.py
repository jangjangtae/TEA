#!/usr/bin/env python3
"""Run a real Crafter rollout and save HUD frames for visual checking."""

import argparse
import os
from pathlib import Path

import numpy as np


def configure_env(outdir):
  os.environ['CRAFTER_OUTPUT_DIR'] = str(outdir)
  os.environ['CRAFTER_RECORD_GIFS'] = '1'
  os.environ['CRAFTER_FAULT'] = '1'
  os.environ['CRAFTER_FAULT_PROFILE'] = 'train'
  os.environ['CRAFTER_FAULT_COOLDOWN'] = '0'
  os.environ['CRAFTER_ACTION_DROP_PROB'] = '1.0'
  os.environ['CRAFTER_FALLBACK_ACTION'] = '0'
  os.environ['CRAFTER_FAULT_VERBOSE'] = '0'
  os.environ['CRAFTER_TESTER_REWARD'] = '0'
  os.environ['CRAFTER_USE_RND'] = '0'


def save_png(frame, path):
  try:
    from PIL import Image
  except ImportError as exc:
    raise RuntimeError('PIL is required to save PNG frames.') from exc
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


def rgb_mean(frame):
  return tuple(float(frame[..., channel].mean()) for channel in range(3))


def is_red_enough(mean_rgb):
  red, green, blue = mean_rgb
  return red >= 170 and red >= green * 2.0 and red >= blue * 2.0


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--outdir',
      default='debug/crafter_hud_red_rollout',
      help='Directory for generated real-Crafter frames.')
  parser.add_argument('--steps', type=int, default=5)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument(
      '--action',
      type=int,
      default=1,
      help='Requested action. The test forces execution to fallback action 0.')
  args = parser.parse_args()

  outdir = Path(args.outdir).expanduser().resolve()
  configure_env(outdir)

  from embodied.envs.crafter import Crafter

  env = Crafter(
      task='reward',
      size=(64, 64),
      logs=False,
      logdir=str(outdir),
      seed=args.seed,
  )

  obs = env.step({'reset': True, 'action': np.int32(0)})
  if not env._episode_frames:
    raise AssertionError('Reset did not record a HUD frame.')

  reset_frame = env._episode_frames[-1]
  save_png(reset_frame, outdir / 'real_reset_frame.png')

  bug_frames = []
  fault_flags = []
  for step in range(args.steps):
    obs = env.step({'reset': False, 'action': np.int32(args.action)})
    fault_applied = int(obs['log/fault_applied'])
    fault_flags.append(fault_applied)
    frame = env._episode_frames[-1]
    bug_frames.append(frame)
    save_png(frame, outdir / f'real_step_{step + 1:03d}.png')

  gif_path = save_gif([reset_frame] + bug_frames, outdir / 'real_crafter_hud_red_rollout.gif')

  first_bug_mean = rgb_mean(bug_frames[0])
  if not any(fault_flags):
    raise AssertionError(f'No fault was applied. fault_flags={fault_flags}')
  if not is_red_enough(first_bug_mean):
    raise AssertionError(
        f'First real fault frame is not red enough: mean RGB={first_bug_mean}')

  print('PASS: real Crafter rollout produced red HUD frames on fault steps.')
  print(f'Output directory: {outdir}')
  print(f'reset: {outdir / "real_reset_frame.png"} mean_rgb={rgb_mean(reset_frame)}')
  for step, (flag, frame) in enumerate(zip(fault_flags, bug_frames), start=1):
    print(
        f'step {step}: fault_applied={flag} '
        f'path={outdir / f"real_step_{step:03d}.png"} '
        f'mean_rgb={rgb_mean(frame)}')
  if gif_path is not None:
    print(f'gif: {gif_path}')
  else:
    print('gif: skipped because imageio is not installed')


if __name__ == '__main__':
  main()

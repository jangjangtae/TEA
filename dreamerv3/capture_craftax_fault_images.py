#!/usr/bin/env python3
"""Capture Craftax clean and fault-manifestation frames for paper figures."""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ACTION_NAMES = {
    0: "noop",
    1: "left",
    2: "right",
    3: "up",
    4: "down",
    5: "do",
    6: "sleep",
    7: "place stone",
    8: "place table",
    9: "place furnace",
    10: "place plant",
    11: "make wood pickaxe",
    12: "make stone pickaxe",
    13: "make iron pickaxe",
    14: "make wood sword",
    15: "make stone sword",
    16: "make iron sword",
    17: "noop2",
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--outdir", default="/home/railab/logdir/craftax_paper_images")
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--scale", type=int, default=8)
  parser.add_argument("--platform", default="cpu")
  return parser.parse_args()


def set_fault_env(enabled):
  os.environ["CRAFTAX_FAULT_SEED"] = "123"
  os.environ["CRAFTAX_FAULT_SAMPLER"] = "0"
  os.environ["CRAFTAX_FAULT"] = "1" if enabled else "0"
  os.environ["CRAFTAX_FAULT_PROFILE"] = "diagnostic"
  os.environ["CRAFTAX_FAULT_FAMILY"] = "action_exec"
  os.environ["CRAFTAX_FAULT_TYPE"] = "sticky_after_repeat_switch"
  os.environ["CRAFTAX_FAULT_SEVERITY"] = "1.0"
  os.environ["CRAFTAX_FAULT_MANIFEST_PROB"] = "1.0"
  os.environ["CRAFTAX_FAULT_COOLDOWN"] = "0"
  os.environ["CRAFTAX_USE_RND"] = "0"


def make_env(seed, platform):
  from embodied.envs.craftax import Craftax
  return Craftax(
      task="classic_pixels",
      seed=seed,
      length=1000,
      logs=True,
      platform=platform,
  )


def step(env, action, reset=False):
  return env.step({
      "reset": np.asarray(reset, np.bool_),
      "action": np.asarray(action, np.int32),
  })


def save_image(array, path, scale=8):
  image = Image.fromarray(np.asarray(array, np.uint8), "RGB")
  if scale != 1:
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
  image.save(path)
  return image


def font(size=18):
  for candidate in (
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  ):
    if Path(candidate).exists():
      return ImageFont.truetype(candidate, size)
  return ImageFont.load_default()


def annotate_fault(image, info, outpath):
  image = image.convert("RGBA")
  overlay = Image.new("RGBA", image.size, (210, 20, 20, 0))
  draw_overlay = ImageDraw.Draw(overlay)
  draw_overlay.rectangle([0, 0, image.width, image.height], fill=(210, 20, 20, 46))
  image = Image.alpha_composite(image, overlay)

  draw = ImageDraw.Draw(image)
  border = max(8, image.width // 64)
  for i in range(border):
    draw.rectangle([i, i, image.width - 1 - i, image.height - 1 - i],
                   outline=(255, 35, 35, 255))

  title = "FAULT MANIFESTED"
  subtitle = (
      f"{info['family']}: "
      f"{info['requested_action_name'].upper()} -> "
      f"{info['env_action_name'].upper()}")
  title_font = font(max(20, image.width // 22))
  subtitle_font = font(max(13, image.width // 38))

  pad = max(10, image.width // 40)
  title_box = draw.textbbox((0, 0), title, font=title_font)
  subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
  box_w = max(title_box[2] - title_box[0], subtitle_box[2] - subtitle_box[0]) + 2 * pad
  box_h = (title_box[3] - title_box[1]) + (subtitle_box[3] - subtitle_box[1]) + 3 * pad
  x0, y0 = pad, pad
  draw.rounded_rectangle(
      [x0, y0, x0 + box_w, y0 + box_h],
      radius=6,
      fill=(0, 0, 0, 150),
      outline=(255, 75, 75, 255),
      width=2,
  )
  draw.text((x0 + pad, y0 + pad), title, font=title_font, fill=(255, 245, 245, 255))
  draw.text(
      (x0 + pad, y0 + 2 * pad + title_box[3] - title_box[1]),
      subtitle,
      font=subtitle_font,
      fill=(255, 225, 225, 255),
  )
  image.convert("RGB").save(outpath)


def overlay_fault_no_text(image, outpath):
  image = image.convert("RGBA")
  overlay = Image.new("RGBA", image.size, (210, 20, 20, 0))
  draw_overlay = ImageDraw.Draw(overlay)
  draw_overlay.rectangle([0, 0, image.width, image.height], fill=(210, 20, 20, 46))
  image = Image.alpha_composite(image, overlay)
  draw = ImageDraw.Draw(image)
  border = max(8, image.width // 64)
  for i in range(border):
    draw.rectangle([i, i, image.width - 1 - i, image.height - 1 - i],
                   outline=(255, 35, 35, 255))
  image.convert("RGB").save(outpath)


def annotate_clean(image, outpath):
  image = image.convert("RGBA")
  draw = ImageDraw.Draw(image)
  label = "Clean Craftax State"
  label_font = font(max(18, image.width // 25))
  pad = max(10, image.width // 42)
  box = draw.textbbox((0, 0), label, font=label_font)
  draw.rounded_rectangle(
      [pad, pad, pad + box[2] - box[0] + 2 * pad, pad + box[3] - box[1] + 2 * pad],
      radius=6,
      fill=(0, 0, 0, 135),
  )
  draw.text((2 * pad, 2 * pad), label, font=label_font, fill=(245, 255, 245, 255))
  image.convert("RGB").save(outpath)


def make_side_by_side(clean, fault, outpath):
  gap = max(12, clean.width // 24)
  canvas = Image.new("RGB", (clean.width * 2 + gap, clean.height), (245, 245, 245))
  canvas.paste(clean.convert("RGB"), (0, 0))
  canvas.paste(fault.convert("RGB"), (clean.width + gap, 0))
  canvas.save(outpath)


def main():
  args = parse_args()
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  # Clean reference screenshot.
  set_fault_env(False)
  clean_env = make_env(args.seed, args.platform)
  clean_obs = step(clean_env, 0, reset=True)
  clean_raw = save_image(clean_obs["image"], outdir / "craftax_clean_raw.png", args.scale)
  annotate_clean(clean_raw, outdir / "craftax_clean_annotated.png")

  # Fault screenshot: three repeated LEFT actions followed by RIGHT triggers
  # sticky_after_repeat_switch. The requested RIGHT is replaced by previous LEFT.
  set_fault_env(True)
  fault_env = make_env(args.seed, args.platform)
  obs = step(fault_env, 0, reset=True)
  trigger_obs = None
  for action in (1, 1, 1, 2):
    obs = step(fault_env, action)
    if int(obs.get("log/fault_applied", 0)):
      trigger_obs = obs
      break
  if trigger_obs is None:
    raise RuntimeError("Fault did not manifest during deterministic trigger sequence.")

  fault_raw = save_image(trigger_obs["image"], outdir / "craftax_fault_raw.png", args.scale)
  requested = int(trigger_obs.get("log/requested_action", -1))
  executed = int(trigger_obs.get("log/env_action", -1))
  info = {
      "family": "action-exec",
      "type": "sticky-after-repeat-switch",
      "requested_action_name": ACTION_NAMES.get(requested, str(requested)),
      "env_action_name": ACTION_NAMES.get(executed, str(executed)),
  }
  annotate_fault(fault_raw, info, outdir / "craftax_fault_red_overlay.png")
  overlay_fault_no_text(fault_raw, outdir / "craftax_fault_red_overlay_notext.png")

  clean_annotated = Image.open(outdir / "craftax_clean_annotated.png")
  fault_annotated = Image.open(outdir / "craftax_fault_red_overlay.png")
  make_side_by_side(
      clean_annotated, fault_annotated,
      outdir / "craftax_clean_vs_fault_red_overlay.png")
  fault_notext = Image.open(outdir / "craftax_fault_red_overlay_notext.png")
  make_side_by_side(
      clean_raw, fault_notext,
      outdir / "craftax_clean_vs_fault_red_overlay_notext.png")

  print("Wrote images to:", outdir)
  for path in sorted(outdir.glob("*.png")):
    print(path)
  print(
      "fault_applied=1",
      "requested=", requested,
      ACTION_NAMES.get(requested, requested),
      "executed=", executed,
      ACTION_NAMES.get(executed, executed),
  )


if __name__ == "__main__":
  main()

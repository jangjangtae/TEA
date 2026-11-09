#!/usr/bin/env python3
"""Patch installed Crafter with semantic fault manifestation probability.

The project currently uses the pip-installed ``crafter`` package. This script
keeps the local environment patch reproducible when the virtualenv is rebuilt.
Run it inside the same Python environment used for training.
"""

import importlib.util
from pathlib import Path


def replace_once(text, old, new, label):
  if new in text:
    return text, False
  if old not in text:
    raise RuntimeError(f'Could not find patch anchor: {label}')
  return text.replace(old, new, 1), True


def patch(path):
  path = Path(path)
  text = path.read_text(encoding='utf-8')
  changed = False

  replacements = [
      (
          "    self._semantic_fault_ep_prob = float(os.getenv('CRAFTER_SEMANTIC_FAULT_EP_PROB', '0.5'))\n"
          "    self._semantic_retry_gap = int(os.getenv('CRAFTER_SEMANTIC_RETRY_GAP', '40'))\n",
          "    self._semantic_fault_ep_prob = float(os.getenv('CRAFTER_SEMANTIC_FAULT_EP_PROB', '0.5'))\n"
          "    self._semantic_fault_manifest_prob = float(os.getenv('CRAFTER_SEMANTIC_FAULT_MANIFEST_PROB', '1.0'))\n"
          "    self._semantic_retry_gap = int(os.getenv('CRAFTER_SEMANTIC_RETRY_GAP', '40'))\n",
          'semantic manifest env var',
      ),
      (
          "        'severity': 1.0,\n"
          "        'trigger': self._semantic_trigger_name(subtype),\n",
          "        'severity': float(self._semantic_fault_manifest_prob),\n"
          "        'trigger': self._semantic_trigger_name(subtype),\n",
          'semantic spec severity',
      ),
      (
          "    result['trigger_count'] = int(self._semantic_trigger_count)\n"
          "    result['first_trigger_step'] = int(self._semantic_first_trigger_step)\n\n"
          "  def _restore_inventory(self, inventory):\n",
          "    result['trigger_count'] = int(self._semantic_trigger_count)\n"
          "    result['first_trigger_step'] = int(self._semantic_first_trigger_step)\n\n"
          "  def _semantic_should_manifest(self):\n"
          "    prob = float(np.clip(self._semantic_fault_manifest_prob, 0.0, 1.0))\n"
          "    return bool(self._world.random.uniform() < prob)\n\n"
          "  def _restore_inventory(self, inventory):\n",
          'semantic manifest helper',
      ),
      (
          "            result['ctx_upgrade_collect'] = 1\n"
          "            self._mark_semantic_trigger(result, 'upgrade_collect')\n"
          "            give_items = {}\n",
          "            result['ctx_upgrade_collect'] = 1\n"
          "            self._mark_semantic_trigger(result, 'upgrade_collect')\n"
          "            if not self._semantic_should_manifest():\n"
          "              return result\n"
          "            give_items = {}\n",
          'upgrade collect gate',
      ),
      (
          "            result['ctx_retry_craft'] = 1\n"
          "            self._mark_semantic_trigger(result, 'retry_craft')\n"
          "            self._player.inventory[item] = prev_state['inventory'][item]\n",
          "            result['ctx_retry_craft'] = 1\n"
          "            self._mark_semantic_trigger(result, 'retry_craft')\n"
          "            if not self._semantic_should_manifest():\n"
          "              return result\n"
          "            self._player.inventory[item] = prev_state['inventory'][item]\n",
          'craft missing gate',
      ),
      (
          "            result['ctx_retry_craft'] = 1\n"
          "            self._mark_semantic_trigger(result, 'retry_craft')\n"
          "            self._restore_inventory(prev_state['inventory'])\n",
          "            result['ctx_retry_craft'] = 1\n"
          "            self._mark_semantic_trigger(result, 'retry_craft')\n"
          "            if not self._semantic_should_manifest():\n"
          "              return result\n"
          "            self._restore_inventory(prev_state['inventory'])\n",
          'recipe retry gate',
      ),
      (
          "            result['ctx_relocate_station'] = 1\n"
          "            self._mark_semantic_trigger(result, 'relocate_station')\n"
          "            if subtype == 'station_place_ghost_on_relocate':\n",
          "            result['ctx_relocate_station'] = 1\n"
          "            self._mark_semantic_trigger(result, 'relocate_station')\n"
          "            if not self._semantic_should_manifest():\n"
          "              self._successful_place_counts[place_name] += 1\n"
          "              return result\n"
          "            if subtype == 'station_place_ghost_on_relocate':\n",
          'station relocate gate',
      ),
      (
          "        self._player.achievements[matched] = prev_state['achievements'][matched]\n"
          "        if subtype in ('achievement_unlock_missing_after_reconfirm', 'progress_confirmation_requires_revisit'):\n",
          "        result['ctx_valid_progress'] = 1\n"
          "        self._mark_semantic_trigger(result, 'valid_progress')\n"
          "        if not self._semantic_should_manifest():\n"
          "          return result\n"
          "        self._player.achievements[matched] = prev_state['achievements'][matched]\n"
          "        if subtype in ('achievement_unlock_missing_after_reconfirm', 'progress_confirmation_requires_revisit'):\n",
          'valid progress gate',
      ),
      (
          "              result['ctx_relocate_station'] = 1\n"
          "              self._mark_semantic_trigger(result, 'station_reuse')\n"
          "              self._broken_station_positions[tuple(pre_target)] = place_name\n",
          "              result['ctx_relocate_station'] = 1\n"
          "              self._mark_semantic_trigger(result, 'station_reuse')\n"
          "              if not self._semantic_should_manifest():\n"
          "                self._successful_place_counts[place_name] += 1\n"
          "                return result\n"
          "              self._broken_station_positions[tuple(pre_target)] = place_name\n",
          'station broken arm gate',
      ),
      (
          "              result['ctx_station_reuse'] = 1\n"
          "              self._mark_semantic_trigger(result, 'station_reuse')\n"
          "              self._restore_inventory(prev_state['inventory'])\n",
          "              result['ctx_station_reuse'] = 1\n"
          "              self._mark_semantic_trigger(result, 'station_reuse')\n"
          "              if not self._semantic_should_manifest():\n"
          "                return result\n"
          "              self._restore_inventory(prev_state['inventory'])\n",
          'station use gate',
      ),
      (
          "            result['ctx_delayed_after_use'] = 1\n"
          "            self._mark_semantic_trigger(result, 'delayed_after_use')\n"
          "            self._schedule_delayed_inventory_bug(item)\n",
          "            result['ctx_delayed_after_use'] = 1\n"
          "            self._mark_semantic_trigger(result, 'delayed_after_use')\n"
          "            if not self._semantic_should_manifest():\n"
          "              return result\n"
          "            self._schedule_delayed_inventory_bug(item)\n",
          'delayed inventory gate',
      ),
      (
          "        'semantic_fault_profile': self._semantic_fault_profile,\n"
          "        'semantic_trigger_context': int(semantic_fault['trigger_context']),\n",
          "        'semantic_fault_profile': self._semantic_fault_profile,\n"
          "        'semantic_fault_manifest_prob': float(self._semantic_fault_manifest_prob),\n"
          "        'semantic_trigger_context': int(semantic_fault['trigger_context']),\n",
          'semantic manifest info',
      ),
  ]

  for old, new, label in replacements:
    text, did_change = replace_once(text, old, new, label)
    changed = changed or did_change

  path.write_text(text, encoding='utf-8')
  return changed


def main():
  spec = importlib.util.find_spec('crafter.env')
  if spec is None or spec.origin is None:
    raise SystemExit('Could not locate installed crafter.env module.')
  path = Path(spec.origin)
  changed = patch(path)
  print(('patched' if changed else 'already patched') + f': {path}')


if __name__ == '__main__':
  main()

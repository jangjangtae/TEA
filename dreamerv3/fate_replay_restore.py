"""Opt-in, copied clean replay initialization for the continuation diagnostic."""

import argparse
import json
import os
import shutil
from pathlib import Path

import elements
import numpy as np

from embodied.core.replay import Replay
from dreamerv3.audit_fate_revision_run import require
from dreamerv3.fate_diagnostic_followup import digest, write_json
from dreamerv3.fate_replication_queue import verify_files


REPLAY_ONLY_PROTOCOL = 'tea_replay_only_v1'
MATCHED_TRAIN_START = 2048


class ReplayScheduleAudit:
  def __init__(self, directory, first_callback):
    self.path = Path(str(directory)) / 'replay_schedule_audit.json'
    self.first_callback = first_callback
    self.data = dict(complete=False, target_first_callback=first_callback,
                     first_update_callback=None, last_update_callback=None, updates=0)

  def ready(self, step):
    return step >= self.first_callback

  def record(self, step):
    require(self.ready(step), 'Restored replay trained before matched warmup')
    first = self.data['updates'] == 0
    if first:
      require(step == self.first_callback, 'Missed matched first update')
      self.data['first_update_callback'] = step
    self.data['updates'] += 1
    self.data['last_update_callback'] = step
    if first:
      write_json(self.path, self.data)

  def finish(self, step):
    expected = 1 + (step - self.first_callback) // 8
    require(self.data['updates'] == expected and expected > 0,
            'Optimizer schedule does not match fresh-replay control')
    self.data.update(complete=True, callbacks=step, expected_updates=expected)
    write_json(self.path, self.data)


def validate_protocol(spec, args):
  if spec.get('protocol') == REPLAY_ONLY_PROTOCOL:
    require(args.train_ratio == 128 and args.batch_size == 16 and args.batch_length == 64
            and args.envs == 16 and spec['capacity'] == 100000 and spec['length'] == 65
            and spec['matched_train_start'] == MATCHED_TRAIN_START,
            'Replay-only comparison changed training schedule or replay shape')
    f = args.fault
    objective = ((f.log_only and f.reward_mode == 'dense' and f.beta == 0) or
                 (not f.log_only and f.reward_mode == 'excess_delta_threshold' and f.beta == .2))
    require(f.task_reward_source == 'delivered' and f.ref_ckpt == spec['checkpoint']
            and objective and f.reward_delta_threshold == .5 and f.clip == 2,
            'Replay-only comparison changed reward/reference contract')
    require(spec['historical_rewards'] == 'original_clean_task_reward_no_relabel',
            'Historical rewards must not be relabeled')
    return MATCHED_TRAIN_START
  require('protocol' not in spec, 'Unknown replay restoration protocol')
  require(args.train_ratio == 512 and args.fault.log_only and
          args.fault.task_reward_source == 'delivered', 'Not the joint TaskOnly diagnostic')
  return 0


def select_chunks(directory, length, capacity):
  replay = Replay(length=length, capacity=capacity)
  names = sorted(Path(directory).glob('*.npz'), reverse=True)
  sizes = replay._numitems([p.name for p in names])
  selected, count = [], 0
  for path in names:
    selected.append(path)
    count += int(sizes[elements.UUID(path.stem.split('-')[1])])
    if count >= capacity:
      break
  require(count >= capacity, 'Not enough saved clean replay sequences')
  return selected


def copy_and_load(replay, manifest, destination):
  spec = json.loads(Path(manifest).read_text())
  require(replay.length == spec['length'] and replay.capacity == spec['capacity'],
          'Replay shape/capacity does not match the frozen snapshot')
  require(not len(replay) and not replay.current, 'Replay must start empty')
  destination = Path(destination).resolve()
  require(Path(str(replay.directory)).resolve() == destination, 'Wrong replay destination')
  require(not list(destination.iterdir()), 'Refusing to mix with existing replay files')
  verify_files(spec['files'])
  require(digest(Path(spec['checkpoint']) / 'agent.pkl') == spec['agent_sha256'],
          'Source agent changed')
  for name, sha in spec['files'].items():
    src = Path(name)
    target = destination / src.name
    require(src.resolve() != target, 'Source replay must remain read-only')
    shutil.copyfile(src, target)
    require(digest(target) == sha, 'Replay copy differs from source')
  replay.load(amount=spec['capacity'])
  require(len(replay) == spec['capacity'], 'Incomplete replay restore')
  require(len(replay.chunks) == len(spec['files']), 'Replay silently skipped chunks')
  expected_keys = {'image', 'reward', 'action', 'is_first', 'is_last',
                   'is_terminal', 'dyn/deter', 'dyn/stoch', 'stepid'}
  total = 0
  for chunk in replay.chunks.values():
    require(set(chunk.data) == expected_keys, 'Unexpected stored transition fields')
    for key, values in chunk.data.items():
      require(len(values) == chunk.length and np.isfinite(values).all(),
              f'Invalid replay data: {chunk.filename} {key}')
    require(chunk.data['image'].dtype == np.uint8 and
            chunk.data['image'].shape[1:] == (64, 64, 3), 'Wrong replay observations')
    require(np.all((chunk.data['action'] >= 0) & (chunk.data['action'] < 17)),
            'Wrong replay actions')
    total += chunk.length
  # Inspect raw sequences without advancing the replay sampler's RNG.
  ids = list(replay.items)
  for itemid in (ids[0], ids[len(ids) // 2], ids[-1]):
    chunkid, index = replay.items[itemid]
    seq = replay._getseq(chunkid, index)
    require(all(len(x) == replay.length for x in seq.values()), 'Broken sequence chain')
  require(not replay.current and not replay.streams,
          'Old trajectories must not be appended to new environment streams')
  return dict(passed=True, manifest=str(Path(manifest).resolve()),
      manifest_sha256=digest(manifest), checkpoint=spec['checkpoint'],
      agent_sha256=spec['agent_sha256'], capacity=replay.capacity,
      length=replay.length, loaded_sequences=len(replay), loaded_chunks=len(replay.chunks),
      stored_transitions=total, data_bytes=sum(c.nbytes for c in replay.chunks.values()),
      source_written=False, restoration='latest capacity-sized saved replay, not exact live FIFO/RNG state')


def restore_initial_replay(replay, args):
  manifest = Path(os.environ['FATE_INITIAL_REPLAY_MANIFEST'])
  spec = json.loads(manifest.read_text())
  directory = Path(str(args.logdir))
  require(not (directory / 'ckpt/latest').exists(),
          'Initial replay import is not resume; use an isolated queue attempt')
  require(str(args.from_checkpoint) == spec['checkpoint'], 'Replay belongs to another prior')
  matched_start = validate_protocol(spec, args)
  report = copy_and_load(replay, manifest, directory / 'replay')
  report.update(matched_train_start=matched_start,
                historical_rewards='original_clean_task_reward_no_relabel')
  write_json(directory / 'replay_restore_audit.json', report)
  print(f"Restored clean replay: {report['loaded_sequences']} sequences, "
        f"{report['loaded_chunks']} chunks", flush=True)
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--out', type=Path, required=True)
  args = parser.parse_args()
  require(not args.out.exists(), 'Use a fresh validation directory')
  spec = json.loads(args.manifest.read_text())
  replay = Replay(length=spec['length'], capacity=spec['capacity'],
                  directory=str(args.out / 'replay'))
  report = copy_and_load(replay, args.manifest, args.out / 'replay')
  batch = replay.sample(16)
  require(batch['image'].shape == (16, spec['length'], 64, 64, 3), 'Wrong sampled batch')
  require(all(np.isfinite(value).all() for value in batch.values()), 'Nonfinite sampled data')
  report['sample_shape'] = list(batch['image'].shape)
  write_json(args.out / 'PASS.json', report)
  replay.workers.shutdown(wait=True)
  print(json.dumps(report), flush=True)


if __name__ == '__main__':
  main()

"""Opt-in identity checks for independently loaded active and frozen agents."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def parameter_digest(params):
  result = hashlib.sha256()
  for key in sorted(params):
    value = np.ascontiguousarray(params[key])
    if not np.isfinite(value).all():
      raise AssertionError(f'Nonfinite parameters: {key}')
    result.update(json.dumps([key, value.dtype.str, value.shape]).encode())
    result.update(value.tobytes())
  return result.hexdigest()


def checkpoint_data(path):
  with (Path(str(path)) / 'agent.pkl').open('rb') as handle:
    return pickle.load(handle)


def verify_loaded(agent, path):
  saved = checkpoint_data(path)
  actual = agent.save()
  expected = parameter_digest(saved['params'])
  assert parameter_digest(actual['params']) == expected, f'Wrong loaded agent: {path}'
  counters = dict(saved['counters'], batches=saved['counters']['updates'])
  assert actual['counters'] == counters, 'Checkpoint counters not restored as specified'
  return dict(checkpoint=str(path), parameters_sha256=expected,
              tensor_count=len(saved['params']), counters=actual['counters'])


class ReferenceAudit:
  def __init__(self, directory, active, reference, active_path, reference_path):
    assert reference is not None, 'Reference audit requires a separate reference agent'
    assert active is not reference
    self.path = Path(str(directory)) / 'reference_identity_audit.json'
    self.data = dict(active_initial=verify_loaded(active, active_path),
                     reference_initial=verify_loaded(reference, reference_path), complete=False)
    self.write()

  def write(self):
    self.path.write_text(json.dumps(self.data, indent=2, allow_nan=False) + '\n')

  def finish(self, active, reference):
    current = reference.save()
    fingerprint = parameter_digest(current['params'])
    assert fingerprint == self.data['reference_initial']['parameters_sha256'], 'Reference parameters changed'
    assert current['counters']['updates'] == self.data['reference_initial']['counters']['updates'], 'Reference trained'
    active_data = active.save()
    assert active_data['counters']['updates'] > self.data['active_initial']['counters']['updates'], 'No active updates'
    self.data.update(complete=True, reference_final_sha256=fingerprint,
        reference_final_counters=current['counters'], active_final_counters=active_data['counters'],
        active_final_sha256=parameter_digest(active_data['params']))
    self.write()

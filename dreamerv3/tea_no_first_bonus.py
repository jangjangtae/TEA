"""Opt-in TEA boundary ablation; preserve surprise and subsequent differencing."""

import math


class FirstTransitionBonusMask:
  def __init__(self, config):
    get = config.get
    if get('log_only', True) or get('reward_mode') != 'excess_delta_threshold':
      raise ValueError('Boundary masking requires paid temporal-excess rewards')
    if get('constraint_mode', 'none') != 'none' or get('beta_schedule', 'fixed') != 'fixed':
      raise ValueError('Boundary ablation requires fixed unconstrained TEA')
    for key in ('semantic_coverage_beta', 'unique_suspicious_beta', 'repeat_suspicious_penalty'):
      if get(key, 0):
        raise ValueError('Boundary ablation does not combine other shaping bonuses')
    self.steps = {}
    self.ended = {}
    self.records = {}

  def apply(self, tran, worker, result):
    reset = bool(tran['is_first'])
    if reset:
      if worker in self.ended and not self.ended[worker]:
        raise ValueError('Reset without episode termination')
      self.steps[worker] = 0
      self.ended[worker] = False
    else:
      if worker not in self.steps or self.ended[worker]:
        raise ValueError('Missing reset before real transition')
      self.steps[worker] += 1
      self.ended[worker] = bool(tran['is_last'])
    boundary = not reset and self.steps[worker] == 1
    candidate = float(result['fault_reward_bonus'])
    if not math.isfinite(candidate) or candidate < 0:
      raise ValueError('Invalid candidate TEA bonus')
    if boundary:
      # Do not zero the surprise: the next transition still differences against it.
      result['fault_reward_bonus'] = 0.0
      result['fault_reward_score'] = 0.0
      result['augmented_reward'] = float(result['learning_base_reward'])
    self.records[worker] = {
        'tea_no_first_bonus': True,
        'tea_first_non_reset': boundary,
        'tea_unmasked_bonus': candidate,
        'tea_removed_bonus': candidate if boundary else 0.0,
    }

  def trace_fields(self, worker):
    return dict(self.records[worker])

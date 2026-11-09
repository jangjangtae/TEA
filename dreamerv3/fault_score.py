import json
from collections import defaultdict
from pathlib import Path

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import numpy as np


BUG_TYPE_NAMES = {
    0: 'none',
    1: 'drop_to_fallback',
    2: 'remap_after_success_switch',
    3: 'delay_after_success',
    4: 'sticky_after_repeat_switch',
    5: 'ignore_nonzero_after_reward',
    6: 'ignore_switch_late_episode',
    7: 'reward_delay_on_positive',
    8: 'reward_scale_half_on_positive_switch',
    9: 'reward_zero_on_positive',
    10: 'early_done_after_success_switch',
    11: 'remap_after_repeat_switch',
    12: 'delay_after_late_episode_switch',
    13: 'ignore_nonzero_after_two_rewards',
    14: 'reward_zero_after_repeat_switch',
    15: 'reward_delay_after_two_rewards',
    16: 'early_done_after_repeat_switch',
    17: 'revisit_action_ignore',
    18: 'revisit_action_delay',
    19: 'delayed_switch_failure',
    20: 'tool_collect_desync_on_upgrade',
    21: 'collect_result_delayed_after_tool_upgrade',
    22: 'upgrade_branch_inconsistent_collect_behavior',
    23: 'craft_result_missing_on_retry',
    24: 'craft_output_delayed_on_retry',
    25: 'recipe_precondition_mischeck_on_retry',
    26: 'recipe_retry_requires_revisit',
    27: 'station_place_ghost_on_relocate',
    28: 'station_second_use_inconsistent_after_placement',
    29: 'station_usable_flag_broken_after_relocate',
    30: 'station_state_partial_reset_after_relocate',
    31: 'achievement_unlock_missing_after_valid_progress',
    32: 'achievement_unlock_missing_after_reconfirm',
    33: 'progress_confirmation_requires_revisit',
    34: 'delayed_inventory_desync_after_station_use',
}

MINIGRID_BUG_TYPE_NAMES = {
    0: 'none',
    1: 'broken_door',
    2: 'heavy_key',
    3: 'action_flip',
    4: 'teleport',
    5: 'door_gone',
    6: 'lava_gap',
}

REWARD_MODE_IDS = {
    'dense': 1,
    'threshold': 2,
    'binary_threshold': 2,
    'excess_threshold': 3,
    'excess': 3,
    'delta_threshold': 4,
    'binary_delta_threshold': 4,
    'excess_delta_threshold': 5,
    'delta_excess': 5,
}

CONTEXT_SCHEMA = {
    'version': 1,
    'fields': [
        'action', 'inventory_bucket', 'nearby_tile',
        'achievement_stage', 'nearby_mob'],
    'fallback_order': ['full', 'action_stage', 'action', 'global'],
}

CONTEXT_LEVEL_IDS = {
    'global': 0,
    'action': 1,
    'action_stage': 2,
    'full': 3,
}


def _cfg_get(config, key, default=None):
  if config is None:
    return default
  if hasattr(config, 'get'):
    return config.get(key, default)
  return getattr(config, key, default)


ONLINE_SCORE_SOURCES = (
    'online_latent_kl',
    'online_kl',
    'active_latent_kl',
    'active_kl',
    'lbs',
    'latent_bayesian_surprise',
)

TASK_REWARD_SOURCE_IDS = {
    'delivered': 0,
    'post_fault': 0,
    'environment': 0,
    'pre_fault': 1,
    'raw': 1,
    'legacy': 1,
}


def uses_reference_model(config=None):
  """Whether this fault-score configuration needs a frozen reference agent."""
  score_source = str(_cfg_get(
      config, 'score_source', 'latent_reward') or 'latent_reward').lower()
  return score_source not in ONLINE_SCORE_SOURCES


def transition_task_rewards(tran, config=None):
  """Return the selected, delivered, and pre-fault task rewards.

  The delivered reward is the reward emitted after the environment fault has
  been applied but before any FATE bonus is added. The pre-fault reward is kept
  for diagnostics and exact legacy reproduction only.
  """
  delivered = _scalar(tran.get(
      'log/env_reward', tran.get('reward', 0.0)))
  pre_fault = _scalar(tran.get('log/task_reward_raw', delivered))
  source = str(_cfg_get(
      config, 'task_reward_source', 'delivered') or 'delivered').lower()
  if source not in TASK_REWARD_SOURCE_IDS:
    raise ValueError(
        f'Unknown fault task reward source: {source}. Expected delivered or '
        'pre_fault.')
  selected = pre_fault if TASK_REWARD_SOURCE_IDS[source] == 1 else delivered
  return selected, delivered, pre_fault, TASK_REWARD_SOURCE_IDS[source]


def reduce_metric(x):
  x = nn.cast(x)
  if hasattr(x, 'ndim') and x.ndim > 1:
    return x.mean(tuple(range(1, x.ndim)))
  return x


def agent_fault_metrics(
    dyn, rew, feat2tensor, feat, obs, config=None, tokens=None):
  """Fault signals from a clean reference world model at policy time."""
  use_latent_kl = _cfg_get(config, 'use_latent_kl', True)
  use_reward_error = _cfg_get(config, 'use_reward_error', True)
  w_kl = float(_cfg_get(config, 'w_kl', 1.0))
  w_reward = float(_cfg_get(config, 'w_reward', 1.0))

  reward_true = nn.cast(obs['reward'])
  zeros = jnp.zeros_like(reduce_metric(reward_true))

  latent_kl = zeros
  if use_latent_kl and ('prior_logit' in feat) and ('logit' in feat):
    posterior = dyn._dist(feat['logit'])
    prior = dyn._dist(jax.lax.stop_gradient(feat['prior_logit']))
    latent_kl = reduce_metric(posterior.kl(prior))

  reward_pred = zeros
  reward_error = zeros
  if use_reward_error and ('prior_logit' in feat):
    prior_feat = {
        'deter': feat['deter'],
        'stoch': dyn._dist(feat['prior_logit']).pred(),
    }
    reward_pred = reduce_metric(rew(feat2tensor(prior_feat), 1).pred())
    reward_error = reduce_metric(jnp.abs(reward_pred - reward_true))

  fault_score_raw = w_kl * latent_kl + w_reward * reward_error

  novelty_lhs = zeros
  novelty_bound = zeros
  novelty_violation = zeros
  novelty_score = zeros
  novelty_triggered = zeros
  if bool(_cfg_get(config, 'use_kl_bound', False)) and tokens is not None:
    post = dyn._dist(feat['logit'])
    prior_ht = dyn._dist(jax.lax.stop_gradient(feat['prior_logit']))
    h0 = jnp.zeros_like(feat['deter'])
    prior_h0 = dyn._dist(jax.lax.stop_gradient(dyn._prior(h0)))
    post_h0_x = dyn._dist(jax.lax.stop_gradient(dyn._posterior(h0, tokens)))
    novelty_lhs = reduce_metric(post.kl(prior_ht))
    bound_prior = reduce_metric(post.kl(prior_h0))
    bound_observation = reduce_metric(post.kl(post_h0_x))
    novelty_bound = bound_prior - bound_observation
    novelty_violation = novelty_lhs - novelty_bound
    novelty_score = jnp.maximum(novelty_violation, 0.0)
    novelty_triggered = nn.cast(novelty_violation > 0.0)

  return {
      'fault_score_raw': fault_score_raw,
      'fault_score': fault_score_raw,
      'latent_kl_surprise': latent_kl,
      'reward_prediction_error': reward_error,
      'reward_pred': reward_pred,
      'novelty_kl_lhs': novelty_lhs,
      'novelty_kl_bound': novelty_bound,
      'novelty_bound_violation': novelty_violation,
      'novelty_bound_score': novelty_score,
      'novelty_bound_triggered': novelty_triggered,
  }


def load_norm_stats(path):
  if not path:
    return {}
  path = Path(str(path)).expanduser()
  if not path.exists():
    raise FileNotFoundError(path)
  with path.open('r', encoding='utf-8') as f:
    return json.load(f)


def normalize_score(raw, stats=None, mode='p95', eps=1e-8):
  raw = float(raw)
  stats = stats or {}
  mode = (mode or 'none').lower()
  if mode == 'none':
    return raw
  if mode == 'zscore':
    mean = float(stats.get('fault_score_mean', 0.0))
    std = float(stats.get('fault_score_std', 1.0))
    return (raw - mean) / max(std, eps)
  if mode in ('p90', 'p95', 'p99'):
    denom = float(stats.get(f'fault_score_{mode}', 0.0))
    if denom <= eps:
      denom = float(stats.get('fault_score_p95', 1.0))
    return raw / max(denom, eps)
  raise ValueError(f'Unknown fault score norm mode: {mode}')


def context_components(tran):
  return {
      'action': int(_scalar(tran.get(
          'log/requested_action', tran.get('action', 0)))),
      'inventory_bucket': int(_scalar(tran.get(
          'log/context_inventory_bucket', 0))),
      'nearby_tile': int(_scalar(tran.get('log/context_nearby_tile', 0))),
      'achievement_stage': int(_scalar(tran.get(
          'log/context_achievement_stage', 0))),
      'nearby_mob': int(_scalar(tran.get('log/context_nearby_mob', 0))),
  }


def context_keys(tran):
  ctx = context_components(tran)
  return {
      'full': (
          f"full:a={ctx['action']}|i={ctx['inventory_bucket']}|"
          f"t={ctx['nearby_tile']}|g={ctx['achievement_stage']}|"
          f"m={ctx['nearby_mob']}"),
      'action_stage': (
          f"action_stage:a={ctx['action']}|g={ctx['achievement_stage']}"),
      'action': f"action:a={ctx['action']}",
      'global': 'global',
  }


def normalize_transition_score(raw, tran, stats=None, config=None, eps=1e-8):
  """Normalize with the most specific sufficiently sampled clean context."""
  stats = stats or {}
  mode = str(_cfg_get(config, 'norm_mode', 'p95') or 'none').lower()
  if mode not in ('context_p90', 'context_p95', 'context_p99',
                  'context_zscore', 'contextual_p90', 'contextual_p95',
                  'contextual_p99', 'contextual_zscore'):
    normalized = normalize_score(raw, stats, mode, eps)
    baseline = float(stats.get('fault_score_p95', 0.0))
    std = float(stats.get('fault_score_std', 1.0))
    zscore = (float(raw) - float(stats.get('fault_score_mean', 0.0))) / max(std, eps)
    return normalized, 'global', 'global', baseline, zscore

  context_stats = stats.get('context_stats', {})
  min_count = int(_cfg_get(config, 'context_min_count', 50))
  selected_key = 'global'
  selected_level = 'global'
  selected = stats
  for level, key in context_keys(tran).items():
    candidate = context_stats.get(key)
    if candidate and int(candidate.get('fault_score_count', 0)) >= min_count:
      selected_key = key
      selected_level = level
      selected = candidate
      break

  suffix = mode.rsplit('_', 1)[-1]
  mean = float(selected.get('fault_score_mean', stats.get('fault_score_mean', 0.0)))
  std = float(selected.get('fault_score_std', stats.get('fault_score_std', 1.0)))
  p95 = float(selected.get('fault_score_p95', stats.get('fault_score_p95', 0.0)))
  zscore = (float(raw) - mean) / max(std, eps)
  if suffix == 'zscore':
    normalized = zscore
  else:
    quantile = float(selected.get(
        f'fault_score_{suffix}', stats.get(f'fault_score_{suffix}', 0.0)))
    if quantile <= eps:
      quantile = float(stats.get(f'fault_score_{suffix}', 1.0))
    normalized = float(raw) / max(quantile, eps)
  return normalized, selected_key, selected_level, p95, zscore


def reward_gate_active(tran, config=None):
  """Whether a fault reward bonus is allowed for this transition."""
  mode = str(_cfg_get(config, 'reward_gate', 'none') or 'none').lower()
  if mode in ('none', 'all', 'always', ''):
    return True

  semantic_keys = (
      'log/semantic_trigger_context',
      'log/semantic_ctx_upgrade_collect',
      'log/semantic_ctx_retry_craft',
      'log/semantic_ctx_relocate_station',
      'log/semantic_ctx_valid_progress',
      'log/semantic_ctx_station_reuse',
      'log/semantic_ctx_delayed_after_use',
  )
  semantic_active = any(_scalar(tran.get(key, 0.0)) > 0.5
      for key in semantic_keys)
  task_reward, _, _, _ = transition_task_rewards(tran, config)
  reward_active = abs(task_reward) > 1e-6
  action_active = int(_scalar(tran.get('action', 0.0))) != 0

  if mode in ('semantic', 'semantic_context', 'semantic_event'):
    return semantic_active
  if mode in ('reward', 'task_reward', 'nonzero_reward'):
    return reward_active
  if mode in ('nonzero_action', 'action'):
    return action_active
  if mode in ('semantic_or_reward', 'event'):
    return semantic_active or reward_active
  if mode in ('semantic_or_action', 'interaction'):
    return semantic_active or action_active
  if mode in ('semantic_reward_action', 'semantic_or_reward_or_action'):
    return semantic_active or reward_active or action_active

  raise ValueError(f'Unknown fault reward gate: {mode}')


def compute_fault_bonus(
    normalized, clipped, beta, config=None, score_delta=0.0, eps=1e-8):
  """Convert a normalized clean-prior score into a policy reward bonus."""
  mode = str(_cfg_get(config, 'reward_mode', 'dense') or 'dense').lower()
  threshold = float(_cfg_get(config, 'reward_threshold', 1.0))
  delta_threshold = float(_cfg_get(config, 'reward_delta_threshold', threshold))
  clip = float(_cfg_get(config, 'clip', 1.0))

  if mode == 'dense':
    shaped = clipped
  elif mode in ('threshold', 'binary_threshold'):
    shaped = 1.0 if float(normalized) >= threshold else 0.0
  elif mode in ('excess_threshold', 'excess'):
    denom = max(float(clip) - threshold, eps)
    shaped = np.clip((float(normalized) - threshold) / denom, 0.0, 1.0)
  elif mode in ('delta_threshold', 'binary_delta_threshold'):
    shaped = 1.0 if float(score_delta) >= delta_threshold else 0.0
    threshold = delta_threshold
  elif mode in ('excess_delta_threshold', 'delta_excess'):
    denom = max(float(clip) - delta_threshold, eps)
    shaped = np.clip((float(score_delta) - delta_threshold) / denom, 0.0, 1.0)
    threshold = delta_threshold
  else:
    raise ValueError(f'Unknown fault reward mode: {mode}')

  return float(beta * shaped), float(shaped), threshold, REWARD_MODE_IDS.get(mode, 0)


def compute_transition_fault(
    tran, config=None, stats=None, force_log_only=False, beta_override=None):
  """Compute normalized fault score and optional reward relabeling for one step."""
  is_first = bool(np.asarray(tran.get('is_first', False)).reshape(()))
  raw_reward, delivered_reward, pre_fault_reward, reward_source_id = (
      transition_task_rewards(tran, config))
  incoming_reward = _scalar(tran.get('reward', delivered_reward))
  # The wrapper may already have added RND. Change only the selected task
  # channel, never discard a wrapper bonus or relabel a logging-only run.
  learning_base_reward = raw_reward + incoming_reward - delivered_reward
  augmented_reward = incoming_reward

  if is_first:
    latent_kl = 0.0
    reward_pred = 0.0
    reward_error = 0.0
  else:
    latent_kl = _scalar(tran.get('log/ref_latent_kl_surprise',
        tran.get('log/ref_bug_kl', 0.0)))
    reward_pred = _scalar(tran.get('log/ref_reward_pred', 0.0))
    reward_error = abs(raw_reward - reward_pred)

  use_latent_kl = bool(_cfg_get(config, 'use_latent_kl', True))
  use_reward_error = bool(_cfg_get(config, 'use_reward_error', True))
  w_kl = float(_cfg_get(config, 'w_kl', 1.0))
  w_reward = float(_cfg_get(config, 'w_reward', 1.0))

  raw = 0.0
  novelty_lhs = _scalar(tran.get('log/ref_novelty_kl_lhs', 0.0))
  novelty_bound = _scalar(tran.get('log/ref_novelty_kl_bound', 0.0))
  novelty_violation = _scalar(tran.get(
      'log/ref_novelty_bound_violation', 0.0))
  novelty_bound_score = _scalar(tran.get(
      'log/ref_novelty_bound_score', max(novelty_violation, 0.0)))
  novelty_bound_triggered = _scalar(tran.get(
      'log/ref_novelty_bound_triggered', novelty_violation > 0.0))
  score_source = str(_cfg_get(
      config, 'score_source', 'latent_reward') or 'latent_reward').lower()
  if score_source in ('kl_bound', 'novelty_bound', 'paper_kl_bound'):
    raw = novelty_bound_score
  elif score_source in ('kl_bound_binary', 'novelty_bound_binary'):
    raw = float(novelty_bound_triggered > 0.5)
  elif score_source in ONLINE_SCORE_SOURCES:
    latent_kl = _scalar(tran.get('fault/latent_kl_surprise',
        tran.get('log/online_latent_kl_surprise',
        tran.get('log/latent_kl_surprise', 0.0))))
    reward_pred = _scalar(tran.get('fault/reward_pred',
        tran.get('log/online_reward_pred', tran.get('log/reward_pred', 0.0))))
    reward_error = abs(raw_reward - reward_pred)
    if use_latent_kl:
      raw += w_kl * latent_kl
    if use_reward_error:
      raw += w_reward * reward_error
  elif score_source in ('latent_reward', 'default', 'latent_kl'):
    if use_latent_kl:
      raw += w_kl * latent_kl
    if use_reward_error:
      raw += w_reward * reward_error
  else:
    raise ValueError(f'Unknown fault score source: {score_source}')

  normalized, context_key, context_level, context_p95, context_zscore = (
      normalize_transition_score(raw, tran, stats, config))
  context_excess_raw = max(0.0, float(raw) - float(context_p95))
  score_prev = _scalar(tran.get('log/fault_score_prev', 0.0))
  score_delta = max(0.0, float(normalized) - float(score_prev))
  clip = float(_cfg_get(config, 'clip', 1.0))
  clipped = float(np.clip(normalized, 0.0, clip))
  beta = float(
      _cfg_get(config, 'beta', 0.1)
      if beta_override is None else beta_override)
  log_only = bool(_cfg_get(config, 'log_only', True)) or force_log_only
  gate_active = reward_gate_active(tran, config)
  fault_bonus, reward_score, reward_threshold, reward_mode_id = (
      compute_fault_bonus(
          normalized, clipped, beta, config, score_delta=score_delta))
  if not gate_active:
    fault_bonus = 0.0
    reward_score = 0.0
  if not log_only:
    augmented_reward = learning_base_reward + fault_bonus

  return {
      'raw_reward': float(raw_reward),
      'task_reward_delivered': float(delivered_reward),
      'task_reward_pre_fault': float(pre_fault_reward),
      'task_reward_source_id': float(reward_source_id),
      'incoming_reward': float(incoming_reward),
      'learning_base_reward': float(learning_base_reward),
      'augmented_reward': float(augmented_reward),
      'latent_kl_surprise': float(latent_kl),
      'reward_prediction_error': float(reward_error),
      'reward_pred': float(reward_pred),
      'novelty_kl_lhs': float(novelty_lhs),
      'novelty_kl_bound': float(novelty_bound),
      'novelty_bound_violation': float(novelty_violation),
      'novelty_bound_score': float(novelty_bound_score),
      'novelty_bound_triggered': float(novelty_bound_triggered),
      'fault_score_raw': float(raw),
      'fault_score': float(normalized),
      'fault_context_key': context_key,
      'fault_context_level_id': float(CONTEXT_LEVEL_IDS[context_level]),
      'fault_context_p95': float(context_p95),
      'fault_context_zscore': float(context_zscore),
      'fault_context_excess_raw': float(context_excess_raw),
      'fault_score_prev': float(score_prev),
      'fault_score_delta': float(score_delta),
      'clipped_fault_score': float(clipped),
      'fault_beta': float(beta),
      'fault_reward_gate_active': float(gate_active),
      'fault_reward_bonus': float(fault_bonus),
      'fault_reward_score': float(reward_score),
      'fault_reward_threshold': float(reward_threshold),
      'fault_reward_mode_id': float(reward_mode_id),
  }


class FaultRewardTracker:
  """Stateful diversity rewards and optional episodic task constraint."""

  def __init__(self, config=None):
    self.config = config
    self.beta = float(_cfg_get(config, 'beta', 0.1))
    self.semantic_seen = defaultdict(set)
    self.suspicious_seen = defaultdict(set)
    self.semantic_seen_global = set()
    self.suspicious_seen_global = set()
    self.suspicious_hits = defaultdict(int)
    self.suspicious_repeats = defaultdict(int)
    self.task_return = defaultdict(float)
    self.task_ema = None
    self.constraint_lambda = float(_cfg_get(
        config, 'constraint_lambda_init', 1.0))
    self.completed_episodes = 0

  def reset_worker(self, worker):
    self.semantic_seen[worker].clear()
    self.suspicious_seen[worker].clear()
    self.suspicious_hits[worker] = 0
    self.suspicious_repeats[worker] = 0
    self.task_return[worker] = 0.0

  def apply(self, tran, worker, result, force_log_only=False):
    is_first = bool(np.asarray(tran.get('is_first', False)).reshape(()))
    if is_first:
      self.reset_worker(worker)
    self.task_return[worker] += float(result['raw_reward'])

    # Diversity must use the semantic gameplay context itself. The selected
    # calibration key can be a coarse fallback (or global for legacy modes).
    key = context_keys(tran)['full']
    valid = not is_first
    semantic_new = int(valid and key not in self.semantic_seen[worker])
    if valid:
      self.semantic_seen[worker].add(key)
      self.semantic_seen_global.add(key)

    threshold = float(_cfg_get(
        self.config, 'suspicious_threshold', 1.0))
    suspicious = int(valid and float(result['fault_score']) > threshold)
    unique_suspicious = int(
        suspicious and key not in self.suspicious_seen[worker])
    repeated_suspicious = int(suspicious and not unique_suspicious)
    if suspicious:
      self.suspicious_seen[worker].add(key)
      self.suspicious_seen_global.add(key)
      self.suspicious_hits[worker] += 1
      self.suspicious_repeats[worker] += repeated_suspicious

    coverage_bonus = (
        float(_cfg_get(self.config, 'semantic_coverage_beta', 0.0)) *
        semantic_new)
    unique_bonus = (
        float(_cfg_get(self.config, 'unique_suspicious_beta', 0.0)) *
        unique_suspicious)
    repeat_penalty = (
        float(_cfg_get(self.config, 'repeat_suspicious_penalty', 0.0)) *
        repeated_suspicious)
    diversity_bonus = coverage_bonus + unique_bonus - repeat_penalty
    total_bonus = float(result['fault_reward_bonus']) + diversity_bonus
    log_only = bool(_cfg_get(self.config, 'log_only', True)) or force_log_only
    constraint_mode = str(_cfg_get(
        self.config, 'constraint_mode', 'none') or 'none').lower()
    warmup = int(_cfg_get(self.config, 'constraint_warmup_episodes', 10))
    old_constraint_modes = ('task_lower_bound', 'task_constraint', 'crl')
    scaled_constraint_modes = (
        'task_lower_bound_scaled', 'task_constraint_scaled',
        'crl_scaled', 'lagrangian_scaled')
    constraint_active = (
        constraint_mode in old_constraint_modes + scaled_constraint_modes and
        self.completed_episodes >= warmup)
    constraint_novelty_unscaled = (
        float(_cfg_get(self.config, 'constraint_novelty_scale', 1.0)) *
        (float(result['fault_reward_score']) + diversity_bonus))
    constraint_novelty_scaled = (
        float(_cfg_get(self.config, 'constraint_novelty_scale', 1.0)) *
        total_bonus)
    constraint_task = (
        self.constraint_lambda *
        float(_cfg_get(self.config, 'constraint_task_scale', 1.0)) *
        float(result['raw_reward']))
    if log_only:
      result['augmented_reward'] = float(result['incoming_reward'])
    elif constraint_active and constraint_mode in scaled_constraint_modes:
      result['augmented_reward'] = constraint_novelty_scaled + constraint_task
    elif constraint_active:
      result['augmented_reward'] = constraint_novelty_unscaled + constraint_task
    elif constraint_mode in old_constraint_modes + scaled_constraint_modes:
      result['augmented_reward'] = float(result['learning_base_reward'])
    else:
      result['augmented_reward'] = float(result['learning_base_reward']) + total_bonus
    result.update({
        'semantic_context_new': float(semantic_new),
        'semantic_context_coverage_episode': float(
            len(self.semantic_seen[worker])),
        'semantic_context_coverage_cumulative': float(
            len(self.semantic_seen_global)),
        'suspicious_context': float(suspicious),
        'unique_suspicious_context': float(unique_suspicious),
        'repeated_suspicious_context': float(repeated_suspicious),
        'unique_suspicious_context_episode': float(
            len(self.suspicious_seen[worker])),
        'unique_suspicious_context_cumulative': float(
            len(self.suspicious_seen_global)),
        'repeated_suspicious_context_ratio': float(
            self.suspicious_repeats[worker] /
            max(self.suspicious_hits[worker], 1)),
        'semantic_coverage_bonus': float(coverage_bonus),
        'unique_suspicious_bonus': float(unique_bonus),
        'repeat_suspicious_penalty': float(repeat_penalty),
        'diversity_reward_bonus': float(diversity_bonus),
        'fault_reward_bonus_total': float(total_bonus),
        'task_score_ema': float(
            self.task_ema if self.task_ema is not None else 0.0),
        'fault_beta_next': float(self.beta),
        'constraint_active': float(constraint_active),
        'constraint_lambda': float(self.constraint_lambda),
        'constraint_novelty_reward': float(
            constraint_novelty_scaled
            if constraint_mode in scaled_constraint_modes
            else constraint_novelty_unscaled),
        'constraint_task_reward': float(constraint_task),
        'constraint_violation': float(
            float(_cfg_get(self.config, 'constraint_task_target', 0.0)) -
            (self.task_ema if self.task_ema is not None else 0.0)),
    })

    if bool(np.asarray(tran.get('is_last', False)).reshape(())):
      self._update_beta(self.task_return[worker])
      self._update_constraint()
      result['task_score_ema'] = float(self.task_ema)
      result['fault_beta_next'] = float(self.beta)
      result['constraint_lambda'] = float(self.constraint_lambda)
      result['constraint_violation'] = float(
          float(_cfg_get(self.config, 'constraint_task_target', 0.0)) -
          self.task_ema)
    return result

  def _update_beta(self, task_score):
    rate = float(_cfg_get(self.config, 'adaptive_task_ema', 0.1))
    self.task_ema = (
        float(task_score) if self.task_ema is None else
        (1.0 - rate) * self.task_ema + rate * float(task_score))
    if not bool(_cfg_get(self.config, 'adaptive_beta', False)):
      return
    target = float(_cfg_get(self.config, 'adaptive_task_target', 0.0))
    if target <= 0.0:
      return
    if self.task_ema < target:
      self.beta *= float(_cfg_get(self.config, 'adaptive_beta_down', 0.9))
    else:
      self.beta *= float(_cfg_get(self.config, 'adaptive_beta_up', 1.05))
    self.beta = float(np.clip(
        self.beta,
        float(_cfg_get(self.config, 'adaptive_beta_min', 0.0)),
        float(_cfg_get(self.config, 'adaptive_beta_max', 1.0))))

  def _update_constraint(self):
    self.completed_episodes += 1
    mode = str(_cfg_get(
        self.config, 'constraint_mode', 'none') or 'none').lower()
    warmup = int(_cfg_get(self.config, 'constraint_warmup_episodes', 10))
    if mode not in (
        'task_lower_bound', 'task_constraint', 'crl',
        'task_lower_bound_scaled', 'task_constraint_scaled',
        'crl_scaled', 'lagrangian_scaled'):
      return
    if self.completed_episodes <= warmup or self.task_ema is None:
      return
    target = float(_cfg_get(self.config, 'constraint_task_target', 0.0))
    violation = target - self.task_ema
    self.constraint_lambda += (
        float(_cfg_get(self.config, 'constraint_lambda_lr', 0.05)) *
        violation)
    self.constraint_lambda = float(np.clip(
        self.constraint_lambda,
        float(_cfg_get(self.config, 'constraint_lambda_min', 0.0)),
        float(_cfg_get(self.config, 'constraint_lambda_max', 10.0))))


def add_transition_fault_logs(tran, result):
  tran['log/raw_reward'] = np.float32(result['raw_reward'])
  tran['log/task_reward_delivered'] = np.float32(
      result['task_reward_delivered'])
  tran['log/task_reward_pre_fault'] = np.float32(
      result['task_reward_pre_fault'])
  tran['log/task_reward_source_id'] = np.float32(
      result['task_reward_source_id'])
  tran['log/augmented_reward'] = np.float32(result['augmented_reward'])
  tran['log/fault_score_raw'] = np.float32(result['fault_score_raw'])
  tran['log/fault_score'] = np.float32(result['fault_score'])
  tran['log/fault_context_level_id'] = np.float32(
      result['fault_context_level_id'])
  tran['log/fault_context_p95'] = np.float32(result['fault_context_p95'])
  tran['log/fault_context_zscore'] = np.float32(
      result['fault_context_zscore'])
  tran['log/fault_context_excess_raw'] = np.float32(
      result['fault_context_excess_raw'])
  tran['log/fault_score_prev'] = np.float32(result['fault_score_prev'])
  tran['log/fault_score_delta'] = np.float32(result['fault_score_delta'])
  tran['log/clipped_fault_score'] = np.float32(result['clipped_fault_score'])
  tran['log/latent_kl_surprise'] = np.float32(result['latent_kl_surprise'])
  tran['log/reward_prediction_error'] = np.float32(
      result['reward_prediction_error'])
  tran['log/reward_pred'] = np.float32(result['reward_pred'])
  for key in (
      'novelty_kl_lhs', 'novelty_kl_bound', 'novelty_bound_violation',
      'novelty_bound_score', 'novelty_bound_triggered'):
    tran[f'log/{key}'] = np.float32(result[key])
  tran['log/fault_beta'] = np.float32(result['fault_beta'])
  tran['log/fault_reward_gate_active'] = np.float32(
      result['fault_reward_gate_active'])
  tran['log/fault_reward_bonus'] = np.float32(result['fault_reward_bonus'])
  tran['log/fault_reward_score'] = np.float32(result['fault_reward_score'])
  tran['log/fault_reward_threshold'] = np.float32(
      result['fault_reward_threshold'])
  tran['log/fault_reward_mode_id'] = np.float32(
      result['fault_reward_mode_id'])
  for key in (
      'semantic_context_new', 'semantic_context_coverage_episode',
      'semantic_context_coverage_cumulative', 'suspicious_context',
      'unique_suspicious_context', 'repeated_suspicious_context',
      'unique_suspicious_context_episode',
      'unique_suspicious_context_cumulative',
      'repeated_suspicious_context_ratio', 'semantic_coverage_bonus',
      'unique_suspicious_bonus', 'repeat_suspicious_penalty',
      'diversity_reward_bonus', 'fault_reward_bonus_total',
      'task_score_ema', 'fault_beta_next'):
    if key in result:
      tran[f'log/{key}'] = np.float32(result[key])
  for key in (
      'constraint_active', 'constraint_lambda',
      'constraint_novelty_reward', 'constraint_task_reward',
      'constraint_violation'):
    if key in result:
      tran[f'log/{key}'] = np.float32(result[key])


def add_reference_outputs(outs, ref_outs):
  """Copy reference-agent fault metrics to log keys and keep legacy aliases."""
  mapping = {
      'fault/fault_score_raw': 'log/ref_fault_score_raw',
      'fault/fault_score': 'log/ref_fault_score',
      'fault/latent_kl_surprise': 'log/ref_latent_kl_surprise',
      'fault/reward_prediction_error': 'log/ref_reward_prediction_error',
      'fault/reward_pred': 'log/ref_reward_pred',
      'fault/novelty_kl_lhs': 'log/ref_novelty_kl_lhs',
      'fault/novelty_kl_bound': 'log/ref_novelty_kl_bound',
      'fault/novelty_bound_violation': 'log/ref_novelty_bound_violation',
      'fault/novelty_bound_score': 'log/ref_novelty_bound_score',
      'fault/novelty_bound_triggered': 'log/ref_novelty_bound_triggered',
      'bug/score': 'log/ref_bug_score',
      'bug/kl': 'log/ref_bug_kl',
      'bug/reward_err': 'log/ref_bug_reward_err',
      'bug/continue_err': 'log/ref_bug_continue_err',
  }
  for key, value in ref_outs.items():
    if key in mapping:
      outs[mapping[key]] = value

  if 'log/ref_fault_score' in outs:
    outs.setdefault('log/ref_bug_score', outs['log/ref_fault_score'])
  if 'log/ref_latent_kl_surprise' in outs:
    outs.setdefault('log/ref_bug_kl', outs['log/ref_latent_kl_surprise'])
  if 'log/ref_reward_prediction_error' in outs:
    outs.setdefault(
        'log/ref_bug_reward_err', outs['log/ref_reward_prediction_error'])


def trace_record(tran, global_step, worker, episode_id, episode_step):
  bug_triggered = int(
      _scalar(tran.get('log/fault_applied', 0.0)) > 0.5 or
      _scalar(tran.get('log/semantic_fault_applied', 0.0)) > 0.5)
  bug_id = int(_scalar(tran.get('log/fault_type_id', 0.0)))
  bug_names = MINIGRID_BUG_TYPE_NAMES if 'log/bug_type_id' in tran else BUG_TYPE_NAMES
  record = {
      'global_step': int(global_step),
      'worker': int(worker),
      'episode_id': int(episode_id),
      'episode_step': int(episode_step),
      'action': _to_python(tran.get('action', 0)),
      'raw_reward': _scalar(tran.get('log/raw_reward', tran.get('reward', 0.0))),
      'task_reward_delivered': _scalar(tran.get(
          'log/task_reward_delivered', tran.get('log/env_reward', 0.0))),
      'task_reward_pre_fault': _scalar(tran.get(
          'log/task_reward_pre_fault', tran.get('log/task_reward_raw', 0.0))),
      'task_reward_source_id': int(_scalar(tran.get(
          'log/task_reward_source_id', 0.0))),
      'augmented_reward': _scalar(tran.get(
          'log/augmented_reward', tran.get('reward', 0.0))),
      'learning_reward': _scalar(tran.get('reward', 0.0)),
      'fault_score': _scalar(tran.get('log/fault_score', 0.0)),
      'fault_context_key': context_keys(tran)['full'],
      'fault_context_level_id': int(_scalar(
          tran.get('log/fault_context_level_id', 0.0))),
      'fault_context_p95': _scalar(
          tran.get('log/fault_context_p95', 0.0)),
      'fault_context_zscore': _scalar(
          tran.get('log/fault_context_zscore', 0.0)),
      'fault_context_excess_raw': _scalar(
          tran.get('log/fault_context_excess_raw', 0.0)),
      'fault_score_raw': _scalar(tran.get('log/fault_score_raw', 0.0)),
      'fault_score_prev': _scalar(tran.get('log/fault_score_prev', 0.0)),
      'fault_score_delta': _scalar(tran.get('log/fault_score_delta', 0.0)),
      'clipped_fault_score': _scalar(tran.get('log/clipped_fault_score', 0.0)),
      'fault_reward_gate_active': _scalar(
          tran.get('log/fault_reward_gate_active', 0.0)),
      'fault_reward_bonus': _scalar(
          tran.get('log/fault_reward_bonus', 0.0)),
      'fault_reward_score': _scalar(
          tran.get('log/fault_reward_score', 0.0)),
      'rnd_intrinsic_reward': _scalar(
          tran.get('log/rnd_intrinsic_reward', 0.0)),
      'rnd_reward_bonus': _scalar(
          tran.get('log/rnd_reward_bonus', 0.0)),
      'rnd_alpha': _scalar(tran.get('log/rnd_alpha', 0.0)),
      'fault_reward_threshold': _scalar(
          tran.get('log/fault_reward_threshold', 0.0)),
      'fault_reward_mode_id': int(_scalar(
          tran.get('log/fault_reward_mode_id', 0.0))),
      'latent_kl_surprise': _scalar(tran.get('log/latent_kl_surprise', 0.0)),
      'reward_prediction_error': _scalar(
          tran.get('log/reward_prediction_error', 0.0)),
      'novelty_kl_lhs': _scalar(tran.get('log/novelty_kl_lhs', 0.0)),
      'novelty_kl_bound': _scalar(
          tran.get('log/novelty_kl_bound', 0.0)),
      'novelty_bound_violation': _scalar(
          tran.get('log/novelty_bound_violation', 0.0)),
      'novelty_bound_score': _scalar(
          tran.get('log/novelty_bound_score', 0.0)),
      'novelty_bound_triggered': int(_scalar(
          tran.get('log/novelty_bound_triggered', 0.0)) > 0.5),
      'bug_triggered': bug_triggered,
      'bug_id': bug_id,
      'bug_type': bug_names.get(bug_id, 'unknown'),
      'unique_bug_count_cumulative': int(_scalar(
          tran.get('log/fault_count_cumulative', 0.0))),
      'unique_tile_coverage_cumulative': int(_scalar(
          tran.get('log/unique_tiles_visited', 0.0))),
      'is_first': bool(np.asarray(tran.get('is_first', False)).reshape(())),
      'is_last': bool(np.asarray(tran.get('is_last', False)).reshape(())),
      'is_terminal': bool(np.asarray(
          tran.get('is_terminal', False)).reshape(())),
  }
  for key in (
      'semantic_context_new', 'semantic_context_coverage_episode',
      'semantic_context_coverage_cumulative', 'suspicious_context',
      'unique_suspicious_context', 'repeated_suspicious_context',
      'unique_suspicious_context_episode',
      'unique_suspicious_context_cumulative',
      'repeated_suspicious_context_ratio', 'semantic_coverage_bonus',
      'unique_suspicious_bonus', 'repeat_suspicious_penalty',
      'diversity_reward_bonus', 'fault_reward_bonus_total',
      'task_score_ema', 'fault_beta_next'):
    record[key] = _scalar(tran.get(f'log/{key}', 0.0))
  for key in (
      'constraint_active', 'constraint_lambda',
      'constraint_novelty_reward', 'constraint_task_reward',
      'constraint_violation'):
    record[key] = _scalar(tran.get(f'log/{key}', 0.0))
  record.update(context_components(tran))
  for key in (
      'log/fault_applied', 'log/fault_episode', 'log/semantic_fault_applied',
      'log/semantic_fault_episode', 'log/tile_coverage_ratio',
      'log/fault_assigned_type_id', 'log/fault_condition_met',
      'log/fault_draw_eligible', 'log/fault_draw_success',
      'log/fault_new_injection', 'log/fault_action_release',
      'log/fault_reward_release', 'log/fault_affected_step',
      'log/fault_action_changed', 'log/fault_reward_changed',
      'log/fault_cooldown_remaining', 'log/fault_draw_probability',
      'log/rnd_update_enabled', 'log/achievement_count', 'log/health',
      'log/food', 'log/drink', 'log/energy'):
    if key in tran:
      record[key] = _to_python(tran[key])
  return record


def summarize(values, prefix):
  values = np.asarray(values, np.float64)
  values = values[np.isfinite(values)]
  if values.size == 0:
    return {
        f'{prefix}_mean': 0.0,
        f'{prefix}_std': 0.0,
        f'{prefix}_p90': 0.0,
        f'{prefix}_p95': 0.0,
        f'{prefix}_p99': 0.0,
        f'{prefix}_count': 0,
    }
  return {
      f'{prefix}_mean': float(np.mean(values)),
      f'{prefix}_std': float(np.std(values)),
      f'{prefix}_p90': float(np.quantile(values, 0.90)),
      f'{prefix}_p95': float(np.quantile(values, 0.95)),
      f'{prefix}_p99': float(np.quantile(values, 0.99)),
      f'{prefix}_count': int(values.size),
  }


def write_json(path, data):
  path = Path(str(path)).expanduser()
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write('\n')


def _scalar(x, default=0.0):
  if x is None:
    return default
  arr = np.asarray(x)
  if arr.size == 0:
    return default
  return float(arr.reshape(-1)[0])


def _to_python(x):
  if isinstance(x, np.ndarray):
    if x.ndim == 0:
      return x.item()
    return x.tolist()
  if isinstance(x, (np.bool_, np.integer, np.floating)):
    return x.item()
  return x

import json
import os
from pathlib import Path
from collections import deque, defaultdict

import crafter
import elements
import embodied
import numpy as np
import math


FAULT_FAMILY_IDS = {
    'none': 0,
    'legacy_action_exec': 1,
    'action_exec': 2,
    'context_exec': 3,
    'reward_timing': 4,
    'termination_logic': 5,
    'semantic_high_level': 6,
}

FAULT_FREQUENCY_TIER_IDS = {
    'custom': 0,
    'diagnostic': 1,
    'benchmark': 2,
    'realistic': 3,
    'sparse': 4,
}

FAULT_FREQUENCY_TIER_DEFAULTS = {
    # Frequent enough for instrumentation checks and visual debugging.
    'diagnostic': {
        'CRAFTER_FAULT_EP_PROB': '0.5',
        'CRAFTER_FAULT_SEVERITIES': '0.2,0.3',
        'CRAFTER_SEMANTIC_FAULT_EP_PROB': '0.5',
        'CRAFTER_SEMANTIC_FAULT_MANIFEST_PROB': '1.0',
    },
    # Main benchmark tier: trigger contexts are reachable, manifestations are
    # not guaranteed, so trigger reach and manifestation can be analyzed apart.
    'benchmark': {
        'CRAFTER_FAULT_EP_PROB': '0.25',
        'CRAFTER_FAULT_SEVERITIES': '0.05,0.1',
        'CRAFTER_SEMANTIC_FAULT_EP_PROB': '0.25',
        'CRAFTER_SEMANTIC_FAULT_MANIFEST_PROB': '0.5',
    },
    # Evaluation-only sparse tiers for realism checks.
    'realistic': {
        'CRAFTER_FAULT_EP_PROB': '0.1',
        'CRAFTER_FAULT_SEVERITIES': '0.01,0.05',
        'CRAFTER_SEMANTIC_FAULT_EP_PROB': '0.1',
        'CRAFTER_SEMANTIC_FAULT_MANIFEST_PROB': '0.2',
    },
    'sparse': {
        'CRAFTER_FAULT_EP_PROB': '0.03',
        'CRAFTER_FAULT_SEVERITIES': '0.005,0.01',
        'CRAFTER_SEMANTIC_FAULT_EP_PROB': '0.03',
        'CRAFTER_SEMANTIC_FAULT_MANIFEST_PROB': '0.05',
    },
}


FAULT_PROFILE_ALIASES = {
    # Backward-compatible names used by older experiment scripts.
    'train': 'benchmark_train',
    'eval': 'benchmark_seen',
    'eval_seen': 'benchmark_seen',
    'eval_holdout': 'benchmark_holdout',
    # Research benchmark v2 aliases. These are intentionally opt-in so older
    # runs keep their exact fault distribution.
    'v2_train': 'benchmark_v2_train',
    'v2_seen': 'benchmark_v2_seen',
    'v2_holdout': 'benchmark_v2_holdout',
    'v2_sparse': 'benchmark_v2_sparse',
}

FAULT_PROFILE_IDS = {
    'unknown': 0,
    'benchmark_train': 1,
    'benchmark_seen': 2,
    'benchmark_holdout': 3,
    'diagnostic_train': 4,
    'benchmark_v2_train': 5,
    'benchmark_v2_seen': 6,
    'benchmark_v2_holdout': 7,
    'benchmark_v2_sparse': 8,
}

FAULT_PROFILE_DEFAULTS = {
    # Main training suite. These faults are tied to ordinary Crafter progress
    # loops: collecting/crafting rewards, repeating useful actions, then
    # switching to the next action after a successful transition.
    'benchmark_train': {
        'families': 'action_exec,context_exec,reward_timing',
        'action': (
            'remap_after_success_switch,delay_after_success,'
            'sticky_after_repeat_switch'),
        'context': 'ignore_nonzero_after_reward,ignore_nonzero_after_two_rewards',
        'reward': 'reward_delay_on_positive,reward_scale_half_on_positive_switch',
        'termination': '',
        'cooldown': '8',
        'stochastic_manifest': True,
    },
    # Same subtypes as training. This split estimates in-distribution fault
    # seeking, not generalization.
    'benchmark_seen': {
        'families': 'action_exec,context_exec,reward_timing',
        'action': (
            'remap_after_success_switch,delay_after_success,'
            'sticky_after_repeat_switch'),
        'context': 'ignore_nonzero_after_reward,ignore_nonzero_after_two_rewards',
        'reward': 'reward_delay_on_positive,reward_scale_half_on_positive_switch',
        'termination': '',
        'cooldown': '0',
        'stochastic_manifest': True,
    },
    # Unseen but related Crafter transition faults. They still arise from
    # natural gameplay contexts, but require revisit, delayed follow-up, or
    # repeated successful progress rather than the exact training triggers.
    'benchmark_holdout': {
        'families': 'action_exec,context_exec,reward_timing',
        'action': (
            'revisit_action_delay,delayed_switch_failure,'
            'remap_after_repeat_switch'),
        'context': 'revisit_action_ignore',
        'reward': 'reward_zero_after_repeat_switch,reward_delay_after_two_rewards',
        'termination': '',
        'cooldown': '0',
        'stochastic_manifest': True,
    },
    # Explicit smoke-test profile. Use this for unit tests and quick visual
    # checks where deterministic manifestation is useful.
    'diagnostic_train': {
        'families': 'action_exec,context_exec,reward_timing,termination_logic',
        'action': (
            'remap_after_success_switch,delay_after_success,'
            'sticky_after_repeat_switch,remap_after_repeat_switch,'
            'delay_after_late_episode_switch'),
        'context': (
            'ignore_nonzero_after_reward,ignore_switch_late_episode,'
            'ignore_nonzero_after_two_rewards,revisit_action_ignore'),
        'reward': (
            'reward_delay_on_positive,reward_scale_half_on_positive_switch,'
            'reward_zero_on_positive,reward_zero_after_repeat_switch,'
            'reward_delay_after_two_rewards'),
        'termination': (
            'early_done_after_success_switch,early_done_after_repeat_switch'),
        'cooldown': '0',
        'stochastic_manifest': False,
    },
    # Benchmark v2 focuses on game-QA style faults. The train/seen split keeps
    # ordinary progress contexts, while holdout requires more accumulated game
    # state such as revisits, repeated progress, or delayed follow-up actions.
    'benchmark_v2_train': {
        'families': 'action_exec,context_exec,reward_timing',
        'action': (
            'delay_after_success,remap_after_success_switch,'
            'sticky_after_repeat_switch'),
        'context': 'ignore_nonzero_after_reward',
        'reward': 'reward_delay_on_positive,reward_scale_half_on_positive_switch',
        'termination': '',
        'cooldown': '12',
        'stochastic_manifest': True,
    },
    'benchmark_v2_seen': {
        'families': 'action_exec,context_exec,reward_timing',
        'action': (
            'delay_after_success,remap_after_success_switch,'
            'sticky_after_repeat_switch'),
        'context': 'ignore_nonzero_after_reward',
        'reward': 'reward_delay_on_positive,reward_scale_half_on_positive_switch',
        'termination': '',
        'cooldown': '8',
        'stochastic_manifest': True,
    },
    'benchmark_v2_holdout': {
        'families': 'action_exec,context_exec,reward_timing',
        'action': (
            'revisit_action_delay,delayed_switch_failure,'
            'remap_after_repeat_switch'),
        'context': 'revisit_action_ignore,ignore_nonzero_after_two_rewards',
        'reward': 'reward_zero_after_repeat_switch,reward_delay_after_two_rewards',
        'termination': '',
        'cooldown': '8',
        'stochastic_manifest': True,
    },
    # Same operators as holdout, but with longer cooldown. Use together with
    # CRAFTER_FAULT_FREQ_TIER=realistic or sparse for final sparse evaluation.
    'benchmark_v2_sparse': {
        'families': 'action_exec,context_exec,reward_timing',
        'action': (
            'revisit_action_delay,delayed_switch_failure,'
            'remap_after_repeat_switch'),
        'context': 'revisit_action_ignore,ignore_nonzero_after_two_rewards',
        'reward': 'reward_zero_after_repeat_switch,reward_delay_after_two_rewards',
        'termination': '',
        'cooldown': '16',
        'stochastic_manifest': True,
    },
}


def _apply_fault_frequency_tier_env():
  tier = os.getenv(
      'CRAFTER_FAULT_FREQ_TIER',
      os.getenv('CRAFTER_FAULT_RARITY_TIER', 'custom')).strip().lower()
  tier = tier or 'custom'
  if tier not in FAULT_FREQUENCY_TIER_IDS:
    raise ValueError(
        f'Unknown CRAFTER_FAULT_FREQ_TIER={tier!r}. Expected one of '
        f'{sorted(FAULT_FREQUENCY_TIER_IDS)}.')
  for key, value in FAULT_FREQUENCY_TIER_DEFAULTS.get(tier, {}).items():
    os.environ.setdefault(key, value)
  return tier


FAULT_TYPE_IDS = {
    'none': 0,
    'drop_to_fallback': 1,
    'remap_after_success_switch': 2,
    'delay_after_success': 3,
    'sticky_after_repeat_switch': 4,
    'ignore_nonzero_after_reward': 5,
    'ignore_switch_late_episode': 6,
    'reward_delay_on_positive': 7,
    'reward_scale_half_on_positive_switch': 8,
    'reward_zero_on_positive': 9,
    'early_done_after_success_switch': 10,
    'remap_after_repeat_switch': 11,
    'delay_after_late_episode_switch': 12,
    'ignore_nonzero_after_two_rewards': 13,
    'reward_zero_after_repeat_switch': 14,
    'reward_delay_after_two_rewards': 15,
    'early_done_after_repeat_switch': 16,
    'revisit_action_ignore': 17,
    'revisit_action_delay': 18,
    'delayed_switch_failure': 19,
    'tool_collect_desync_on_upgrade': 20,
    'collect_result_delayed_after_tool_upgrade': 21,
    'upgrade_branch_inconsistent_collect_behavior': 22,
    'craft_result_missing_on_retry': 23,
    'craft_output_delayed_on_retry': 24,
    'recipe_precondition_mischeck_on_retry': 25,
    'recipe_retry_requires_revisit': 26,
    'station_place_ghost_on_relocate': 27,
    'station_second_use_inconsistent_after_placement': 28,
    'station_usable_flag_broken_after_relocate': 29,
    'station_state_partial_reset_after_relocate': 30,
    'achievement_unlock_missing_after_valid_progress': 31,
    'achievement_unlock_missing_after_reconfirm': 32,
    'progress_confirmation_requires_revisit': 33,
    'delayed_inventory_desync_after_station_use': 34,
}

# ====================================================================
# 1. RND(Random Network Distillation) 모듈 (순수 NumPy 버전)
# JAX와의 CuDNN 충돌을 원천 차단하기 위해 PyTorch를 완전히 제거했습니다.
# ====================================================================
class RunningMeanStd:
  def __init__(self, epsilon=1e-4):
    self.mean = 0.0
    self.var = 1.0
    self.count = float(epsilon)

  def update(self, values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
      return
    batch_mean = float(arr.mean())
    batch_var = float(arr.var())
    batch_count = float(arr.size)
    self._update_from_moments(batch_mean, batch_var, batch_count)

  def _update_from_moments(self, batch_mean, batch_var, batch_count):
    delta = batch_mean - self.mean
    total_count = self.count + batch_count
    new_mean = self.mean + delta * batch_count / total_count

    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
    new_var = m2 / total_count

    self.mean = float(new_mean)
    self.var = float(max(new_var, 1e-12))
    self.count = float(total_count)


class RNDModule:
  def __init__(
      self,
      obs_shape=(64, 64, 3),
      downsample=4,
      hidden_dim=128,
      output_dim=128,
      lr=0.005,
      seed=0,
  ):
    self.lr = float(lr)
    self.output_dim = int(output_dim)
    self.downsample = int(max(1, downsample))
    self.rng = np.random.default_rng(seed)

    sample = np.zeros(obs_shape, dtype=np.uint8)[::self.downsample, ::self.downsample, :]
    self.input_dim = int(sample.size)

    # 1. 고정된 타겟 네트워크 (학습 안 함, 랜덤 가중치)
    self.W1_t = self.rng.standard_normal((self.input_dim, hidden_dim)) / np.sqrt(self.input_dim)
    self.b1_t = np.zeros(hidden_dim, dtype=np.float32)
    self.W2_t = self.rng.standard_normal((hidden_dim, output_dim)) / np.sqrt(hidden_dim)
    self.b2_t = np.zeros(output_dim, dtype=np.float32)

    # 2. 타겟을 예측할 프레디터 네트워크 (학습 함)
    self.W1_p = self.rng.standard_normal((self.input_dim, hidden_dim)) / np.sqrt(self.input_dim)
    self.b1_p = np.zeros(hidden_dim, dtype=np.float32)
    self.W2_p = self.rng.standard_normal((hidden_dim, output_dim)) / np.sqrt(hidden_dim)
    self.b2_p = np.zeros(output_dim, dtype=np.float32)

  def _preprocess(self, obs):
    small_obs = obs[::self.downsample, ::self.downsample, :]
    return small_obs.flatten().astype(np.float32) / 255.0

  def compute_intrinsic_reward(self, obs, update=True):
    x = self._preprocess(obs)

    # 1. Target 네트워크 통과 (Forward)
    z1_t = np.dot(x, self.W1_t) + self.b1_t
    h_t = np.maximum(0, z1_t)  # ReLU
    y_t = np.dot(h_t, self.W2_t) + self.b2_t

    # 2. Predictor 네트워크 통과 (Forward)
    z1_p = np.dot(x, self.W1_p) + self.b1_p
    h_p = np.maximum(0, z1_p)  # ReLU
    y_p = np.dot(h_p, self.W2_p) + self.b2_p

    # 3. 내재적 보상 (MSE 오차 계산)
    diff = y_p - y_t
    mse = float(np.mean(diff ** 2))

    # 4. 프레디터 학습 (NumPy 수동 Backpropagation)
    if update:
      dy_p = diff / self.output_dim

      dW2_p = np.outer(h_p, dy_p)
      db2_p = dy_p

      dh_p = np.dot(self.W2_p, dy_p)
      dz1_p = dh_p * (z1_p > 0)  # ReLU 미분

      dW1_p = np.outer(x, dz1_p)
      db1_p = dz1_p

      # 가중치 업데이트 (Gradient Descent)
      self.W2_p -= self.lr * dW2_p
      self.b2_p -= self.lr * db2_p
      self.W1_p -= self.lr * dW1_p
      self.b1_p -= self.lr * db1_p

    return mse, mse

# ====================================================================
# 2. 메인 Crafter 환경 클래스
# ====================================================================
class Crafter(embodied.Env):

  def __init__(
      self,
      task,
      size=(64, 64),
      logs=False,
      logdir=None,
      seed=None,
      fault_enabled=False,
      action_drop_prob=0.0,
      fallback_action=0,
      fault_start_episode=0,
      fault_seed=None,
      trace_path=None,
  ):
    assert task in ('reward', 'noreward')
    self._fault_frequency_tier = _apply_fault_frequency_tier_env()
    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    self._logs = logs
    self._logdir = logdir and elements.Path(logdir)
    self._logdir and self._logdir.mkdir()

    self._episode = 0
    self._length = None
    self._reward = None
    self._achievements = crafter.constants.achievements.copy()
    self._done = True

    # RNG
    self._fault_seed = fault_seed if fault_seed is not None else seed
    self._rng = np.random.default_rng(self._fault_seed)

    # Legacy fixed fault mode
    self._fault_enabled = bool(fault_enabled)
    self._action_drop_prob = float(action_drop_prob)
    self._fallback_action = int(fallback_action)
    self._fault_start_episode = int(fault_start_episode)

    self._fault_enabled = bool(int(os.getenv('CRAFTER_FAULT', '0')))
    self._action_drop_prob = float(os.getenv('CRAFTER_ACTION_DROP_PROB', '0.0'))
    self._fallback_action = int(os.getenv('CRAFTER_FALLBACK_ACTION', '0'))

    # --------------------------------------------------
    # Fault profile
    # --------------------------------------------------
    self._fault_sampler = bool(int(os.getenv('CRAFTER_FAULT_SAMPLER', '0')))
    self._fault_profile_requested = os.getenv(
        'CRAFTER_FAULT_PROFILE', 'benchmark_train').strip().lower()
    self._fault_profile = FAULT_PROFILE_ALIASES.get(
        self._fault_profile_requested, self._fault_profile_requested)
    if self._fault_profile not in FAULT_PROFILE_DEFAULTS:
      raise ValueError(
          f'Unknown CRAFTER_FAULT_PROFILE={self._fault_profile_requested!r} '
          f'(resolved to {self._fault_profile!r}). Expected one of '
          f'{sorted(FAULT_PROFILE_DEFAULTS)} or aliases '
          f'{sorted(FAULT_PROFILE_ALIASES)}.')
    self._fault_profile_defaults = FAULT_PROFILE_DEFAULTS[self._fault_profile]
    default_stochastic_manifest = (
        '1' if self._fault_profile_defaults.get('stochastic_manifest', False)
        else '0')
    self._fault_stochastic_manifest = bool(int(os.getenv(
        'CRAFTER_FAULT_STOCHASTIC_MANIFEST',
        default_stochastic_manifest)))

    self._fault_episode_prob = float(os.getenv('CRAFTER_FAULT_EP_PROB', '0.3'))
    self._fault_episode_prob_min = float(os.getenv('CRAFTER_FAULT_EP_PROB_MIN', '0.1'))
    self._fault_episode_prob_max = float(os.getenv('CRAFTER_FAULT_EP_PROB_MAX', '0.5'))
    self._fault_episode_prob_step = float(os.getenv('CRAFTER_FAULT_EP_PROB_STEP', '0.02'))

    self._fault_severities = self._parse_csv(
        os.getenv('CRAFTER_FAULT_SEVERITIES', '0.1,0.2'),
        cast=float)
    self._semantic_manifest_prob = float(
        os.getenv('CRAFTER_SEMANTIC_FAULT_MANIFEST_PROB', '1.0'))

    # cooldown
    default_cooldown = str(self._fault_profile_defaults.get('cooldown', '0'))
    self._fault_cooldown_steps = int(os.getenv('CRAFTER_FAULT_COOLDOWN', default_cooldown))
    self._fault_cooldown_min = int(os.getenv('CRAFTER_FAULT_COOLDOWN_MIN', '0'))
    self._fault_cooldown_max = int(os.getenv('CRAFTER_FAULT_COOLDOWN_MAX', '20'))
    self._fault_cooldown_step = int(os.getenv('CRAFTER_FAULT_COOLDOWN_STEP', '1'))
    self._target_fault_apply_rate = float(os.getenv('CRAFTER_TARGET_FAULT_APPLY_RATE', '0.015'))

    # scheduler
    self._adaptive_scheduler = bool(int(os.getenv('CRAFTER_ADAPTIVE_SCHEDULER', '1')))
    self._sched_baseline_score = float(os.getenv('CRAFTER_SCHED_BASELINE_SCORE', '8.0'))
    self._sched_floor_ratio = float(os.getenv('CRAFTER_SCHED_FLOOR_RATIO', '0.7'))
    self._sched_window = int(os.getenv('CRAFTER_SCHED_WINDOW', '30'))

    default_families = self._fault_profile_defaults['families']
    default_action = self._fault_profile_defaults['action']
    default_context = self._fault_profile_defaults['context']
    default_reward = self._fault_profile_defaults['reward']
    default_term = self._fault_profile_defaults['termination']

    self._fault_families = self._parse_csv(
        os.getenv('CRAFTER_FAULT_FAMILIES', default_families))
    self._action_subtypes = self._parse_csv(
        os.getenv('CRAFTER_ACTION_SUBTYPES', default_action))
    self._context_subtypes = self._parse_csv(
        os.getenv('CRAFTER_CONTEXT_SUBTYPES', default_context))
    self._reward_subtypes = self._parse_csv(
        os.getenv('CRAFTER_REWARD_SUBTYPES', default_reward))
    self._termination_subtypes = self._parse_csv(
        os.getenv('CRAFTER_TERMINATION_SUBTYPES', default_term))

    self._fault_verbose = bool(int(os.getenv('CRAFTER_FAULT_VERBOSE', '0')))

    # --------------------------------------------------
    # Unified output directory and file paths
    # --------------------------------------------------
    explicit_trace = None
    if trace_path is not None:
      explicit_trace = Path(str(trace_path)).expanduser()
    else:
      env_trace = os.getenv('CRAFTER_TRACE_PATH', '').strip()
      if env_trace:
        explicit_trace = Path(env_trace).expanduser()

    env_output_dir = os.getenv('CRAFTER_OUTPUT_DIR', '').strip()

    # Priority:
    #   1) CRAFTER_OUTPUT_DIR (explicit per-run train/eval folder)
    #   2) explicit trace path parent
    #   3) env wrapper logdir
    #   4) default fallback
    if env_output_dir:
      self._output_dir = Path(env_output_dir).expanduser()
    elif explicit_trace is not None:
      self._output_dir = explicit_trace.resolve().parent
    elif self._logdir:
      self._output_dir = Path(str(self._logdir)).expanduser()
    else:
      self._output_dir = Path('/home/railab/logdir')

    self._output_dir.mkdir(parents=True, exist_ok=True)

    # Keep all env-side files in the same directory. If a trace path is
    # explicitly supplied, honor its filename so split eval traces stay apart.
    self._trace_path = (
        explicit_trace if explicit_trace is not None
        else self._output_dir / 'fault_trace.jsonl')
    self._stats_path = self._output_dir / 'stats.jsonl'
    self._episode_summary_path = self._output_dir / 'episode_summary.jsonl'

    self._trace_path.touch(exist_ok=True)
    self._stats_path.touch(exist_ok=True)
    self._episode_summary_path.touch(exist_ok=True)

    if self._fault_verbose:
      print(f'[OutputDir] {self._output_dir}')
      print(f'[FaultTrace] {self._trace_path}')
      print(f'[Stats] {self._stats_path}')
      print(f'[EpisodeSummary] {self._episode_summary_path}')

    # episode state
    self._fault_count = 0
    self._fault_episode = 0
    self._semantic_fault_episode_seen = 0
    self._fault_spec = None

    self._last_reward = 0.0
    self._last_requested_action = 0
    self._prev_executed_action = 0

    self._requested_hist = deque([0, 0], maxlen=2)
    self._executed_hist = deque([0, 0], maxlen=2)
    self._reward_hist = deque([0.0, 0.0, 0.0, 0.0], maxlen=4)

    self._sticky_action = None
    self._sticky_remaining = 0
    self._pending_reward = 0.0
    self._after_positive_window = 0
    self._fault_cooldown = 0

    # adaptive scheduler stats
    self._recent_clean_scores = deque(maxlen=self._sched_window)
    self._recent_fault_apply_rates = deque(maxlen=self._sched_window)
    self._subtype_apply_counts = defaultdict(int)

    # global-step aligned logging for stable plotting
    self._global_step = 0
    self._episode_start_global_step = 0

    # --------------------------------------------------
    # Coverage metrics
    # --------------------------------------------------
    self._coverage_enabled = bool(int(os.getenv('CRAFTER_COVERAGE', '1')))
    self._world_area = int(np.prod(getattr(self._env._world, 'area', size)))

    self._reset_episode_aggregation()

    # --------------------------------------------------
    # Probe-required holdout support
    # --------------------------------------------------
    self._num_actions = int(self._env.action_space.n)

    # episode step counter for delayed-context bugs
    self._episode_step = 0
    self._last_positive_reward_step = -10**9

    # revisit/context memory
    self._ctx_visit_steps = defaultdict(lambda: deque(maxlen=4))
    self._last_ctx_key = None

    # tunable thresholds
    self._revisit_gap = int(os.getenv('CRAFTER_REVISIT_GAP', '12'))
    self._delayed_switch_min = int(os.getenv('CRAFTER_DELAYED_SWITCH_MIN', '2'))
    self._delayed_switch_max = int(os.getenv('CRAFTER_DELAYED_SWITCH_MAX', '3'))

    # --------------------------------------------------
    # Tester-oriented semantic reward shaping
    # --------------------------------------------------
    self._tester_reward_enabled = bool(int(os.getenv('CRAFTER_TESTER_REWARD', '1')))
    self._tester_alpha_task = float(os.getenv('CRAFTER_TESTER_ALPHA_TASK', '1.0'))
    self._tester_ctx_reward = float(os.getenv('CRAFTER_TESTER_CTX_REWARD', '0.02'))
    self._tester_anom_reward = float(os.getenv('CRAFTER_TESTER_ANOM_REWARD', '0.02'))
    self._tester_reproduce_reward = float(os.getenv('CRAFTER_TESTER_REPRODUCE_REWARD', '0.10'))
    self._tester_compare_reward = float(os.getenv('CRAFTER_TESTER_COMPARE_REWARD', '0.10'))
    self._tester_confirm_reward = float(os.getenv('CRAFTER_TESTER_CONFIRM_REWARD', '0.12'))
    self._tester_followup_reward = float(os.getenv('CRAFTER_TESTER_FOLLOWUP_REWARD', '0.08'))

    self._tester_ctx_progress_reward = float(os.getenv('CRAFTER_TESTER_CTX_PROGRESS_REWARD', '0.008'))
    self._tester_ctx_verify_reward = float(os.getenv('CRAFTER_TESTER_CTX_VERIFY_REWARD', '0.03'))
    self._tester_anom_progress_reward = float(os.getenv('CRAFTER_TESTER_ANOM_PROGRESS_REWARD', '0.01'))
    self._tester_anom_verify_reward = float(os.getenv('CRAFTER_TESTER_ANOM_VERIFY_REWARD', '0.03'))
    
    self._tester_repeat_penalty = float(os.getenv('CRAFTER_TESTER_REPEAT_PENALTY', '0.003'))
    self._tester_same_action_cap = int(os.getenv('CRAFTER_TESTER_SAME_ACTION_CAP', '5'))
    self._tester_repeat_max_step = float(os.getenv('CRAFTER_TESTER_REPEAT_MAX_STEP', '0.03'))
    self._tester_repeat_suspend_on_context = bool(int(os.getenv('CRAFTER_TESTER_REPEAT_SUSPEND_ON_CONTEXT', '1')))
    self._tester_repeat_only_on_idle = bool(int(os.getenv('CRAFTER_TESTER_REPEAT_ONLY_ON_IDLE', '1')))
    self._tester_reward_cap = float(os.getenv('CRAFTER_TESTER_REWARD_CAP', '0.35'))

    # --------------------------------------------------
    # [ICLR Novelty] Dynamic Weighting & Saturation
    # --------------------------------------------------
    self._tester_task_saturation_decay = float(os.getenv('CRAFTER_TESTER_TASK_SATURATION', '0.5'))
    self._tester_validation_amp = float(os.getenv('CRAFTER_TESTER_AMP', '5.0'))
    self._tester_task_suppression = float(os.getenv('CRAFTER_TESTER_SUPPRESSION', '0.2'))

    # --------------------------------------------------
    # [Visual Sanity Check] GIF Recording
    # --------------------------------------------------
    self._record_gifs = bool(int(os.getenv('CRAFTER_RECORD_GIFS', '0')))
    self._episode_frames = []

    # --------------------------------------------------
    # [A9] RND Configuration
    # --------------------------------------------------
    self._use_rnd = bool(int(os.getenv('CRAFTER_USE_RND', '0')))
    self._rnd_alpha = float(os.getenv('CRAFTER_RND_ALPHA', '0.05'))
    self._rnd_update = bool(int(os.getenv('CRAFTER_RND_UPDATE', '1')))
    self._rnd_normalize = bool(int(os.getenv('CRAFTER_RND_NORM', '1')))
    self._rnd_clip = float(os.getenv('CRAFTER_RND_CLIP', '5.0'))
    self._rnd_downsample = int(os.getenv('CRAFTER_RND_DOWNSAMPLE', '4'))
    self._rnd_hidden_dim = int(os.getenv('CRAFTER_RND_HIDDEN_DIM', '128'))
    self._rnd_output_dim = int(os.getenv('CRAFTER_RND_OUTPUT_DIM', '128'))
    self._rnd_lr = float(os.getenv('CRAFTER_RND_LR', '0.005'))
    default_rnd_seed = 0 if self._fault_seed is None else int(self._fault_seed)
    self._rnd_seed = int(os.getenv('CRAFTER_RND_SEED', str(default_rnd_seed)))
    self._rnd_rms = RunningMeanStd() if (self._use_rnd and self._rnd_normalize) else None
    if self._use_rnd:
      self._rnd_mod = RNDModule(
          obs_shape=tuple(self._env.observation_space.shape),
          downsample=self._rnd_downsample,
          hidden_dim=self._rnd_hidden_dim,
          output_dim=self._rnd_output_dim,
          lr=self._rnd_lr,
          seed=self._rnd_seed,
      )

  # --------------------------------------------------
  # Helpers
  # --------------------------------------------------
  def _parse_csv(self, raw, cast=str):
    items = []
    for x in raw.split(','):
      x = x.strip()
      if not x:
        continue
      items.append(cast(x))
    return items

  def _choice(self, values, default=None):
    if not values:
      return default
    idx = int(self._rng.integers(0, len(values)))
    return values[idx]

  def _normalize_rnd_reward(self, intrinsic_reward, update_stats=True):
    intrinsic_reward = float(max(0.0, intrinsic_reward))
    if not self._use_rnd or not self._rnd_normalize or self._rnd_rms is None:
      if self._rnd_clip > 0:
        intrinsic_reward = float(np.clip(intrinsic_reward, 0.0, self._rnd_clip))
      return intrinsic_reward

    if update_stats:
      self._rnd_rms.update([intrinsic_reward])

    std = float(np.sqrt(max(self._rnd_rms.var, 1e-12)))
    norm_reward = intrinsic_reward / (std + 1e-8)
    if self._rnd_clip > 0:
      norm_reward = float(np.clip(norm_reward, 0.0, self._rnd_clip))
    return float(norm_reward)

  def _weighted_choice(self, items, counts):
    if not items:
      return None
    weights = []
    for item in items:
      c = counts.get(item, 0)
      weights.append(1.0 / math.sqrt(c + 1.0))
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    idx = int(self._rng.choice(len(items), p=weights))
    return items[idx]

  def _family_subtypes(self, family):
    if family == 'action_exec':
      return self._action_subtypes
    if family == 'context_exec':
      return self._context_subtypes
    if family == 'reward_timing':
      return self._reward_subtypes
    if family == 'termination_logic':
      return self._termination_subtypes
    return []

  def _valid_families(self):
    valid = []
    for fam in self._fault_families:
      if self._family_subtypes(fam):
        valid.append(fam)
    return valid

  def _trigger_name_from_subtype(self, subtype):
    mapping = {
        'remap_after_success_switch': 'after_positive_window + switched_action',
        'delay_after_success': 'after_positive_window',
        'sticky_after_repeat_switch': 'repeat_then_switch',
        'ignore_nonzero_after_reward': 'after_positive_window + nonzero_action',
        'ignore_switch_late_episode': 'late_episode + switched_action',
        'reward_delay_on_positive': 'positive_reward',
        'reward_scale_half_on_positive_switch': 'positive_reward + switched_action',
        'reward_zero_on_positive': 'positive_reward',
        'early_done_after_success_switch': 'after_positive_window + switched_action',
        'drop_to_fallback': 'everywhere',
        'remap_after_repeat_switch': 'repeat_then_switch',
        'delay_after_late_episode_switch': 'late_episode + switched_action',
        'ignore_nonzero_after_two_rewards': 'two_recent_positive_rewards',
        'reward_zero_after_repeat_switch': 'positive_reward + repeat_then_switch',
        'reward_delay_after_two_rewards': 'positive_reward + two_recent_positive_rewards',
        'early_done_after_repeat_switch': 'repeat_then_switch',
        'revisit_action_ignore': 'recent_revisit + nonzero_action',
        'revisit_action_delay': 'recent_revisit + switched_action',
        'delayed_switch_failure': 'positive_reward_after_2to3_steps + switched_action',
    }
    return mapping.get(subtype, 'custom')

  def _sample_fault_spec(self):
    self._fault_spec = None
    self._fault_episode = 0

    if self._fault_sampler:
      if self._rng.random() >= self._fault_episode_prob:
        return

      families = self._valid_families()
      if not families:
        return

      family_counts = {f: sum(
          self._subtype_apply_counts.get(s, 0) for s in self._family_subtypes(f))
          for f in families}
      family = self._weighted_choice(families, family_counts)
      subtype = self._weighted_choice(
          self._family_subtypes(family), self._subtype_apply_counts)

      severity = float(self._choice(self._fault_severities, 0.1))
      self._fault_spec = {
          'family': family,
          'type': subtype,
          'severity': severity,
          'trigger': self._trigger_name_from_subtype(subtype),
      }
      self._fault_episode = 1
      return

    if self._fault_enabled:
      self._fault_spec = {
          'family': 'legacy_action_exec',
          'type': 'drop_to_fallback',
          'severity': self._action_drop_prob,
          'trigger': 'everywhere',
          'fallback_action': self._fallback_action,
      }
      self._fault_episode = 1

  def _is_nonzero(self, action):
    return int(action) != 0

  def _switched_action(self, requested_action):
    requested_action = int(requested_action)
    prev = int(self._last_requested_action)
    return (
        self._is_nonzero(requested_action) and
        self._is_nonzero(prev) and
        requested_action != prev
    )

  def _repeat_then_switch(self, requested_action):
    requested_action = int(requested_action)
    a0, a1 = int(self._requested_hist[0]), int(self._requested_hist[1])
    return (
        self._is_nonzero(a0) and
        a0 == a1 and
        self._is_nonzero(requested_action) and
        requested_action != a1
    )

  def _late_episode(self):
    return self._length is not None and self._length >= 50

  def _recent_positive_count(self, n=4):
    vals = list(self._reward_hist)[-n:]
    return sum(1 for r in vals if float(r) > 0.0)

  def _success_window_active(self):
    return self._after_positive_window > 0

  def _random_other_action(self, requested_action):
    n = int(self._env.action_space.n)
    if n <= 1:
      return int(requested_action)
    candidates = [a for a in range(n) if a != int(requested_action)]
    idx = int(self._rng.integers(0, len(candidates)))
    return int(candidates[idx])

  def _context_key_from_image(self, image):
    image = np.asarray(image)
    if image.ndim != 3:
      return None
    small = image[::8, ::8]
    if small.shape[-1] == 3:
      small = small.mean(-1)
    small = small.astype(np.uint8, copy=False)
    return small.tobytes()

  def _record_context(self, image):
    key = self._context_key_from_image(image)
    self._last_ctx_key = key
    if key is not None:
      self._ctx_visit_steps[key].append(self._episode_step)

  def _reset_coverage_state(self):
    self._visited_tiles = set()
    self._target_cells = set()
    self._target_materials = set()
    self._modified_cells = set()
    self._modified_material_types = set()
    self._world_edit_count = 0
    self._last_world_step_edit_count = 0

  def _tile_coverage_ratio(self):
    return float(len(self._visited_tiles) / max(1, self._world_area))

  def _world_edit_ratio(self):
    return float(len(self._modified_cells) / max(1, self._world_area))

  def _snapshot_world_material_map(self):
    world = getattr(self._env, '_world', None)
    if world is None or not hasattr(world, '_mat_map'):
      return None
    return np.array(world._mat_map, copy=True)

  def _record_player_tile_from_info(self, info):
    if not self._coverage_enabled:
      return
    info = dict(info or {})
    pos = info.get('player_pos', None)
    if pos is None:
      return
    try:
      tile = tuple(int(x) for x in np.asarray(pos).tolist())
    except Exception:
      return
    if len(tile) == 2:
      self._visited_tiles.add(tile)

  def _record_current_target_coverage(self):
    if not self._coverage_enabled:
      return
    world = getattr(self._env, '_world', None)
    player = getattr(self._env, '_player', None)
    if world is None or player is None:
      return
    if not hasattr(player, 'pos') or not hasattr(player, 'facing'):
      return
    pos = np.array(player.pos)
    facing = np.array(player.facing)
    target = tuple((pos + facing).tolist())
    self._target_cells.add(tuple(int(x) for x in target))
    material, _ = world[target]
    if material is not None:
      self._target_materials.add(str(material))

  def _update_world_coverage_from_material_map(self, mat_before):
    self._last_world_step_edit_count = 0
    if not self._coverage_enabled or mat_before is None:
      return
    world = getattr(self._env, '_world', None)
    if world is None or not hasattr(world, '_mat_map'):
      return
    mat_after = world._mat_map
    changed = np.argwhere(mat_before != mat_after)
    self._last_world_step_edit_count = int(len(changed))
    if len(changed) == 0:
      return
    mat_names = getattr(world, '_mat_names', {})
    for x, y in changed:
      x, y = int(x), int(y)
      self._modified_cells.add((x, y))
      self._world_edit_count += 1
      before_id = int(mat_before[x, y])
      after_id = int(mat_after[x, y])
      before_name = mat_names.get(before_id, None)
      after_name = mat_names.get(after_id, None)
      if before_name is not None:
        self._modified_material_types.add(str(before_name))
      if after_name is not None:
        self._modified_material_types.add(str(after_name))

  def _recent_revisit_active(self):
    key = self._last_ctx_key
    if key is None:
      return False
    hist = self._ctx_visit_steps.get(key, None)
    if hist is None or len(hist) < 2:
      return False
    return (hist[-1] - hist[-2]) <= self._revisit_gap

  def _delayed_switch_after_success(self, requested_action):
    delay = int(self._episode_step - self._last_positive_reward_step)
    return (
        self._delayed_switch_min <= delay <= self._delayed_switch_max
        and self._switched_action(requested_action)
    )

  def _reset_tester_reward_state(self):
    self._episode_task_reward = 0.0
    self._episode_tester_bonus = 0.0
    self._episode_repeat_penalty = 0.0
    self._same_action_run = 0
    self._last_requested_action_for_reward = None
    
    # [ICLR Novelty] Achievement counting for saturation
    self._achievement_counts = defaultdict(int)
    
    self._semantic_proc_state = defaultdict(lambda: {
        'ctx': 0,
        'anom': 0,
        'reproduce': 0,
        'compare': 0,
        'confirm': 0,
        'followup': 0,
    })
    self._active_semantic_family = None

  def _semantic_context_keys(self, info):
    info = dict(info or {})
    keys = []
    mapping = [
        ('upgrade_collect', info.get('semantic_ctx_upgrade_collect', info.get('semantic_upgrade_collect', 0))),
        ('retry_craft', info.get('semantic_ctx_retry_craft', info.get('semantic_retry_craft', 0))),
        ('relocate_station', info.get('semantic_ctx_relocate_station', info.get('semantic_relocate_station', 0))),
        ('valid_progress', info.get('semantic_ctx_valid_progress', info.get('semantic_valid_progress', 0))),
        ('station_reuse', info.get('semantic_ctx_station_reuse', info.get('semantic_station_reuse', 0))),
        ('delayed_after_use', info.get('semantic_ctx_delayed_after_use', info.get('semantic_delayed_after_use', 0))),
    ]
    for name, flag in mapping:
      if int(flag) > 0:
        keys.append(name)
    return keys

  def _is_progress_context(self, key):
    return key in ('valid_progress',)

  def _is_progress_fault_type(self, fault_type):
    return fault_type in (
        'achievement_unlock_missing_after_valid_progress',
        'achievement_unlock_missing_after_reconfirm',
        'progress_feedback_delayed_until_next_interaction',
        'progress_confirmation_requires_revisit',
    )

  def _ctx_reward_for_key(self, key):
    if self._is_progress_context(key):
      return self._tester_ctx_progress_reward
    if key in ('retry_craft', 'relocate_station'):
      return self._tester_ctx_verify_reward * 3.0
    return self._tester_ctx_verify_reward

  def _anom_reward_for_type(self, fault_type):
    if self._is_progress_fault_type(fault_type):
      return self._tester_anom_progress_reward
    return self._tester_anom_verify_reward

  def _tester_reward_from_semantics(self, task_reward, requested_action, info):
    task_reward = float(task_reward)
    info = dict(info or {})
    components = {}

    if self._last_requested_action_for_reward is None or int(requested_action) != int(self._last_requested_action_for_reward):
      self._same_action_run = 1
    else:
      self._same_action_run += 1
    self._last_requested_action_for_reward = int(requested_action)

    # --------------------------------------------------
    # 1. Task Saturation (Decay task reward for mastery)
    # --------------------------------------------------
    scaled_task_reward = task_reward
    if task_reward > 0:
      for ach, count in info.get('achievements', {}).items():
        if count > self._achievement_counts[ach]:
          decay = self._tester_task_saturation_decay ** self._achievement_counts[ach]
          scaled_task_reward *= decay
          self._achievement_counts[ach] = count

    context_keys = self._semantic_context_keys(info)

    semantic_fault_applied = int(info.get('semantic_fault_applied', 0))
    semantic_fault_type = str(info.get('semantic_fault_type', 'none'))
    post_fault_window = int(info.get('semantic_post_fault_window', 0))
    post_fault_switch = int(info.get('semantic_post_fault_switch', 0))
    post_fault_nonzero = int(info.get('semantic_post_fault_nonzero', 0))

    # --------------------------------------------------
    # 2. Validation Mode Context Sensing
    # --------------------------------------------------
    in_semantic_validation = bool(
        context_keys or semantic_fault_applied or post_fault_window or
        post_fault_switch or post_fault_nonzero or self._active_semantic_family is not None
    )

    # --------------------------------------------------
    # 3. Dynamic Bonus Scaling (Amplification)
    # --------------------------------------------------
    amp = self._tester_validation_amp if in_semantic_validation else 1.0
    bonus = 0.0
    repeat_penalty = 0.0

    # Only punish clearly unproductive idling repeats.
    should_penalize_repeat = self._same_action_run > self._tester_same_action_cap
    if self._tester_repeat_only_on_idle and task_reward > 0:
      should_penalize_repeat = False
    if self._tester_repeat_suspend_on_context and in_semantic_validation:
      should_penalize_repeat = False

    if should_penalize_repeat:
      over = float(self._same_action_run - self._tester_same_action_cap)
      repeat_penalty = min(self._tester_repeat_max_step, self._tester_repeat_penalty * over)

    for key in context_keys:
      state = self._semantic_proc_state[key]
      if not state['ctx']:
        ctx_reward = self._ctx_reward_for_key(key) * amp
        state['ctx'] = 1
        bonus += ctx_reward
        components[f'ctx:{key}'] = components.get(f'ctx:{key}', 0.0) + ctx_reward
        self._active_semantic_family = key

    if semantic_fault_applied and semantic_fault_type != 'none':
      key = semantic_fault_type
      self._active_semantic_family = key
      state = self._semantic_proc_state[key]
      if not state['anom']:
        anom_reward = self._anom_reward_for_type(key) * amp
        state['anom'] = 1
        bonus += anom_reward
        components[f'anom:{key}'] = components.get(f'anom:{key}', 0.0) + anom_reward
      elif not state['reproduce']:
        state['reproduce'] = 1
        bonus += self._tester_reproduce_reward * amp
        components[f'reproduce:{key}'] = components.get(f'reproduce:{key}', 0.0) + (self._tester_reproduce_reward * amp)

    active_key = self._active_semantic_family
    if active_key is not None:
      state = self._semantic_proc_state[active_key]
      if post_fault_window > 0 and state['anom'] and not state['followup']:
        state['followup'] = 1
        bonus += self._tester_followup_reward * amp
        components[f'followup:{active_key}'] = components.get(f'followup:{active_key}', 0.0) + (self._tester_followup_reward * amp)
      if post_fault_switch > 0 and state['anom'] and not state['compare']:
        state['compare'] = 1
        bonus += self._tester_compare_reward * amp
        components[f'compare:{active_key}'] = components.get(f'compare:{active_key}', 0.0) + (self._tester_compare_reward * amp)
      if post_fault_nonzero > 0 and state['anom'] and not state['confirm']:
        state['confirm'] = 1
        bonus += self._tester_confirm_reward * amp
        components[f'confirm:{active_key}'] = components.get(f'confirm:{active_key}', 0.0) + (self._tester_confirm_reward * amp)

    # Amplified dynamic reward cap
    current_cap = self._tester_reward_cap * (amp if in_semantic_validation else 1.0)
    if bonus > current_cap:
      bonus = current_cap

    # --------------------------------------------------
    # 4. Mode Switching (Task Suppression)
    # --------------------------------------------------
    if in_semantic_validation:
      final_reward = (scaled_task_reward * self._tester_task_suppression) + bonus - repeat_penalty
    else:
      final_reward = scaled_task_reward + bonus - repeat_penalty

    self._episode_task_reward += scaled_task_reward
    self._episode_tester_bonus += bonus
    self._episode_repeat_penalty += repeat_penalty

    info['task_reward_raw'] = float(task_reward)
    info['tester_bonus'] = float(bonus)
    info['tester_repeat_penalty'] = float(repeat_penalty)
    info['tester_reward'] = float(final_reward)
    info['tester_reward_components'] = components
    info['tester_repeat_suppressed'] = int(bool(self._tester_repeat_suspend_on_context and in_semantic_validation))
    info['tester_mode_active'] = int(in_semantic_validation)
    
    return float(final_reward), info

  def _reset_episode_aggregation(self):
    self._ep_fault_exists_episode = 0
    self._ep_fault_applied_any = 0
    self._ep_fault_manifest_count = 0
    self._ep_fault_trigger_context_count = 0
    self._ep_lowlevel_trigger_context_count = 0
    self._ep_semantic_trigger_context_count = 0
    self._ep_fault_manifest_prob_max = 0.0
    self._ep_semantic_fault_applied_any = 0
    self._ep_semantic_fault_count = 0
    self._ep_fault_family = 'none'
    self._ep_fault_type = 'none'
    self._ep_fault_trigger = 'none'
    self._ep_semantic_trigger_count = 0
    self._ep_semantic_first_trigger_step = -1
    self._ep_semantic_ctx_upgrade_collect_count = 0
    self._ep_semantic_ctx_retry_craft_count = 0
    self._ep_semantic_ctx_relocate_station_count = 0
    self._ep_semantic_ctx_station_reuse_count = 0
    self._ep_semantic_valid_progress_count = 0
    self._ep_semantic_station_reuse_count = 0
    self._ep_semantic_delayed_after_use_count = 0
    self._ep_semantic_post_fault_window_count = 0
    self._ep_semantic_post_fault_nonzero_count = 0
    self._ep_semantic_post_fault_switch_count = 0
    self._ep_rnd_intrinsic_raw_sum = 0.0
    self._ep_rnd_intrinsic_norm_sum = 0.0
    self._ep_rnd_loss_sum = 0.0
    self._ep_rnd_steps = 0
    self._reset_coverage_state()
    self._reset_tester_reward_state()

  def _update_episode_aggregation(
      self,
      fault_applied,
      fault_family,
      fault_type,
      fault_trigger,
      fault_exists_episode,
      fault_trigger_context,
      lowlevel_trigger_context,
      fault_manifest_prob,
      semantic_fault_applied,
      semantic_trigger_context,
      semantic_trigger_count,
      semantic_first_trigger_step,
      semantic_ctx_upgrade_collect,
      semantic_ctx_retry_craft,
      semantic_ctx_relocate_station,
      semantic_ctx_station_reuse,
      semantic_valid_progress_count,
      semantic_station_reuse_count,
      semantic_delayed_after_use_count,
      semantic_post_fault_window,
      semantic_post_fault_nonzero,
      semantic_post_fault_switch):
    self._ep_fault_exists_episode = max(
        self._ep_fault_exists_episode, int(fault_exists_episode))
    self._ep_fault_applied_any = max(self._ep_fault_applied_any, int(fault_applied))
    self._ep_fault_manifest_count += int(fault_applied)
    self._ep_fault_trigger_context_count += int(fault_trigger_context)
    self._ep_lowlevel_trigger_context_count += int(lowlevel_trigger_context)
    self._ep_semantic_trigger_context_count += int(semantic_trigger_context)
    self._ep_fault_manifest_prob_max = max(
        float(self._ep_fault_manifest_prob_max), float(fault_manifest_prob))
    self._ep_semantic_fault_applied_any = max(
        self._ep_semantic_fault_applied_any, int(semantic_fault_applied))
    self._ep_semantic_fault_count += int(semantic_fault_applied)

    if fault_family and fault_family != 'none':
      self._ep_fault_family = fault_family
    if fault_type and fault_type != 'none':
      self._ep_fault_type = fault_type
    if fault_trigger and fault_trigger != 'none':
      self._ep_fault_trigger = fault_trigger

    self._ep_semantic_trigger_count = max(
        self._ep_semantic_trigger_count, int(semantic_trigger_count))
    if int(semantic_first_trigger_step) >= 0:
      if self._ep_semantic_first_trigger_step < 0:
        self._ep_semantic_first_trigger_step = int(semantic_first_trigger_step)
      else:
        self._ep_semantic_first_trigger_step = min(
            self._ep_semantic_first_trigger_step, int(semantic_first_trigger_step))

    self._ep_semantic_ctx_upgrade_collect_count = max(
        self._ep_semantic_ctx_upgrade_collect_count, int(semantic_ctx_upgrade_collect))
    self._ep_semantic_ctx_retry_craft_count = max(
        self._ep_semantic_ctx_retry_craft_count, int(semantic_ctx_retry_craft))
    self._ep_semantic_ctx_relocate_station_count = max(
        self._ep_semantic_ctx_relocate_station_count, int(semantic_ctx_relocate_station))
    self._ep_semantic_ctx_station_reuse_count = max(
        self._ep_semantic_ctx_station_reuse_count, int(semantic_ctx_station_reuse))
    self._ep_semantic_valid_progress_count = max(
        self._ep_semantic_valid_progress_count, int(semantic_valid_progress_count))
    self._ep_semantic_station_reuse_count = max(
        self._ep_semantic_station_reuse_count, int(semantic_station_reuse_count))
    self._ep_semantic_delayed_after_use_count = max(
        self._ep_semantic_delayed_after_use_count, int(semantic_delayed_after_use_count))

    self._ep_semantic_post_fault_window_count += int(semantic_post_fault_window)
    self._ep_semantic_post_fault_nonzero_count += int(semantic_post_fault_nonzero)
    self._ep_semantic_post_fault_switch_count += int(semantic_post_fault_switch)

  def _resolve_episode_fault_identity(self, info=None):
    info = dict(info or {})
    fault_family = getattr(self, '_ep_fault_family', 'none')
    fault_type = getattr(self, '_ep_fault_type', 'none')
    fault_trigger = getattr(self, '_ep_fault_trigger', 'none')

    ctx_upgrade = int(getattr(self, '_ep_semantic_ctx_upgrade_collect_count', 0))
    ctx_retry = int(getattr(self, '_ep_semantic_ctx_retry_craft_count', 0))
    ctx_relocate = int(getattr(self, '_ep_semantic_ctx_relocate_station_count', 0))
    ctx_progress = int(getattr(self, '_ep_semantic_valid_progress_count', 0))
    ctx_station_reuse = int(getattr(self, '_ep_semantic_station_reuse_count', 0))
    ctx_delayed = int(getattr(self, '_ep_semantic_delayed_after_use_count', 0))
    semantic_fault_episode = int(info.get('semantic_fault_episode', 0)) or int(getattr(self, '_semantic_fault_episode_seen', 0))

    if fault_family == 'none' and semantic_fault_episode:
      fault_family = 'semantic_high_level'

    if fault_type == 'none' and fault_family == 'semantic_high_level':
      candidates = [
          ('tool_collect_desync_on_upgrade', ctx_upgrade),
          ('craft_result_missing_on_retry', ctx_retry),
          ('station_place_ghost_on_relocate', ctx_relocate),
          ('achievement_unlock_missing_after_valid_progress', ctx_progress),
          ('station_usable_flag_broken_after_relocate', ctx_station_reuse),
          ('delayed_inventory_desync_after_station_use', ctx_delayed),
      ]
      best_name, best_count = max(candidates, key=lambda x: x[1])
      if best_count > 0:
        fault_type = best_name
        trigger_map = {
            'tool_collect_desync_on_upgrade': 'first_valid_collect_after_pickaxe_upgrade',
            'upgrade_branch_inconsistent_collect_behavior': 'branch_specific_collect_after_pickaxe_upgrade',
            'craft_result_missing_on_retry': 'repeat_craft_same_recipe_within_episode',
            'station_place_ghost_on_relocate': 'second_or_later_table_or_furnace_placement',
            'achievement_unlock_missing_after_valid_progress': 'valid_progress_milestone_reached',
            'achievement_unlock_missing_after_reconfirm': 'valid_progress_requires_reconfirm',
            'station_usable_flag_broken_after_relocate': 'station_reuse_after_relocate',
            'station_second_use_inconsistent_after_placement': 'second_use_after_station_placement',
            'recipe_precondition_mischeck_on_retry': 'repeat_craft_same_recipe_within_episode',
            'delayed_inventory_desync_after_station_use': 'delayed_after_station_use',
            'progress_confirmation_requires_revisit': 'progress_requires_revisit',
        }
        if fault_trigger == 'none':
          fault_trigger = trigger_map.get(best_name, 'custom')

    return fault_family, fault_type, fault_trigger

  def _write_episode_summary(self, length, reward, info):
    if self._episode_summary_path is None:
      return

    info = dict(info or {})
    semantic_first_trigger_step = int(self._ep_semantic_first_trigger_step)
    semantic_first_trigger_global_step = (
        int(self._episode_start_global_step + semantic_first_trigger_step)
        if semantic_first_trigger_step >= 0 else -1)

    ctx_upgrade = int(self._ep_semantic_ctx_upgrade_collect_count)
    ctx_retry = int(self._ep_semantic_ctx_retry_craft_count)
    ctx_relocate = int(self._ep_semantic_ctx_relocate_station_count)
    fault_family, fault_type, fault_trigger = self._resolve_episode_fault_identity(info)

    summary = {
        'episode': int(self._episode),
        'episode_length': int(length),
        'episode_reward': round(float(reward), 6),
        'start_global_step': int(self._episode_start_global_step),
        'end_global_step': int(self._global_step),
        'fault_episode': int(info.get('fault_episode', int(self._fault_episode or self._semantic_fault_episode_seen))),
        'fault_exists_episode': int(info.get(
            'fault_exists_episode',
            int(self._fault_episode or self._semantic_fault_episode_seen))),
        'fault_trigger_context_reach': int(self._ep_fault_trigger_context_count > 0),
        'fault_trigger_context_count': int(self._ep_fault_trigger_context_count),
        'lowlevel_trigger_context_count': int(self._ep_lowlevel_trigger_context_count),
        'semantic_trigger_context_count': int(self._ep_semantic_trigger_context_count),
        'fault_applied_any': int(self._ep_fault_applied_any),
        'fault_manifested_any': int(self._ep_fault_applied_any),
        'fault_manifest_count': int(self._ep_fault_manifest_count),
        'fault_manifest_prob_max': round(float(self._ep_fault_manifest_prob_max), 6),
        'fault_profile': self._fault_profile,
        'fault_profile_requested': self._fault_profile_requested,
        'fault_stochastic_manifest': int(self._fault_stochastic_manifest),
        'fault_frequency_tier': self._fault_frequency_tier,
        'fault_count': int(self._fault_count),
        'fault_rate': float(self._fault_count / max(1, length)),
        'fault_family': fault_family,
        'fault_type': fault_type,
        'fault_trigger': fault_trigger,
        'semantic_fault_episode': int(info.get('semantic_fault_episode', 0) or self._semantic_fault_episode_seen),
        'semantic_fault_applied_any': int(self._ep_semantic_fault_applied_any),
        'semantic_fault_count': int(self._ep_semantic_fault_count),
        'semantic_fault_type': fault_type if fault_family == 'semantic_high_level' else info.get('semantic_fault_type', 'none'),
        'semantic_trigger_reach': int(self._ep_semantic_trigger_count > 0),
        'semantic_trigger_count': int(self._ep_semantic_trigger_count),
        'semantic_first_trigger_step': semantic_first_trigger_step,
        'semantic_first_trigger_global_step': semantic_first_trigger_global_step,
        'semantic_ctx_upgrade_collect_count': ctx_upgrade,
        'semantic_ctx_retry_craft_count': ctx_retry,
        'semantic_ctx_relocate_station_count': ctx_relocate,
        'semantic_valid_progress_count': int(self._ep_semantic_valid_progress_count),
        'semantic_station_reuse_count': int(self._ep_semantic_station_reuse_count),
        'semantic_delayed_after_use_count': int(self._ep_semantic_delayed_after_use_count),
        'semantic_bug_family_coverage': int(
            (ctx_upgrade > 0) + (ctx_retry > 0) + (ctx_relocate > 0) +
            (int(self._ep_semantic_valid_progress_count) > 0) +
            (int(self._ep_semantic_station_reuse_count) > 0) +
            (int(self._ep_semantic_delayed_after_use_count) > 0)
        ),
        'episode_task_reward': round(float(self._episode_task_reward), 6),
        'episode_tester_bonus': round(float(self._episode_tester_bonus), 6),
        'episode_repeat_penalty': round(float(self._episode_repeat_penalty), 6),
        'episode_rnd_intrinsic_raw_mean': round(float(self._ep_rnd_intrinsic_raw_sum / max(1, self._ep_rnd_steps)), 6),
        'episode_rnd_intrinsic_mean': round(float(self._ep_rnd_intrinsic_norm_sum / max(1, self._ep_rnd_steps)), 6),
        'episode_rnd_loss_mean': round(float(self._ep_rnd_loss_sum / max(1, self._ep_rnd_steps)), 6),
        'rnd_update_enabled': int(self._rnd_update),
        'episode_unique_contexts': int(len(self._ctx_visit_steps)),
        'episode_unique_tiles_visited': int(len(self._visited_tiles)),
        'episode_tile_coverage_ratio': round(float(self._tile_coverage_ratio()), 6),
        'episode_unique_target_cells': int(len(self._target_cells)),
        'episode_unique_target_materials': int(len(self._target_materials)),
        'episode_unique_cells_modified': int(len(self._modified_cells)),
        'episode_world_edit_count': int(self._world_edit_count),
        'episode_world_edit_ratio': round(float(self._world_edit_ratio()), 6),
        'episode_unique_materials_modified': int(len(self._modified_material_types)),
        'semantic_post_fault_window_count': int(self._ep_semantic_post_fault_window_count),
        'semantic_post_fault_nonzero_count': int(self._ep_semantic_post_fault_nonzero_count),
        'semantic_post_fault_switch_count': int(self._ep_semantic_post_fault_switch_count),
    }

    with self._episode_summary_path.open('a', encoding='utf-8') as f:
      f.write(json.dumps(summary) + '\n')
      f.flush()

  def _current_fault_manifest_prob(self):
    if self._fault_spec is None:
      return 0.0
    if self._fault_stochastic_manifest:
      return float(self._fault_spec.get('severity', 0.0))
    return 1.0

  def _lowlevel_fault_trigger_context(self, requested_action, reward=None, done=False):
    if self._fault_spec is None:
      return 0

    family = self._fault_spec.get('family', 'none')
    subtype = self._fault_spec.get('type', 'none')

    if family == 'legacy_action_exec':
      return int(self._episode >= self._fault_start_episode)

    if family in ('action_exec', 'context_exec'):
      if self._sticky_remaining > 0 and self._sticky_action is not None:
        return int(subtype == 'sticky_after_repeat_switch')
      checks = {
          'remap_after_success_switch': (
              self._success_window_active() and self._switched_action(requested_action)),
          'delay_after_success': (
              self._success_window_active() and self._is_nonzero(requested_action)),
          'sticky_after_repeat_switch': self._repeat_then_switch(requested_action),
          'ignore_nonzero_after_reward': (
              self._success_window_active() and self._is_nonzero(requested_action)),
          'ignore_nonzero_after_two_rewards': (
              self._recent_positive_count(4) >= 2 and self._is_nonzero(requested_action)),
          'ignore_switch_late_episode': (
              self._late_episode() and self._switched_action(requested_action)),
          'remap_after_repeat_switch': self._repeat_then_switch(requested_action),
          'delay_after_late_episode_switch': (
              self._late_episode() and self._switched_action(requested_action)),
          'revisit_action_ignore': (
              self._recent_revisit_active() and self._is_nonzero(requested_action)),
          'revisit_action_delay': (
              self._recent_revisit_active() and self._switched_action(requested_action)),
          'delayed_switch_failure': self._delayed_switch_after_success(requested_action),
      }
      return int(bool(checks.get(subtype, False)))

    if family == 'reward_timing':
      if reward is None or float(reward) <= 0.0:
        return 0
      checks = {
          'reward_delay_on_positive': True,
          'reward_zero_on_positive': True,
          'reward_scale_half_on_positive_switch': self._switched_action(requested_action),
          'reward_zero_after_repeat_switch': self._repeat_then_switch(requested_action),
          'reward_delay_after_two_rewards': self._recent_positive_count(4) >= 2,
      }
      return int(bool(checks.get(subtype, False)))

    if family == 'termination_logic':
      if bool(done):
        return 0
      checks = {
          'early_done_after_success_switch': (
              self._success_window_active() and self._switched_action(requested_action)),
          'early_done_after_repeat_switch': self._repeat_then_switch(requested_action),
      }
      return int(bool(checks.get(subtype, False)))

    return 0

  def _can_fire_now(self):
    if self._fault_spec is None:
      return False
    if self._episode < self._fault_start_episode:
      return False
    if self._fault_cooldown > 0:
      return False

    if self._fault_stochastic_manifest:
      sev = float(self._fault_spec.get('severity', 0.0))
      return self._rng.random() < sev
    return True

  def _set_cooldown(self):
    self._fault_cooldown = int(self._fault_cooldown_steps)

  def _update_scheduler_after_episode(self):
    if not self._adaptive_scheduler:
      return

    ep_fault_rate = float(self._fault_count / max(1, self._length))
    self._recent_fault_apply_rates.append(ep_fault_rate)

    if (self._fault_episode == 0) and (self._semantic_fault_episode_seen == 0):
      self._recent_clean_scores.append(float(self._reward))

    clean_ok = True
    if self._sched_baseline_score > 0 and len(self._recent_clean_scores) >= max(5, self._sched_window // 3):
      clean_mean = float(np.mean(self._recent_clean_scores))
      clean_ok = clean_mean >= self._sched_floor_ratio * self._sched_baseline_score

    mean_fault_apply = (
        float(np.mean(self._recent_fault_apply_rates))
        if len(self._recent_fault_apply_rates) > 0 else 0.0
    )

    # adaptive fault episode probability
    if clean_ok:
      self._fault_episode_prob = min(
          self._fault_episode_prob_max,
          self._fault_episode_prob + self._fault_episode_prob_step)
    else:
      self._fault_episode_prob = max(
          self._fault_episode_prob_min,
          self._fault_episode_prob - self._fault_episode_prob_step)

    # adaptive cooldown
    if mean_fault_apply < self._target_fault_apply_rate:
      self._fault_cooldown_steps = max(
          self._fault_cooldown_min,
          self._fault_cooldown_steps - self._fault_cooldown_step)
    elif mean_fault_apply > 1.5 * self._target_fault_apply_rate:
      self._fault_cooldown_steps = min(
          self._fault_cooldown_max,
          self._fault_cooldown_steps + self._fault_cooldown_step)

  # --------------------------------------------------
  # Fault application
  # --------------------------------------------------
  def _apply_action_fault(self, requested_action):
    env_action = int(requested_action)
    applied = 0
    fault_type = 'none'

    if self._sticky_remaining > 0 and self._sticky_action is not None:
      env_action = int(self._sticky_action)
      self._sticky_remaining -= 1
      applied = 1
      fault_type = 'sticky_after_repeat_switch'
      self._set_cooldown()
      return env_action, applied, fault_type

    if self._fault_spec is None:
      return env_action, applied, fault_type

    family = self._fault_spec['family']
    subtype = self._fault_spec['type']

    if family == 'legacy_action_exec':
      if self._can_fire_now():
        fallback = int(self._fault_spec.get('fallback_action', self._fallback_action))
        env_action = fallback
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if family not in ('action_exec', 'context_exec'):
      return env_action, applied, fault_type

    if subtype == 'remap_after_success_switch':
      if self._success_window_active() and self._switched_action(requested_action) and self._can_fire_now():
        env_action = self._random_other_action(requested_action)
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'delay_after_success':
      if self._success_window_active() and self._is_nonzero(requested_action) and self._can_fire_now():
        env_action = int(self._last_requested_action)
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'sticky_after_repeat_switch':
      if self._repeat_then_switch(requested_action) and self._can_fire_now():
        self._sticky_action = int(requested_action)
        self._sticky_remaining = 1
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'ignore_nonzero_after_reward':
      if self._success_window_active() and self._is_nonzero(requested_action) and self._can_fire_now():
        env_action = 0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'ignore_nonzero_after_two_rewards':
      if self._recent_positive_count(4) >= 2 and self._is_nonzero(requested_action) and self._can_fire_now():
        env_action = 0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'ignore_switch_late_episode':
      if self._late_episode() and self._switched_action(requested_action) and self._can_fire_now():
        env_action = 0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'remap_after_repeat_switch':
      if self._repeat_then_switch(requested_action) and self._can_fire_now():
        env_action = self._random_other_action(requested_action)
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'delay_after_late_episode_switch':
      if self._late_episode() and self._switched_action(requested_action) and self._can_fire_now():
        env_action = int(self._last_requested_action)
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'revisit_action_ignore':
      if self._recent_revisit_active() and self._is_nonzero(requested_action) and self._can_fire_now():
        env_action = 0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'revisit_action_delay':
      if self._recent_revisit_active() and self._switched_action(requested_action) and self._can_fire_now():
        env_action = int(self._last_requested_action)
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'delayed_switch_failure':
      if self._delayed_switch_after_success(requested_action) and self._can_fire_now():
        env_action = self._random_other_action(requested_action)
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    return env_action, applied, fault_type

  def _apply_reward_fault(self, reward, requested_action):
    reward = float(reward)
    applied = 0
    fault_type = 'none'

    if self._pending_reward != 0.0:
      reward += float(self._pending_reward)
      self._pending_reward = 0.0

    if self._fault_spec is None:
      return reward, applied, fault_type

    family = self._fault_spec['family']
    subtype = self._fault_spec['type']

    if family != 'reward_timing':
      return reward, applied, fault_type

    if reward <= 0.0:
      return reward, applied, fault_type

    if subtype == 'reward_delay_on_positive':
      if self._can_fire_now():
        self._pending_reward += float(reward)
        reward = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return reward, applied, fault_type

    if subtype == 'reward_scale_half_on_positive_switch':
      if self._switched_action(requested_action) and self._can_fire_now():
        reward = 0.5 * reward
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return reward, applied, fault_type

    if subtype == 'reward_zero_on_positive':
      if self._can_fire_now():
        reward = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return reward, applied, fault_type

    if subtype == 'reward_zero_after_repeat_switch':
      if self._repeat_then_switch(requested_action) and self._can_fire_now():
        reward = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return reward, applied, fault_type

    if subtype == 'reward_delay_after_two_rewards':
      if self._recent_positive_count(4) >= 2 and self._can_fire_now():
        self._pending_reward += float(reward)
        reward = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return reward, applied, fault_type

    return reward, applied, fault_type

  def _apply_termination_fault(self, requested_action, done, info):
    done = bool(done)
    info = dict(info)
    applied = 0
    fault_type = 'none'

    if self._fault_spec is None:
      return done, info, applied, fault_type

    family = self._fault_spec['family']
    subtype = self._fault_spec['type']

    if family != 'termination_logic':
      return done, info, applied, fault_type

    if done:
      return done, info, applied, fault_type

    if subtype == 'early_done_after_success_switch':
      if self._success_window_active() and self._switched_action(requested_action) and self._can_fire_now():
        done = True
        info['discount'] = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return done, info, applied, fault_type

    if subtype == 'early_done_after_repeat_switch':
      if self._repeat_then_switch(requested_action) and self._can_fire_now():
        done = True
        info['discount'] = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return done, info, applied, fault_type

    return done, info, applied, fault_type

  # --------------------------------------------------
  # Spaces
  # --------------------------------------------------
  @property
  def obs_space(self):
    spaces = {
        'image': elements.Space(np.uint8, self._env.observation_space.shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/reward': elements.Space(np.float32),
        'log/fault_applied': elements.Space(np.int32),
        'log/fault_episode': elements.Space(np.int32),
        'log/fault_exists_episode': elements.Space(np.int32),
        'log/fault_trigger_context': elements.Space(np.int32),
        'log/lowlevel_trigger_context': elements.Space(np.int32),
        'log/fault_manifested': elements.Space(np.int32),
        'log/fault_manifest_prob': elements.Space(np.float32),
        'log/fault_profile_id': elements.Space(np.int32),
        'log/fault_frequency_tier_id': elements.Space(np.int32),
        'log/fault_family_id': elements.Space(np.int32),
        'log/fault_type_id': elements.Space(np.int32),
        'log/fault_count_cumulative': elements.Space(np.int32),
        'log/semantic_fault_applied': elements.Space(np.int32),
        'log/semantic_fault_episode': elements.Space(np.int32),
        'log/semantic_trigger_context': elements.Space(np.int32),
        'log/semantic_trigger_count': elements.Space(np.int32),
        'log/semantic_first_trigger_step': elements.Space(np.int32),
        'log/semantic_ctx_upgrade_collect': elements.Space(np.int32),
        'log/semantic_ctx_retry_craft': elements.Space(np.int32),
        'log/semantic_ctx_relocate_station': elements.Space(np.int32),
        'log/semantic_ctx_valid_progress': elements.Space(np.int32),
        'log/semantic_ctx_station_reuse': elements.Space(np.int32),
        'log/semantic_ctx_delayed_after_use': elements.Space(np.int32),
        'log/semantic_post_fault_window': elements.Space(np.int32),
        'log/semantic_post_fault_nonzero': elements.Space(np.int32),
        'log/semantic_post_fault_switch': elements.Space(np.int32),
        'log/semantic_upgrade_collect_count': elements.Space(np.int32),
        'log/semantic_retry_craft_count': elements.Space(np.int32),
        'log/semantic_relocate_station_count': elements.Space(np.int32),
        'log/semantic_valid_progress_count': elements.Space(np.int32),
        'log/semantic_station_reuse_count': elements.Space(np.int32),
        'log/semantic_delayed_after_use_count': elements.Space(np.int32),
        'log/task_reward_raw': elements.Space(np.float32),
        'log/tester_bonus': elements.Space(np.float32),
        'log/tester_repeat_penalty': elements.Space(np.float32),
        'log/rnd_intrinsic_reward': elements.Space(np.float32),
        'log/rnd_intrinsic_reward_raw': elements.Space(np.float32),
        'log/rnd_loss': elements.Space(np.float32),
        'log/rnd_update_enabled': elements.Space(np.int32),
        'log/revisit_context': elements.Space(np.int32),
        'log/delayed_switch_context': elements.Space(np.int32),
        'log/unique_contexts': elements.Space(np.int32),
        'log/unique_tiles_visited': elements.Space(np.int32),
        'log/tile_coverage_ratio': elements.Space(np.float32),
        'log/unique_target_cells': elements.Space(np.int32),
        'log/unique_target_materials': elements.Space(np.int32),
        'log/unique_cells_modified': elements.Space(np.int32),
        'log/unique_materials_modified': elements.Space(np.int32),
        'log/world_edit_count': elements.Space(np.int32),
        'log/world_step_edit_count': elements.Space(np.int32),
        'log/world_edit_ratio': elements.Space(np.float32),
        'log/global_step': elements.Space(np.int32),
    }
    if self._logs:
      spaces.update({
          f'log/achievement_{k}': elements.Space(np.int32)
          for k in self._achievements
      })
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': elements.Space(bool),
    }

  # --------------------------------------------------
  # Main step
  # --------------------------------------------------
  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0.0
      self._fault_count = 0
      self._semantic_fault_episode_seen = 0
      self._done = False

      self._last_reward = 0.0
      self._last_requested_action = 0
      self._prev_executed_action = 0
      self._pending_reward = 0.0
      self._sticky_action = None
      self._sticky_remaining = 0
      self._after_positive_window = 0
      self._fault_cooldown = 0

      self._requested_hist = deque([0, 0], maxlen=2)
      self._executed_hist = deque([0, 0], maxlen=2)
      self._reward_hist = deque([0.0, 0.0, 0.0, 0.0], maxlen=4)

      self._episode_step = 0
      self._last_positive_reward_step = -10**9
      self._ctx_visit_steps = defaultdict(lambda: deque(maxlen=4))
      self._last_ctx_key = None

      self._episode_start_global_step = int(self._global_step)
      self._reset_episode_aggregation()

      self._sample_fault_spec()
      if self._fault_verbose and self._fault_spec is not None:
        print(f'[FaultSpec] episode={self._episode} profile={self._fault_profile} '
              f'p_fault={self._fault_episode_prob:.3f} cooldown={self._fault_cooldown_steps} '
              f'spec={self._fault_spec}')

      image = self._env.reset()
      self._record_context(image)
      self._record_player_tile_from_info({'player_pos': getattr(self._env._player, 'pos', None)})
      self._record_current_target_coverage()

      if self._record_gifs:
        self._episode_frames = []
        self._record_hud_frame(image, 0, {})

      return self._obs(
          image=image,
          reward=0.0,
          info={
              'fault_episode': int(self._fault_episode),
              'fault_exists_episode': int(self._fault_episode),
              'fault_trigger_context': 0,
              'lowlevel_trigger_context': 0,
              'fault_manifested': 0,
              'fault_manifest_prob': float(self._current_fault_manifest_prob()),
              'fault_profile': self._fault_profile,
              'fault_frequency_tier': self._fault_frequency_tier,
              'fault_applied': np.int32(0),
              'semantic_fault_episode': np.int32(0),
              'semantic_fault_applied': np.int32(0),
              'fault_family': 'none',
              'fault_type': 'none',
              'fault_trigger': 'none',
              'fault_severity': 0.0,
              'task_reward_raw': 0.0,
              'tester_bonus': 0.0,
              'tester_repeat_penalty': 0.0,
              'rnd_intrinsic_reward': 0.0,
              'rnd_intrinsic_reward_raw': 0.0,
              'rnd_loss': 0.0,
              'rnd_update_enabled': int(self._rnd_update),
              'unique_contexts': 1,
              'unique_tiles_visited': int(len(self._visited_tiles)),
              'tile_coverage_ratio': float(self._tile_coverage_ratio()),
              'unique_target_cells': int(len(self._target_cells)),
              'unique_target_materials': int(len(self._target_materials)),
              'unique_cells_modified': 0,
              'unique_materials_modified': 0,
              'world_edit_count': 0,
              'world_step_edit_count': 0,
              'world_edit_ratio': 0.0,
          },
          is_first=True,
          is_last=False,
          is_terminal=False,
      )

    requested_action = int(action['action'])
    lowlevel_trigger_pre = self._lowlevel_fault_trigger_context(requested_action)

    if self._coverage_enabled:
      self._record_current_target_coverage()
      mat_before = self._snapshot_world_material_map()
    else:
      mat_before = None

    # action fault
    env_action, action_fault_applied, action_fault_type = self._apply_action_fault(
        requested_action)

    # env step
    image, raw_reward, self._done, info = self._env.step(env_action)
    env_done_before_termination = bool(self._done)

    info = dict(info)
    self._record_player_tile_from_info(info)
    self._update_world_coverage_from_material_map(mat_before)
    info['unique_contexts'] = int(len(self._ctx_visit_steps))
    info['unique_tiles_visited'] = int(len(self._visited_tiles))
    info['tile_coverage_ratio'] = float(self._tile_coverage_ratio())
    info['unique_target_cells'] = int(len(self._target_cells))
    info['unique_target_materials'] = int(len(self._target_materials))
    info['unique_cells_modified'] = int(len(self._modified_cells))
    info['unique_materials_modified'] = int(len(self._modified_material_types))
    info['world_edit_count'] = int(self._world_edit_count)
    info['world_step_edit_count'] = int(self._last_world_step_edit_count)
    info['world_edit_ratio'] = float(self._world_edit_ratio())
    
    # --------------------------------------------------
    # [A9] RND Intrinsic Reward Calculation
    # --------------------------------------------------
    intrinsic_r_raw = 0.0
    intrinsic_r = 0.0
    rnd_loss = 0.0
    if self._use_rnd:
        intrinsic_r_raw, rnd_loss = self._rnd_mod.compute_intrinsic_reward(
            image, update=self._rnd_update)
        intrinsic_r = self._normalize_rnd_reward(
            intrinsic_r_raw, update_stats=self._rnd_update)
        info['rnd_intrinsic_reward_raw'] = intrinsic_r_raw
        info['rnd_intrinsic_reward'] = intrinsic_r
        info['rnd_loss'] = rnd_loss
        info['rnd_update_enabled'] = int(self._rnd_update)
        self._ep_rnd_intrinsic_raw_sum += float(intrinsic_r_raw)
        self._ep_rnd_intrinsic_norm_sum += float(intrinsic_r)
        self._ep_rnd_loss_sum += float(rnd_loss)
        self._ep_rnd_steps += 1
    else:
        info['rnd_intrinsic_reward_raw'] = 0.0
        info['rnd_intrinsic_reward'] = 0.0
        info['rnd_loss'] = 0.0
        info['rnd_update_enabled'] = 0

    # semantic high-level fault info from crafter.Env
    semantic_fault_episode = int(info.get('semantic_fault_episode', 0))
    semantic_fault_applied = int(info.get('semantic_fault_applied', 0))
    semantic_fault_family = info.get(
        'semantic_fault_family',
        'semantic_high_level' if (semantic_fault_episode or semantic_fault_applied) else 'none')
    semantic_fault_type = info.get('semantic_fault_type', 'none')
    semantic_fault_trigger = info.get('semantic_fault_trigger', 'none')
    semantic_fault_severity = float(info.get(
        'semantic_fault_severity',
        1.0 if semantic_fault_applied else 0.0))
    semantic_trigger_context = int(info.get('semantic_trigger_context', 0))
    semantic_trigger_type = info.get('semantic_trigger_type', 'none')
    semantic_trigger_count = int(info.get('semantic_trigger_count', 0))
    semantic_first_trigger_step = int(info.get('semantic_first_trigger_step', -1))
    semantic_ctx_upgrade_collect = int(info.get('semantic_ctx_upgrade_collect', 0))
    semantic_ctx_retry_craft = int(info.get('semantic_ctx_retry_craft', 0))
    semantic_ctx_relocate_station = int(info.get('semantic_ctx_relocate_station', 0))
    semantic_ctx_valid_progress = int(info.get('semantic_ctx_valid_progress', 0))
    semantic_ctx_station_reuse = int(info.get('semantic_ctx_station_reuse', 0))
    semantic_ctx_delayed_after_use = int(info.get('semantic_ctx_delayed_after_use', 0))
    semantic_post_fault_window = int(info.get('semantic_post_fault_window', 0))
    semantic_post_fault_nonzero = int(info.get('semantic_post_fault_nonzero', 0))
    semantic_post_fault_switch = int(info.get('semantic_post_fault_switch', 0))
    semantic_upgrade_collect_count = int(info.get('semantic_upgrade_collect_count', 0))
    semantic_retry_craft_count = int(info.get('semantic_retry_craft_count', 0))
    semantic_relocate_station_count = int(info.get('semantic_relocate_station_count', 0))
    semantic_valid_progress_count = int(info.get('semantic_valid_progress_count', 0))
    semantic_station_reuse_count = int(info.get('semantic_station_reuse_count', 0))
    semantic_delayed_after_use_count = int(info.get('semantic_delayed_after_use_count', 0))
    self._semantic_fault_episode_seen = max(
        int(self._semantic_fault_episode_seen), semantic_fault_episode, semantic_fault_applied)

    # reward fault
    reward, reward_fault_applied, reward_fault_type = self._apply_reward_fault(
        raw_reward, requested_action)

    # termination fault
    self._done, info, term_fault_applied, term_fault_type = self._apply_termination_fault(
        requested_action, self._done, info)

    # tester-oriented semantic reward shaping
    task_reward_for_agent = float(reward)
    if self._tester_reward_enabled:
      reward, info = self._tester_reward_from_semantics(
          task_reward_for_agent, requested_action, info)
    else:
      info['task_reward_raw'] = float(task_reward_for_agent)
      info['tester_bonus'] = 0.0
      info['tester_repeat_penalty'] = 0.0
      info['tester_reward'] = float(task_reward_for_agent)
      self._episode_task_reward += float(task_reward_for_agent)

    # --------------------------------------------------
    # [A9] Apply RND Reward to Final Reward
    # --------------------------------------------------
    if self._use_rnd:
        reward = reward + (self._rnd_alpha * intrinsic_r)

    fault_applied = int(
        action_fault_applied or reward_fault_applied or term_fault_applied or semantic_fault_applied)
    lowlevel_trigger_context = int(
        lowlevel_trigger_pre or
        self._lowlevel_fault_trigger_context(
            requested_action, reward=raw_reward, done=env_done_before_termination))
    fault_trigger_context = int(lowlevel_trigger_context or semantic_trigger_context)
    fault_exists_episode = int(
        self._fault_episode or semantic_fault_episode or self._semantic_fault_episode_seen)
    lowlevel_manifest_prob = (
        self._current_fault_manifest_prob() if int(self._fault_episode) else 0.0)
    semantic_manifest_prob = (
        float(info.get('semantic_fault_manifest_prob', self._semantic_manifest_prob))
        if int(semantic_fault_episode) else 0.0)
    fault_manifest_prob = max(float(lowlevel_manifest_prob), float(semantic_manifest_prob))

    if fault_applied:
      self._fault_count += 1
      subtype = None
      if semantic_fault_applied:
        subtype = semantic_fault_type
      elif action_fault_applied:
        subtype = action_fault_type
      elif reward_fault_applied:
        subtype = reward_fault_type
      elif term_fault_applied:
        subtype = term_fault_type
      if subtype is not None and subtype != 'none':
        self._subtype_apply_counts[subtype] += 1

    if semantic_fault_applied:
      fault_family = semantic_fault_family
      fault_type = semantic_fault_type
      fault_trigger = semantic_fault_trigger
      fault_severity = semantic_fault_severity
    elif action_fault_applied:
      fault_family = self._fault_spec['family']
      fault_type = action_fault_type
      fault_trigger = self._fault_spec.get('trigger', 'none') if self._fault_spec is not None else 'none'
      fault_severity = float(self._fault_spec.get('severity', 0.0)) if self._fault_spec is not None else 0.0
    elif reward_fault_applied:
      fault_family = self._fault_spec['family']
      fault_type = reward_fault_type
      fault_trigger = self._fault_spec.get('trigger', 'none') if self._fault_spec is not None else 'none'
      fault_severity = float(self._fault_spec.get('severity', 0.0)) if self._fault_spec is not None else 0.0
    elif term_fault_applied:
      fault_family = self._fault_spec['family']
      fault_type = term_fault_type
      fault_trigger = self._fault_spec.get('trigger', 'none') if self._fault_spec is not None else 'none'
      fault_severity = float(self._fault_spec.get('severity', 0.0)) if self._fault_spec is not None else 0.0
    else:
      if semantic_fault_episode:
        fault_family = semantic_fault_family
        fault_type = 'none'
        fault_trigger = semantic_fault_trigger
        fault_severity = semantic_fault_severity
      else:
        fault_family = self._fault_spec['family'] if self._fault_spec is not None else 'none'
        fault_type = 'none'
        fault_trigger = self._fault_spec.get('trigger', 'none') if self._fault_spec is not None else 'none'
        fault_severity = float(self._fault_spec.get('severity', 0.0)) if self._fault_spec is not None else 0.0

    info['reward'] = float(reward)
    info['fault_applied'] = int(fault_applied)
    info['fault_manifested'] = int(fault_applied)
    info['fault_trigger_context'] = int(fault_trigger_context)
    info['lowlevel_trigger_context'] = int(lowlevel_trigger_context)
    info['fault_exists_episode'] = int(fault_exists_episode)
    info['fault_manifest_prob'] = float(fault_manifest_prob)
    info['fault_profile'] = self._fault_profile
    info['fault_profile_requested'] = self._fault_profile_requested
    info['fault_stochastic_manifest'] = int(self._fault_stochastic_manifest)
    info['fault_frequency_tier'] = self._fault_frequency_tier
    info['semantic_fault_manifest_prob'] = float(semantic_manifest_prob)
    episode_fault_flag = int(fault_exists_episode)
    info['fault_episode'] = episode_fault_flag
    info['fault_family'] = fault_family
    info['fault_type'] = fault_type
    info['fault_trigger'] = fault_trigger
    info['fault_severity'] = float(fault_severity)
    info['semantic_fault_episode'] = int(semantic_fault_episode)
    info['semantic_fault_applied'] = int(semantic_fault_applied)
    info['semantic_fault_family'] = semantic_fault_family
    info['semantic_fault_type'] = semantic_fault_type
    info['semantic_fault_trigger'] = semantic_fault_trigger
    info['semantic_trigger_context'] = int(semantic_trigger_context)
    info['semantic_trigger_type'] = semantic_trigger_type
    info['semantic_trigger_count'] = int(semantic_trigger_count)
    info['semantic_first_trigger_step'] = int(semantic_first_trigger_step)
    info['semantic_ctx_upgrade_collect'] = int(semantic_ctx_upgrade_collect)
    info['semantic_ctx_retry_craft'] = int(semantic_ctx_retry_craft)
    info['semantic_ctx_relocate_station'] = int(semantic_ctx_relocate_station)
    info['semantic_ctx_valid_progress'] = int(semantic_ctx_valid_progress)
    info['semantic_ctx_station_reuse'] = int(semantic_ctx_station_reuse)
    info['semantic_ctx_delayed_after_use'] = int(semantic_ctx_delayed_after_use)
    info['semantic_post_fault_window'] = int(semantic_post_fault_window)
    info['semantic_post_fault_nonzero'] = int(semantic_post_fault_nonzero)
    info['semantic_post_fault_switch'] = int(semantic_post_fault_switch)
    info['semantic_upgrade_collect_count'] = int(semantic_upgrade_collect_count)
    info['semantic_retry_craft_count'] = int(semantic_retry_craft_count)
    info['semantic_relocate_station_count'] = int(semantic_relocate_station_count)
    info['semantic_valid_progress_count'] = int(semantic_valid_progress_count)
    info['semantic_station_reuse_count'] = int(semantic_station_reuse_count)
    info['semantic_delayed_after_use_count'] = int(semantic_delayed_after_use_count)
    info['requested_action'] = int(requested_action)
    info['executed_action'] = int(env_action)

    self._reward += reward
    self._length += 1
    self._global_step += 1

    self._update_episode_aggregation(
        fault_applied=fault_applied,
        fault_family=fault_family,
        fault_type=fault_type,
        fault_trigger=fault_trigger,
        fault_exists_episode=fault_exists_episode,
        fault_trigger_context=fault_trigger_context,
        lowlevel_trigger_context=lowlevel_trigger_context,
        fault_manifest_prob=fault_manifest_prob,
        semantic_fault_applied=semantic_fault_applied,
        semantic_trigger_context=semantic_trigger_context,
        semantic_trigger_count=semantic_trigger_count,
        semantic_first_trigger_step=semantic_first_trigger_step,
        semantic_ctx_upgrade_collect=semantic_upgrade_collect_count,
        semantic_ctx_retry_craft=semantic_retry_craft_count,
        semantic_ctx_relocate_station=semantic_relocate_station_count,
        semantic_ctx_station_reuse=semantic_ctx_station_reuse,
        semantic_valid_progress_count=semantic_valid_progress_count,
        semantic_station_reuse_count=semantic_station_reuse_count,
        semantic_delayed_after_use_count=semantic_delayed_after_use_count,
        semantic_post_fault_window=semantic_post_fault_window,
        semantic_post_fault_nonzero=semantic_post_fault_nonzero,
        semantic_post_fault_switch=semantic_post_fault_switch)

    if self._trace_path is not None:
      record = {
          'episode': int(self._episode),
          'episode_step': int(self._length),
          'global_step': int(self._global_step),
          'reward': float(reward),
          'fault_episode': int(episode_fault_flag),
          'fault_exists_episode': int(fault_exists_episode),
          'fault_trigger_context': int(fault_trigger_context),
          'lowlevel_trigger_context': int(lowlevel_trigger_context),
          'fault_manifested': int(fault_applied),
          'fault_manifest_prob': float(fault_manifest_prob),
          'fault_profile': self._fault_profile,
          'fault_profile_requested': self._fault_profile_requested,
          'fault_stochastic_manifest': int(self._fault_stochastic_manifest),
          'fault_frequency_tier': self._fault_frequency_tier,
          'semantic_fault_episode': int(self._semantic_fault_episode_seen or semantic_fault_episode),
          'fault_applied': int(fault_applied),
          'fault_family': fault_family,
          'fault_type': fault_type,
          'semantic_fault_type': semantic_fault_type,
          'task_reward_raw': float(info.get('task_reward_raw', 0.0)),
          'tester_bonus': float(info.get('tester_bonus', 0.0)),
          'tester_repeat_penalty': float(info.get('tester_repeat_penalty', 0.0)),
          'rnd_intrinsic_reward_raw': float(info.get('rnd_intrinsic_reward_raw', 0.0)),
          'rnd_intrinsic_reward': float(info.get('rnd_intrinsic_reward', 0.0)),
          'rnd_loss': float(info.get('rnd_loss', 0.0)),
          'rnd_update_enabled': int(info.get('rnd_update_enabled', 0)),
          'fault_trigger': info['fault_trigger'],
          'fault_severity': float(info['fault_severity']),
          'requested_action': int(requested_action),
          'executed_action': int(env_action),
          'semantic_trigger_count': int(semantic_trigger_count),
          'semantic_first_trigger_step': int(semantic_first_trigger_step),
          'semantic_ctx_upgrade_collect': int(semantic_ctx_upgrade_collect),
          'semantic_ctx_retry_craft': int(semantic_ctx_retry_craft),
          'semantic_ctx_relocate_station': int(semantic_ctx_relocate_station),
          'semantic_ctx_valid_progress': int(semantic_ctx_valid_progress),
          'semantic_ctx_station_reuse': int(semantic_ctx_station_reuse),
          'semantic_ctx_delayed_after_use': int(semantic_ctx_delayed_after_use),
          'semantic_post_fault_window': int(semantic_post_fault_window),
          'semantic_post_fault_nonzero': int(semantic_post_fault_nonzero),
          'semantic_post_fault_switch': int(semantic_post_fault_switch),
          'fault_count_so_far': int(self._fault_count),
          'semantic_trigger_context': int(semantic_trigger_context),
          'semantic_trigger_type': semantic_trigger_type,
          'semantic_trigger_count': int(semantic_trigger_count),
          'semantic_first_trigger_step': int(semantic_first_trigger_step),
          'semantic_ctx_upgrade_collect': int(semantic_ctx_upgrade_collect),
          'semantic_ctx_retry_craft': int(semantic_ctx_retry_craft),
          'semantic_ctx_relocate_station': int(semantic_ctx_relocate_station),
          'semantic_ctx_valid_progress': int(semantic_ctx_valid_progress),
          'semantic_ctx_station_reuse': int(semantic_ctx_station_reuse),
          'semantic_ctx_delayed_after_use': int(semantic_ctx_delayed_after_use),
          'semantic_post_fault_window': int(semantic_post_fault_window),
          'semantic_post_fault_nonzero': int(semantic_post_fault_nonzero),
          'semantic_post_fault_switch': int(semantic_post_fault_switch),
          'is_last': bool(self._done),
          'is_terminal': bool(info.get('discount', 1.0) == 0.0),
      }
      with self._trace_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

    if self._fault_verbose and fault_applied:
      print(
          f'[FaultApplied] ep={self._episode} step={self._length} '
          f'family={fault_family} type={fault_type} '
          f'requested={requested_action} executed={env_action} reward={reward:.3f}'
      )

    if self._done:
      self._update_scheduler_after_episode()
      self._write_stats(self._length, self._reward, info)
      self._write_episode_summary(self._length, self._reward, info)

      if self._record_gifs and (
          self._fault_count > 0 or self._semantic_fault_episode_seen > 0):
          fault_family, fault_type, _ = self._resolve_episode_fault_identity(info)
          gif_filename = self._output_dir / f"bug_{fault_type}_ep{self._episode}.gif"
          try:
              import imageio
              imageio.mimsave(str(gif_filename), self._episode_frames, fps=15)
              if self._fault_verbose:
                  print(f"[GIF SAVED] {gif_filename}")
          except ImportError:
              print("[Warning] imageio is not installed. Please run 'pip install imageio' to save GIFs.")

    # success window based on raw reward event
    if raw_reward > 0.0:
      self._after_positive_window = 3
      self._last_positive_reward_step = int(self._episode_step + 1)
    elif self._after_positive_window > 0:
      self._after_positive_window -= 1

    if self._fault_cooldown > 0:
      self._fault_cooldown -= 1

    self._last_reward = float(reward)
    self._last_requested_action = int(requested_action)
    self._prev_executed_action = int(env_action)
    self._requested_hist.append(int(requested_action))
    self._executed_hist.append(int(env_action))
    self._reward_hist.append(float(reward))

    self._episode_step += 1
    self._record_context(image)

    if self._record_gifs:
        self._record_hud_frame(image, requested_action, info)

    return self._obs(
        image=image,
        reward=reward,
        info=info,
        is_first=False,
        is_last=self._done,
        is_terminal=(info.get('discount', 1.0) == 0.0),
    )

  def _record_hud_frame(self, image, requested_action, info):
    try:
      from PIL import Image, ImageDraw
      img_pil = Image.fromarray(image).resize((512, 512), Image.NEAREST).convert('RGB')
      bug_active = bool(
          int(info.get('fault_applied', 0)) or
          int(info.get('semantic_fault_applied', 0)))
      if bug_active:
        red_screen = Image.new('RGB', img_pil.size, (255, 0, 0))
        img_pil = Image.blend(img_pil, red_screen, 0.75)
      draw = ImageDraw.Draw(img_pil)

      act_str = crafter.constants.actions[requested_action] if requested_action < len(crafter.constants.actions) else "unknown"
      fault_type = info.get('fault_type', 'none')
      if not fault_type or fault_type == 'none':
        fault_type = info.get('semantic_fault_type', 'none')
      if not fault_type or fault_type == 'none':
        fault_type = (
            getattr(self, '_fault_spec', {}).get('type', 'None')
            if getattr(self, '_fault_spec', None) else 'None')
      sem_applied = int(info.get('semantic_fault_applied', 0))

      inv = info.get('inventory', {})
      inv_str = f"Wood:{inv.get('wood',0)}  Stone:{inv.get('stone',0)}  Coal:{inv.get('coal',0)}  Iron:{inv.get('iron',0)}\n"
      inv_str += f"Table:{inv.get('table',0)}  Furnace:{inv.get('furnace',0)}  WoodPick:{inv.get('wood_pickaxe',0)}"

      msg = f"Step: {self._length} | Action: {act_str}\nTarget: {fault_type}"

      if bug_active:
        label = 'SEMANTIC BUG TRIGGERED' if sem_applied > 0 else 'BUG TRIGGERED'
        msg += f"\n\n>>> {label}! <<<"
        draw.rectangle((0, 0, 511, 511), outline=(255, 255, 255), width=8)
        draw.text((15, 15), msg, fill=(255, 255, 255))
      else:
        draw.text((15, 15), msg, fill=(255, 255, 255))

      draw.text((15, 460), f"[Inventory Status]\n{inv_str}", fill=(150, 255, 150))

      self._episode_frames.append(np.array(img_pil))
    except ImportError:
      frame = np.array(image, copy=True)
      if int(info.get('fault_applied', 0)) or int(info.get('semantic_fault_applied', 0)):
        frame[..., 0] = 255
        frame[..., 1:] = (frame[..., 1:] * 0.25).astype(frame.dtype)
      self._episode_frames.append(frame)

  def _obs(
      self,
      image,
      reward,
      info,
      is_first=False,
      is_last=False,
      is_terminal=False,
  ):
    log_reward = 0.0
    if info and 'reward' in info:
      log_reward = info['reward']

    fault_applied = 0
    if info and 'fault_applied' in info:
      fault_applied = info['fault_applied']

    fault_episode = int(self._fault_episode)
    if info and 'fault_episode' in info:
      fault_episode = info['fault_episode']
    fault_exists_episode = int(
        info.get('fault_exists_episode', fault_episode)) if info else int(fault_episode)
    fault_trigger_context = int(info.get('fault_trigger_context', 0)) if info else 0
    lowlevel_trigger_context = int(info.get('lowlevel_trigger_context', 0)) if info else 0
    fault_manifested = int(info.get('fault_manifested', fault_applied)) if info else int(fault_applied)
    fault_manifest_prob = float(info.get('fault_manifest_prob', 0.0)) if info else 0.0
    fault_profile = info.get('fault_profile', self._fault_profile) if info else self._fault_profile
    fault_profile_id = FAULT_PROFILE_IDS.get(str(fault_profile), 0)
    fault_frequency_tier = (
        info.get('fault_frequency_tier', self._fault_frequency_tier)
        if info else self._fault_frequency_tier)
    fault_frequency_tier_id = FAULT_FREQUENCY_TIER_IDS.get(
        str(fault_frequency_tier), 0)
    fault_family = info.get('fault_family', 'none') if info else 'none'
    fault_type = info.get('fault_type', 'none') if info else 'none'
    semantic_fault_type_for_id = (
        info.get('semantic_fault_type', 'none') if info else 'none')
    if fault_type in ('', 'none') and semantic_fault_type_for_id not in ('', 'none'):
      fault_type = semantic_fault_type_for_id
    fault_family_id = FAULT_FAMILY_IDS.get(str(fault_family), 0)
    fault_type_id = FAULT_TYPE_IDS.get(str(fault_type), 0)

    revisit_context = 1 if self._recent_revisit_active() else 0
    delayed_switch_context = 1 if (
        self._delayed_switch_min <= (self._episode_step - self._last_positive_reward_step) <= self._delayed_switch_max
    ) else 0

    semantic_fault_applied = 0
    if info and 'semantic_fault_applied' in info:
      semantic_fault_applied = info['semantic_fault_applied']

    semantic_fault_episode = 0
    if info and 'semantic_fault_episode' in info:
      semantic_fault_episode = info['semantic_fault_episode']

    semantic_trigger_context = int(info.get('semantic_trigger_context', 0)) if info else 0
    semantic_trigger_count = int(info.get('semantic_trigger_count', 0)) if info else 0
    semantic_first_trigger_step = int(info.get('semantic_first_trigger_step', -1)) if info else -1
    semantic_ctx_upgrade_collect = int(info.get('semantic_ctx_upgrade_collect', 0)) if info else 0
    semantic_ctx_retry_craft = int(info.get('semantic_ctx_retry_craft', 0)) if info else 0
    semantic_ctx_relocate_station = int(info.get('semantic_ctx_relocate_station', 0)) if info else 0
    semantic_ctx_valid_progress = int(info.get('semantic_ctx_valid_progress', 0)) if info else 0
    semantic_ctx_station_reuse = int(info.get('semantic_ctx_station_reuse', 0)) if info else 0
    semantic_ctx_delayed_after_use = int(info.get('semantic_ctx_delayed_after_use', 0)) if info else 0
    semantic_post_fault_window = int(info.get('semantic_post_fault_window', 0)) if info else 0
    semantic_post_fault_nonzero = int(info.get('semantic_post_fault_nonzero', 0)) if info else 0
    semantic_post_fault_switch = int(info.get('semantic_post_fault_switch', 0)) if info else 0
    semantic_upgrade_collect_count = int(info.get('semantic_upgrade_collect_count', 0)) if info else 0
    semantic_retry_craft_count = int(info.get('semantic_retry_craft_count', 0)) if info else 0
    semantic_relocate_station_count = int(info.get('semantic_relocate_station_count', 0)) if info else 0
    semantic_valid_progress_count = int(info.get('semantic_valid_progress_count', 0)) if info else 0
    semantic_station_reuse_count = int(info.get('semantic_station_reuse_count', 0)) if info else 0
    semantic_delayed_after_use_count = int(info.get('semantic_delayed_after_use_count', 0)) if info else 0
    task_reward_raw = float(info.get('task_reward_raw', 0.0)) if info else 0.0
    tester_bonus = float(info.get('tester_bonus', 0.0)) if info else 0.0
    tester_repeat_penalty = float(info.get('tester_repeat_penalty', 0.0)) if info else 0.0
    rnd_intrinsic_reward = float(info.get('rnd_intrinsic_reward', 0.0)) if info else 0.0
    rnd_intrinsic_reward_raw = float(info.get('rnd_intrinsic_reward_raw', 0.0)) if info else 0.0
    rnd_loss = float(info.get('rnd_loss', 0.0)) if info else 0.0
    rnd_update_enabled = int(info.get('rnd_update_enabled', 0)) if info else 0
    unique_contexts = int(info.get('unique_contexts', len(self._ctx_visit_steps))) if info else int(len(self._ctx_visit_steps))
    unique_tiles_visited = int(info.get('unique_tiles_visited', len(self._visited_tiles))) if info else int(len(self._visited_tiles))
    tile_coverage_ratio = float(info.get('tile_coverage_ratio', self._tile_coverage_ratio())) if info else float(self._tile_coverage_ratio())
    unique_target_cells = int(info.get('unique_target_cells', len(self._target_cells))) if info else int(len(self._target_cells))
    unique_target_materials = int(info.get('unique_target_materials', len(self._target_materials))) if info else int(len(self._target_materials))
    unique_cells_modified = int(info.get('unique_cells_modified', len(self._modified_cells))) if info else int(len(self._modified_cells))
    unique_materials_modified = int(info.get('unique_materials_modified', len(self._modified_material_types))) if info else int(len(self._modified_material_types))
    world_edit_count = int(info.get('world_edit_count', self._world_edit_count)) if info else int(self._world_edit_count)
    world_step_edit_count = int(info.get('world_step_edit_count', self._last_world_step_edit_count)) if info else int(self._last_world_step_edit_count)
    world_edit_ratio = float(info.get('world_edit_ratio', self._world_edit_ratio())) if info else float(self._world_edit_ratio())

    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        **{'log/reward': np.float32(log_reward)},
        **{'log/fault_applied': np.int32(fault_applied)},
        **{'log/fault_episode': np.int32(fault_episode)},
        **{'log/fault_exists_episode': np.int32(fault_exists_episode)},
        **{'log/fault_trigger_context': np.int32(fault_trigger_context)},
        **{'log/lowlevel_trigger_context': np.int32(lowlevel_trigger_context)},
        **{'log/fault_manifested': np.int32(fault_manifested)},
        **{'log/fault_manifest_prob': np.float32(fault_manifest_prob)},
        **{'log/fault_profile_id': np.int32(fault_profile_id)},
        **{'log/fault_frequency_tier_id': np.int32(fault_frequency_tier_id)},
        **{'log/fault_family_id': np.int32(fault_family_id)},
        **{'log/fault_type_id': np.int32(fault_type_id)},
        **{'log/fault_count_cumulative': np.int32(self._fault_count)},
        **{'log/semantic_fault_applied': np.int32(semantic_fault_applied)},
        **{'log/semantic_fault_episode': np.int32(semantic_fault_episode)},
        **{'log/semantic_trigger_context': np.int32(semantic_trigger_context)},
        **{'log/semantic_trigger_count': np.int32(semantic_trigger_count)},
        **{'log/semantic_first_trigger_step': np.int32(semantic_first_trigger_step)},
        **{'log/semantic_ctx_upgrade_collect': np.int32(semantic_ctx_upgrade_collect)},
        **{'log/semantic_ctx_retry_craft': np.int32(semantic_ctx_retry_craft)},
        **{'log/semantic_ctx_relocate_station': np.int32(semantic_ctx_relocate_station)},
        **{'log/semantic_ctx_valid_progress': np.int32(semantic_ctx_valid_progress)},
        **{'log/semantic_ctx_station_reuse': np.int32(semantic_ctx_station_reuse)},
        **{'log/semantic_ctx_delayed_after_use': np.int32(semantic_ctx_delayed_after_use)},
        **{'log/semantic_post_fault_window': np.int32(semantic_post_fault_window)},
        **{'log/semantic_post_fault_nonzero': np.int32(semantic_post_fault_nonzero)},
        **{'log/semantic_post_fault_switch': np.int32(semantic_post_fault_switch)},
        **{'log/semantic_upgrade_collect_count': np.int32(semantic_upgrade_collect_count)},
        **{'log/semantic_retry_craft_count': np.int32(semantic_retry_craft_count)},
        **{'log/semantic_relocate_station_count': np.int32(semantic_relocate_station_count)},
        **{'log/semantic_valid_progress_count': np.int32(semantic_valid_progress_count)},
        **{'log/semantic_station_reuse_count': np.int32(semantic_station_reuse_count)},
        **{'log/semantic_delayed_after_use_count': np.int32(semantic_delayed_after_use_count)},
        **{'log/task_reward_raw': np.float32(task_reward_raw)},
        **{'log/tester_bonus': np.float32(tester_bonus)},
        **{'log/tester_repeat_penalty': np.float32(tester_repeat_penalty)},
        **{'log/rnd_intrinsic_reward': np.float32(rnd_intrinsic_reward)},
        **{'log/rnd_intrinsic_reward_raw': np.float32(rnd_intrinsic_reward_raw)},
        **{'log/rnd_loss': np.float32(rnd_loss)},
        **{'log/rnd_update_enabled': np.int32(rnd_update_enabled)},
        **{'log/revisit_context': np.int32(revisit_context)},
        **{'log/delayed_switch_context': np.int32(delayed_switch_context)},
        **{'log/unique_contexts': np.int32(unique_contexts)},
        **{'log/unique_tiles_visited': np.int32(unique_tiles_visited)},
        **{'log/tile_coverage_ratio': np.float32(tile_coverage_ratio)},
        **{'log/unique_target_cells': np.int32(unique_target_cells)},
        **{'log/unique_target_materials': np.int32(unique_target_materials)},
        **{'log/unique_cells_modified': np.int32(unique_cells_modified)},
        **{'log/unique_materials_modified': np.int32(unique_materials_modified)},
        **{'log/world_edit_count': np.int32(world_edit_count)},
        **{'log/world_step_edit_count': np.int32(world_step_edit_count)},
        **{'log/world_edit_ratio': np.float32(world_edit_ratio)},
        **{'log/global_step': np.int32(self._global_step)},
    )

    if self._logs:
      achievements = info.get('achievements', {}) if info else {}
      log_achievements = {
          f'log/achievement_{k}': achievements.get(k, 0)
          for k in self._achievements
      }
      obs.update({k: np.int32(v) for k, v in log_achievements.items()})

    return obs

  def _write_stats(self, length, reward, info):
    if self._stats_path is None:
      return

    info = dict(info or {})
    achievements = info.get('achievements', {})
    stats = {
        'episode': self._episode,
        'length': int(length),
        'reward': round(float(reward), 1),
        'task_reward': round(float(info.get('task_reward_raw', self._episode_task_reward)), 6),
        'episode_tester_bonus': round(float(self._episode_tester_bonus), 6),
        'episode_repeat_penalty': round(float(self._episode_repeat_penalty), 6),
        'episode_rnd_intrinsic_raw_mean': round(float(self._ep_rnd_intrinsic_raw_sum / max(1, self._ep_rnd_steps)), 6),
        'episode_rnd_intrinsic_mean': round(float(self._ep_rnd_intrinsic_norm_sum / max(1, self._ep_rnd_steps)), 6),
        'episode_rnd_loss_mean': round(float(self._ep_rnd_loss_sum / max(1, self._ep_rnd_steps)), 6),
        'rnd_update_enabled': int(self._rnd_update),
        'start_global_step': int(self._episode_start_global_step),
        'end_global_step': int(self._global_step),
        'fault_episode': int(info.get('fault_episode', 0)),
        'fault_exists_episode': int(info.get(
            'fault_exists_episode', getattr(self, '_ep_fault_exists_episode', 0))),
        'fault_trigger_context_count': int(getattr(
            self, '_ep_fault_trigger_context_count', 0)),
        'lowlevel_trigger_context_count': int(getattr(
            self, '_ep_lowlevel_trigger_context_count', 0)),
        'semantic_trigger_context_count': int(getattr(
            self, '_ep_semantic_trigger_context_count', 0)),
        'fault_manifested_any': int(getattr(self, '_ep_fault_applied_any', 0)),
        'fault_manifest_count': int(getattr(self, '_ep_fault_manifest_count', 0)),
        'fault_manifest_prob_max': round(float(getattr(
            self, '_ep_fault_manifest_prob_max', info.get('fault_manifest_prob', 0.0))), 6),
        'fault_profile': self._fault_profile,
        'fault_profile_requested': self._fault_profile_requested,
        'fault_stochastic_manifest': int(self._fault_stochastic_manifest),
        'fault_frequency_tier': self._fault_frequency_tier,
        'fault_count': int(self._fault_count),
        'fault_rate': float(self._fault_count / max(1, length)),
        'fault_family': self._ep_fault_family if getattr(self, '_ep_fault_family', 'none') != 'none' else info.get('fault_family', 'none'),
        'fault_type': self._ep_fault_type if getattr(self, '_ep_fault_type', 'none') != 'none' else info.get('fault_type', 'none'),
        'fault_trigger': self._ep_fault_trigger if getattr(self, '_ep_fault_trigger', 'none') != 'none' else info.get('fault_trigger', 'none'),
        'fault_severity': float(info.get('fault_severity', 0.0)),
        'fault_episode_prob': float(self._fault_episode_prob),
        'fault_cooldown_steps': int(self._fault_cooldown_steps),
        'episode_unique_contexts': int(len(self._ctx_visit_steps)),
        'episode_unique_tiles_visited': int(len(self._visited_tiles)),
        'episode_tile_coverage_ratio': round(float(self._tile_coverage_ratio()), 6),
        'episode_unique_target_cells': int(len(self._target_cells)),
        'episode_unique_target_materials': int(len(self._target_materials)),
        'episode_unique_cells_modified': int(len(self._modified_cells)),
        'episode_world_edit_count': int(self._world_edit_count),
        'episode_world_edit_ratio': round(float(self._world_edit_ratio()), 6),
        'episode_unique_materials_modified': int(len(self._modified_material_types)),
        **{
            f'achievement_{k}': achievements.get(k, 0)
            for k in self._achievements
        },
    }
    with self._stats_path.open('a', encoding='utf-8') as f:
      f.write(json.dumps(stats) + '\n')
      f.flush()

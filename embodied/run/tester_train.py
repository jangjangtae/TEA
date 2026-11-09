import collections
import math
import os
from collections import deque
from functools import partial as bind

import elements
import embodied
import numpy as np


def _scalar(x, default=0.0):
  if x is None:
    return default
  arr = np.asarray(x)
  if arr.size == 0:
    return default
  return float(arr.reshape(-1)[0])


class RunningNorm:

  def __init__(self, eps=1e-6, warmup=100):
    self.eps = eps
    self.warmup = warmup
    self.count = 0
    self.mean = 0.0
    self.m2 = 0.0

  def update(self, x):
    x = float(x)
    self.count += 1
    delta = x - self.mean
    self.mean += delta / self.count
    delta2 = x - self.mean
    self.m2 += delta * delta2

  @property
  def std(self):
    if self.count < 2:
      return 1.0
    return math.sqrt(max(self.m2 / (self.count - 1), self.eps))

  def normalize(self, x, clip=5.0, signed=True):
    if self.count < self.warmup:
      return 0.0
    z = (float(x) - self.mean) / max(self.std, self.eps)
    if signed:
      return float(np.clip(z, -clip, clip))
    return float(np.clip(z, 0.0, clip))


class RepeatPenalty:

  def __init__(self, window=4, reward_eps=1e-3):
    self.window = int(window)
    self.reward_eps = float(reward_eps)
    self.buffers = collections.defaultdict(lambda: deque(maxlen=self.window))

  def penalty(self, worker, action, game_reward):
    if action is None:
      return 0.0
    try:
      action = int(np.asarray(action).reshape(-1)[0])
    except Exception:
      return 0.0
    buf = self.buffers[worker]
    buf.append(action)
    if len(buf) < self.window:
      return 0.0
    if len(set(buf)) == 1 and abs(float(game_reward)) <= self.reward_eps:
      return 1.0
    return 0.0


class CoverageTracker:
  """
  Helper for global and local coverage rewards.
  Global coverage provides a small always-on bonus.
  Local coverage rewards episode-first visits inside suspicious regions.
  """

  def __init__(self, stride=8, recent_window=200):
    self.stride = int(stride)
    self.recent_window = int(recent_window)

    self.global_counts = collections.defaultdict(int)
    self.episode_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    self.recent_episode_first = collections.deque(maxlen=self.recent_window)

    self.action_sets = collections.defaultdict(set)
    self.action_bigram_sets = collections.defaultdict(set)
    self.action_hist = collections.defaultdict(lambda: deque(maxlen=2))

  def _hash(self, image):
    image = np.asarray(image)
    if image.ndim != 3:
      return None
    small = image[::self.stride, ::self.stride]
    if small.shape[-1] == 3:
      small = small.mean(-1)
    small = small.astype(np.uint8, copy=False)
    return small.tobytes()

  def reset_episode(self, worker):
    self.episode_counts[worker].clear()
    self.action_sets[worker].clear()
    self.action_bigram_sets[worker].clear()
    self.action_hist[worker].clear()

  def step(self, worker, image, action):
    key = self._hash(image)

    is_global_first = 0.0
    is_episode_first = 0.0
    global_unique = 0.0
    episode_unique = 0.0
    revisit_ratio = 0.0
    recent_episode_first_rate = 0.0
    unique_actions = 0.0
    unique_action_bigrams = 0.0

    if key is not None:
      is_global_first = 1.0 if self.global_counts[key] == 0 else 0.0
      is_episode_first = 1.0 if self.episode_counts[worker][key] == 0 else 0.0

      self.global_counts[key] += 1
      self.episode_counts[worker][key] += 1
      self.recent_episode_first.append(is_episode_first)

      global_unique = float(len(self.global_counts))
      episode_unique = float(len(self.episode_counts[worker]))

      ep_steps = float(sum(self.episode_counts[worker].values()))
      revisit_ratio = 0.0 if ep_steps <= 0 else 1.0 - (episode_unique / ep_steps)
      recent_episode_first_rate = (
          float(np.mean(self.recent_episode_first))
          if self.recent_episode_first else 0.0)

    if action is not None:
      try:
        a = int(np.asarray(action).reshape(-1)[0])
        self.action_sets[worker].add(a)
        self.action_hist[worker].append(a)
        if len(self.action_hist[worker]) == 2:
          self.action_bigram_sets[worker].add(tuple(self.action_hist[worker]))
      except Exception:
        pass

    unique_actions = float(len(self.action_sets[worker]))
    unique_action_bigrams = float(len(self.action_bigram_sets[worker]))

    return {
        'is_global_first': float(is_global_first),
        'is_episode_first': float(is_episode_first),
        'global_unique_states': global_unique,
        'episode_unique_states': episode_unique,
        'episode_revisit_ratio': float(revisit_ratio),
        'recent_episode_first_rate': float(recent_episode_first_rate),
        'unique_actions': unique_actions,
        'unique_action_bigrams': unique_action_bigrams,
    }


class RetentionBandController:
  """
  Controller for anomaly-guided reward scheduling.
  - Uses recent episode score to choose a retention zone.
  - Keeps the task term active with a small weight.
  - Strengthens recovery only in the red zone.
  - Scales bug and local coverage terms by suspicion.
  """

  def __init__(
      self,
      baseline_score=11.8,
      green_ratio=0.85,
      yellow_ratio=0.65,
      repeat_budget=0.08,
      init_lambda_recover=1.0,
      init_lambda_rep=0.1,
      init_w_bug=0.55,
      init_beta_cov=0.15,
      min_w_bug=0.35,
      max_w_bug=0.90,
      min_beta_cov=0.08,
      max_beta_cov=0.25,
      lambda_recover_up_red=0.12,
      lambda_recover_decay=0.997,
      lambda_rep_lr=0.02,
      w_bug_up_green=0.002,
      w_bug_up_yellow=0.001,
      w_bug_down_red=0.01,
      beta_up_green=0.002,
      beta_up_yellow=0.001,
      beta_down_red=0.01,
      max_lambda_recover=5.0,
      max_lambda_rep=3.0,
      task_gate_warmup=0.25,
      task_gate_green=0.20,
      task_gate_yellow=0.25,
      task_gate_red=0.30,
      explore_gate_warmup=0.85,
      explore_gate_green=0.75,
      explore_gate_yellow=0.65,
      explore_gate_red=0.55,
      severe_red_threshold=0.55,
  ):
    self.baseline_score = float(baseline_score)
    self.green_ratio = float(green_ratio)
    self.yellow_ratio = float(yellow_ratio)
    self.repeat_budget = float(repeat_budget)

    self.lambda_recover = float(init_lambda_recover)
    self.lambda_rep = float(init_lambda_rep)

    self.w_bug = float(init_w_bug)
    self.beta_cov = float(init_beta_cov)

    self.min_w_bug = float(min_w_bug)
    self.max_w_bug = float(max_w_bug)
    self.min_beta_cov = float(min_beta_cov)
    self.max_beta_cov = float(max_beta_cov)

    self.lambda_recover_up_red = float(lambda_recover_up_red)
    self.lambda_recover_decay = float(lambda_recover_decay)
    self.lambda_rep_lr = float(lambda_rep_lr)

    self.w_bug_up_green = float(w_bug_up_green)
    self.w_bug_up_yellow = float(w_bug_up_yellow)
    self.w_bug_down_red = float(w_bug_down_red)

    self.beta_up_green = float(beta_up_green)
    self.beta_up_yellow = float(beta_up_yellow)
    self.beta_down_red = float(beta_down_red)

    self.max_lambda_recover = float(max_lambda_recover)
    self.max_lambda_rep = float(max_lambda_rep)

    self.task_gate_warmup = float(task_gate_warmup)
    self.task_gate_green = float(task_gate_green)
    self.task_gate_yellow = float(task_gate_yellow)
    self.task_gate_red = float(task_gate_red)

    self.explore_gate_warmup = float(explore_gate_warmup)
    self.explore_gate_green = float(explore_gate_green)
    self.explore_gate_yellow = float(explore_gate_yellow)
    self.explore_gate_red = float(explore_gate_red)

    self.severe_red_threshold = float(severe_red_threshold)

    self.zone = 'warmup'
    self.retention = 1.0

  @property
  def green_floor(self):
    return self.baseline_score * self.green_ratio

  @property
  def yellow_floor(self):
    return self.baseline_score * self.yellow_ratio

  def classify(self, recent_score_mean):
    if recent_score_mean is None or self.baseline_score <= 0:
      self.zone = 'warmup'
      self.retention = 1.0
      return self.zone

    self.retention = float(recent_score_mean) / self.baseline_score

    if recent_score_mean >= self.green_floor:
      self.zone = 'green'
    elif recent_score_mean >= self.yellow_floor:
      self.zone = 'yellow'
    else:
      self.zone = 'red'
    return self.zone

  def task_gate(self):
    if self.zone == 'warmup':
      return self.task_gate_warmup
    if self.zone == 'green':
      return self.task_gate_green
    if self.zone == 'yellow':
      return self.task_gate_yellow
    return self.task_gate_red

  def explore_gate(self):
    if self.zone == 'warmup':
      return self.explore_gate_warmup
    if self.zone == 'green':
      return self.explore_gate_green
    if self.zone == 'yellow':
      return self.explore_gate_yellow
    return self.explore_gate_red

  def zone_code(self):
    return {
        'green': 2.0,
        'yellow': 1.0,
        'red': 0.0,
        'warmup': -1.0,
    }[self.zone]

  def update(self, recent_score_mean=None, repeat_mean=None):
    zone = self.classify(recent_score_mean)

    if repeat_mean is not None:
      gap_rep = float(repeat_mean) - self.repeat_budget
      self.lambda_rep = float(np.clip(
          self.lambda_rep + self.lambda_rep_lr * gap_rep,
          0.0, self.max_lambda_rep))

    if zone == 'green':
      self.lambda_recover = float(np.clip(
          self.lambda_recover * self.lambda_recover_decay,
          0.5, self.max_lambda_recover))
      self.w_bug = float(np.clip(
          self.w_bug + self.w_bug_up_green,
          self.min_w_bug, self.max_w_bug))
      self.beta_cov = float(np.clip(
          self.beta_cov + self.beta_up_green,
          self.min_beta_cov, self.max_beta_cov))

    elif zone == 'yellow':
      self.lambda_recover = float(np.clip(
          self.lambda_recover * self.lambda_recover_decay,
          0.5, self.max_lambda_recover))
      self.w_bug = float(np.clip(
          self.w_bug + self.w_bug_up_yellow,
          self.min_w_bug, self.max_w_bug))
      self.beta_cov = float(np.clip(
          self.beta_cov + self.beta_up_yellow,
          self.min_beta_cov, self.max_beta_cov))

    elif zone == 'red':
      gap = self.yellow_floor - float(recent_score_mean)
      self.lambda_recover = float(np.clip(
          self.lambda_recover + self.lambda_recover_up_red * max(gap, 0.0),
          0.5, self.max_lambda_recover))

      severe_red = self.retention < self.severe_red_threshold
      if severe_red:
        self.w_bug = float(np.clip(
            self.w_bug - self.w_bug_down_red,
            self.min_w_bug, self.max_w_bug))
        self.beta_cov = float(np.clip(
            self.beta_cov - self.beta_down_red,
            self.min_beta_cov, self.max_beta_cov))


def tester_train(make_agent, make_replay, make_env, make_stream, make_logger, args):
  assert args.from_checkpoint, (
      'tester_train expects a clean gameplay checkpoint as initialization. '
      'Please set --run.from_checkpoint.')

  # --------------------------------------------------
  # Hyperparameters
  # --------------------------------------------------
  repeat_window = int(os.getenv('TESTER_REPEAT_WINDOW', '4'))
  repeat_reward_eps = float(os.getenv('TESTER_REPEAT_REWARD_EPS', '0.001'))
  coverage_stride = int(os.getenv('TESTER_COVERAGE_STRIDE', '8'))
  coverage_recent_window = int(os.getenv('TESTER_COVERAGE_RECENT_WINDOW', '200'))

  from dreamerv3 import fault_score as faultlib

  ref_score_key = os.getenv('TESTER_REF_SCORE_KEY', 'log/ref_fault_score')
  ref_checkpoint = os.getenv('TESTER_REF_CHECKPOINT', args.from_checkpoint)

  baseline_score = float(os.getenv('TESTER_BASELINE_SCORE', '11.8'))
  green_ratio = float(os.getenv('TESTER_GREEN_RATIO', '0.85'))
  yellow_ratio = float(os.getenv('TESTER_YELLOW_RATIO', '0.65'))
  repeat_budget = float(os.getenv('TESTER_REPEAT_BUDGET', '0.08'))

  score_window = int(os.getenv('TESTER_SCORE_EP_WINDOW', '20'))
  repeat_window_stats = int(os.getenv('TESTER_REPEAT_EP_WINDOW', '20'))

  init_lambda_recover = float(os.getenv('TESTER_INIT_LAMBDA_RECOVER', '1.0'))
  init_lambda_rep = float(os.getenv('TESTER_INIT_LAMBDA_REPEAT', '0.1'))

  init_w_bug = float(os.getenv('TESTER_INIT_W_BUG', '0.55'))
  init_beta_cov = float(os.getenv('TESTER_INIT_BETA_COV', '0.15'))

  min_w_bug = float(os.getenv('TESTER_MIN_W_BUG', '0.35'))
  max_w_bug = float(os.getenv('TESTER_MAX_W_BUG', '0.90'))
  min_beta_cov = float(os.getenv('TESTER_MIN_BETA_COV', '0.08'))
  max_beta_cov = float(os.getenv('TESTER_MAX_BETA_COV', '0.25'))

  lambda_recover_up_red = float(os.getenv('TESTER_LAMBDA_RECOVER_UP_RED', '0.12'))
  lambda_recover_decay = float(os.getenv('TESTER_LAMBDA_RECOVER_DECAY', '0.997'))
  lambda_rep_lr = float(os.getenv('TESTER_LAMBDA_REP_LR', '0.02'))

  w_bug_up_green = float(os.getenv('TESTER_W_BUG_UP_GREEN', '0.002'))
  w_bug_up_yellow = float(os.getenv('TESTER_W_BUG_UP_YELLOW', '0.001'))
  w_bug_down_red = float(os.getenv('TESTER_W_BUG_DOWN_RED', '0.01'))

  beta_up_green = float(os.getenv('TESTER_BETA_COV_UP_GREEN', '0.002'))
  beta_up_yellow = float(os.getenv('TESTER_BETA_COV_UP_YELLOW', '0.001'))
  beta_down_red = float(os.getenv('TESTER_BETA_COV_DOWN_RED', '0.01'))

  max_lambda_recover = float(os.getenv('TESTER_MAX_LAMBDA_RECOVER', '5.0'))
  max_lambda_rep = float(os.getenv('TESTER_MAX_LAMBDA_REPEAT', '3.0'))

  task_gate_warmup = float(os.getenv('TESTER_TASK_GATE_WARMUP', '0.25'))
  task_gate_green = float(os.getenv('TESTER_TASK_GATE_GREEN', '0.20'))
  task_gate_yellow = float(os.getenv('TESTER_TASK_GATE_YELLOW', '0.25'))
  task_gate_red = float(os.getenv('TESTER_TASK_GATE_RED', '0.30'))

  explore_gate_warmup = float(os.getenv('TESTER_EXPLORE_GATE_WARMUP', '0.85'))
  explore_gate_green = float(os.getenv('TESTER_EXPLORE_GATE_GREEN', '0.75'))
  explore_gate_yellow = float(os.getenv('TESTER_EXPLORE_GATE_YELLOW', '0.65'))
  explore_gate_red = float(os.getenv('TESTER_EXPLORE_GATE_RED', '0.55'))

  severe_red_threshold = float(os.getenv('TESTER_SEVERE_RED_THRESHOLD', '0.55'))

  # reward weights
  alpha_task_base = float(os.getenv('TESTER_ALPHA_TASK_BASE', '0.20'))
  alpha_cov_global = float(os.getenv('TESTER_ALPHA_COV_GLOBAL', '0.03'))
  alpha_detect = float(os.getenv('TESTER_ALPHA_DETECT', '0.10'))

  task_clip = float(os.getenv('TESTER_TASK_Z_CLIP', '5.0'))
  bug_clip = float(os.getenv('TESTER_BUG_Z_CLIP', '5.0'))
  rep_clip = float(os.getenv('TESTER_REP_Z_CLIP', '5.0'))

  norm_warmup = int(os.getenv('TESTER_NORM_WARMUP', '100'))
  bug_warmup = int(os.getenv('TESTER_BUG_NORM_WARMUP', '100'))

  # suspicion
  suspicion_ema_alpha = float(os.getenv('TESTER_SUSPICION_EMA_ALPHA', '0.10'))
  suspicion_low = float(os.getenv('TESTER_SUSPICION_LOW', '0.75'))
  suspicion_high = float(os.getenv('TESTER_SUSPICION_HIGH', '2.25'))
  suspicion_arm = float(os.getenv('TESTER_SUSPICION_ARM', '0.55'))

  local_window = int(os.getenv('TESTER_LOCAL_WINDOW', '8'))
  detect_suspicion_thresh = float(os.getenv('TESTER_DETECT_SUSPICION', '0.75'))
  detect_streak_steps = int(os.getenv('TESTER_DETECT_STREAK', '3'))

  # --------------------------------------------------
  # Agents / replay / logger
  # --------------------------------------------------
  tester_agent = make_agent()
  ref_agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()

  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  task_norm = RunningNorm(warmup=norm_warmup)
  bug_norm = RunningNorm(warmup=bug_warmup)
  rep_norm = RunningNorm(warmup=norm_warmup)

  coverage = CoverageTracker(
      stride=coverage_stride,
      recent_window=coverage_recent_window)
  repeat_penalty = RepeatPenalty(
      window=repeat_window, reward_eps=repeat_reward_eps)

  recent_scores = deque(maxlen=score_window)
  recent_repeat_means = deque(maxlen=repeat_window_stats)
  episode_fault_seen = collections.defaultdict(int)

  suspicion_ema = collections.defaultdict(float)
  suspicion_countdown = collections.defaultdict(int)
  suspicion_streak = collections.defaultdict(int)
  detect_bonus_given = collections.defaultdict(int)

  controller = RetentionBandController(
      baseline_score=baseline_score,
      green_ratio=green_ratio,
      yellow_ratio=yellow_ratio,
      repeat_budget=repeat_budget,
      init_lambda_recover=init_lambda_recover,
      init_lambda_rep=init_lambda_rep,
      init_w_bug=init_w_bug,
      init_beta_cov=init_beta_cov,
      min_w_bug=min_w_bug,
      max_w_bug=max_w_bug,
      min_beta_cov=min_beta_cov,
      max_beta_cov=max_beta_cov,
      lambda_recover_up_red=lambda_recover_up_red,
      lambda_recover_decay=lambda_recover_decay,
      lambda_rep_lr=lambda_rep_lr,
      w_bug_up_green=w_bug_up_green,
      w_bug_up_yellow=w_bug_up_yellow,
      w_bug_down_red=w_bug_down_red,
      beta_up_green=beta_up_green,
      beta_up_yellow=beta_up_yellow,
      beta_down_red=beta_down_red,
      max_lambda_recover=max_lambda_recover,
      max_lambda_rep=max_lambda_rep,
      task_gate_warmup=task_gate_warmup,
      task_gate_green=task_gate_green,
      task_gate_yellow=task_gate_yellow,
      task_gate_red=task_gate_red,
      explore_gate_warmup=explore_gate_warmup,
      explore_gate_green=explore_gate_green,
      explore_gate_yellow=explore_gate_yellow,
      explore_gate_red=explore_gate_red,
      severe_red_threshold=severe_red_threshold,
  )

  print('Tester training logdir:', logdir)
  print('Reference checkpoint:', ref_checkpoint)
  print('Reference bug key:', ref_score_key)
  print('Reward design: anomaly-guided realistic shaping (no fault gate)')
  print(dict(
      baseline_score=baseline_score,
      green_ratio=green_ratio,
      yellow_ratio=yellow_ratio,
      green_floor=controller.green_floor,
      yellow_floor=controller.yellow_floor,
      alpha_task_base=alpha_task_base,
      alpha_cov_global=alpha_cov_global,
      alpha_detect=alpha_detect,
      suspicion_low=suspicion_low,
      suspicion_high=suspicion_high,
      suspicion_arm=suspicion_arm,
      local_window=local_window,
      detect_suspicion_thresh=detect_suspicion_thresh,
      detect_streak_steps=detect_streak_steps,
  ))

  # --------------------------------------------------
  # Logging
  # --------------------------------------------------
  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    if tran['is_first']:
      episode.reset()
      episode_fault_seen[worker] = 0

    game_reward = _scalar(tran.get('log/game_reward', 0.0))
    tester_reward = _scalar(tran['reward'])
    fault_episode = int(_scalar(tran.get('log/fault_episode', 0.0)) > 0.5)
    fault_applied = int(_scalar(tran.get('log/fault_applied', 0.0)) > 0.5)

    episode_fault_seen[worker] = max(
        episode_fault_seen[worker], fault_episode, fault_applied)

    episode.add('score', game_reward, agg='sum')
    episode.add('tester_score', tester_reward, agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', game_reward, agg='stack')

    for key, value in tran.items():
      if getattr(value, 'dtype', None) == np.uint8 and getattr(value, 'ndim', None) == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')

    if tran['is_last']:
      result = episode.result()
      score = result.pop('score')
      tester_score = result.pop('tester_score')
      length = result.pop('length')
      ep_fault = int(episode_fault_seen[worker])

      logger.add({
          'score': score,
          'tester_score': tester_score,
          'length': length,
          'fault_episode': ep_fault,
      }, prefix='episode')

      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()

      rep_avg = float(result.get('log/raw_repeat_penalty/avg', 0.0))
      recent_repeat_means.append(rep_avg)
      recent_scores.append(float(score))

      recent_score_mean = (
          float(np.mean(recent_scores))
          if len(recent_scores) > 0 else None
      )
      repeat_mean = (
          float(np.mean(recent_repeat_means))
          if len(recent_repeat_means) > 0 else None
      )

      controller.update(recent_score_mean, repeat_mean)

      result['recent_score_mean'] = (
          recent_score_mean if recent_score_mean is not None else 0.0
      )
      result['repeat_mean_recent'] = (
          repeat_mean if repeat_mean is not None else 0.0
      )
      result['retention'] = controller.retention
      result['green_floor'] = controller.green_floor
      result['yellow_floor'] = controller.yellow_floor
      result['lambda_recover'] = controller.lambda_recover
      result['lambda_rep'] = controller.lambda_rep
      result['w_bug'] = controller.w_bug
      result['beta_cov'] = controller.beta_cov
      result['task_gate'] = controller.task_gate()
      result['explore_gate'] = controller.explore_gate()
      result['competence_zone'] = controller.zone_code()

      epstats.add(result)

  # --------------------------------------------------
  # Policy wrapper
  # --------------------------------------------------
  def init_dual_policy(batch_size):
    return {
        'tester': tester_agent.init_policy(batch_size),
        'ref': ref_agent.init_policy(batch_size),
    }

  def dual_policy(carry, obs, mode='train'):
    tester_carry, acts, tester_outs = tester_agent.policy(
        carry['tester'], obs, mode='train')
    ref_carry, _, ref_outs = ref_agent.policy(
        carry['ref'], obs, mode='eval')
    # The reference model must condition its next prior on the same action
    # that is actually sent to the environment by the tester policy.
    ref_carry = (*ref_carry[:-1], tester_carry[-1])

    outs = dict(tester_outs)
    faultlib.add_reference_outputs(outs, ref_outs)

    return {'tester': tester_carry, 'ref': ref_carry}, acts, outs

  # --------------------------------------------------
  # Reward shaping
  # --------------------------------------------------
  @elements.timer.section('shape_reward')
  def shape_reward(tran, worker):
    env_reward = _scalar(tran.get('reward', 0.0))
    game_reward = _scalar(tran.get('log/task_reward_raw', env_reward))

    if tran['is_first']:
      coverage.reset_episode(worker)
      suspicion_ema[worker] = 0.0
      suspicion_countdown[worker] = 0
      suspicion_streak[worker] = 0
      detect_bonus_given[worker] = 0

      raw_bug = 0.0
      raw_cov_global = 0.0
      raw_cov_local = 0.0
      raw_rep = 0.0

      norm_task = 0.0
      norm_bug_delta = 0.0
      norm_rep = 0.0
      suspicion_score = 0.0
      bug_z = 0.0

      task_base_term = 0.0
      cov_global_term = 0.0
      suspicion_term = 0.0
      detect_bonus = 0.0
      recovery_term = 0.0
      tester_reward = 0.0

      cov_stats = {
          'is_global_first': 0.0,
          'is_episode_first': 0.0,
          'global_unique_states': 0.0,
          'episode_unique_states': 0.0,
          'episode_revisit_ratio': 0.0,
          'recent_episode_first_rate': 0.0,
          'unique_actions': 0.0,
          'unique_action_bigrams': 0.0,
      }

    else:
      raw_bug = _scalar(tran.get(ref_score_key, 0.0))
      raw_rep = repeat_penalty.penalty(
          worker, tran.get('action', None), game_reward)

      cov_stats = coverage.step(worker, tran.get('image'), tran.get('action', None))
      raw_cov_global = float(cov_stats['is_episode_first'])

      task_norm.update(game_reward)
      rep_norm.update(raw_rep)
      bug_norm.update(raw_bug)

      norm_task = task_norm.normalize(game_reward, clip=task_clip, signed=True)
      norm_rep = rep_norm.normalize(raw_rep, clip=rep_clip, signed=False)

      if bug_norm.count < bug_norm.warmup:
        bug_z = 0.0
        norm_bug_delta = 0.0
      else:
        bug_z = (float(raw_bug) - bug_norm.mean) / max(bug_norm.std, bug_norm.eps)
        bug_z_pos = max(0.0, bug_z)

        suspicion_ema[worker] = (
            (1.0 - suspicion_ema_alpha) * suspicion_ema[worker]
            + suspicion_ema_alpha * bug_z_pos
        )

        norm_bug_delta = float(np.clip(
            max(0.0, bug_z_pos - suspicion_ema[worker]),
            0.0, bug_clip))

      # suspicion score in [0, 1]
      suspicion_score = float(np.clip(
          (suspicion_ema[worker] - suspicion_low) / max(suspicion_high - suspicion_low, 1e-8),
          0.0, 1.0))

      # suspicious local window
      if suspicion_score >= suspicion_arm:
        suspicion_countdown[worker] = local_window
      elif suspicion_countdown[worker] > 0:
        suspicion_countdown[worker] -= 1

      local_active = suspicion_countdown[worker] > 0
      raw_cov_local = float(cov_stats['is_episode_first']) if local_active else 0.0

      # detection bonus: sustained suspicion crossing
      if suspicion_score >= detect_suspicion_thresh:
        suspicion_streak[worker] += 1
      else:
        suspicion_streak[worker] = 0

      detect_bonus = 0.0
      if (
          suspicion_streak[worker] >= detect_streak_steps
          and not detect_bonus_given[worker]
      ):
        detect_bonus = alpha_detect
        detect_bonus_given[worker] = 1

      # reward terms
      task_base_term = alpha_task_base * controller.task_gate() * norm_task
      cov_global_term = alpha_cov_global * raw_cov_global

      suspicion_term = (
          suspicion_score
          * controller.explore_gate()
          * (
              controller.w_bug * norm_bug_delta
              + controller.beta_cov * raw_cov_local
          )
      )

      recovery_term = 0.0
      if controller.zone == 'red':
        recovery_term = (
            controller.lambda_recover
            * max(0.0, norm_task)
        )

      tester_reward = (
          task_base_term
          + cov_global_term
          + suspicion_term
          + detect_bonus
          + recovery_term
          - controller.lambda_rep * norm_rep
      )

    tran['reward'] = np.float32(tester_reward)

    tran['log/env_reward'] = np.float32(env_reward)
    tran['log/game_reward'] = np.float32(game_reward)
    tran['log/raw_bug_reward'] = np.float32(raw_bug)
    tran['log/raw_cov_global'] = np.float32(raw_cov_global)
    tran['log/raw_cov_local'] = np.float32(raw_cov_local)
    tran['log/raw_repeat_penalty'] = np.float32(raw_rep)

    tran['log/task_reward_norm'] = np.float32(norm_task)
    tran['log/bug_reward_norm'] = np.float32(norm_bug_delta)
    tran['log/repeat_penalty_norm'] = np.float32(norm_rep)
    tran['log/bug_z'] = np.float32(bug_z)
    tran['log/suspicion_ema'] = np.float32(suspicion_ema[worker])
    tran['log/suspicion_score'] = np.float32(suspicion_score)

    tran['log/task_base_term'] = np.float32(task_base_term)
    tran['log/cov_global_term'] = np.float32(cov_global_term)
    tran['log/suspicion_term'] = np.float32(suspicion_term)
    tran['log/detect_bonus'] = np.float32(detect_bonus)
    tran['log/recovery_term'] = np.float32(recovery_term)

    tran['log/lambda_recover'] = np.float32(controller.lambda_recover)
    tran['log/lambda_rep'] = np.float32(controller.lambda_rep)
    tran['log/w_bug'] = np.float32(controller.w_bug)
    tran['log/beta_cov'] = np.float32(controller.beta_cov)
    tran['log/task_gate'] = np.float32(controller.task_gate())
    tran['log/explore_gate'] = np.float32(controller.explore_gate())
    tran['log/retention'] = np.float32(controller.retention)
    tran['log/green_floor'] = np.float32(controller.green_floor)
    tran['log/yellow_floor'] = np.float32(controller.yellow_floor)
    tran['log/competence_zone'] = np.float32(controller.zone_code())

    tran['log/local_active'] = np.float32(1.0 if suspicion_countdown[worker] > 0 else 0.0)
    tran['log/suspicion_streak'] = np.float32(suspicion_streak[worker])

    tran['log/is_global_first_state'] = np.float32(cov_stats['is_global_first'])
    tran['log/is_episode_first_state'] = np.float32(cov_stats['is_episode_first'])
    tran['log/global_unique_states'] = np.float32(cov_stats['global_unique_states'])
    tran['log/episode_unique_states'] = np.float32(cov_stats['episode_unique_states'])
    tran['log/episode_revisit_ratio'] = np.float32(cov_stats['episode_revisit_ratio'])
    tran['log/recent_episode_first_rate'] = np.float32(cov_stats['recent_episode_first_rate'])
    tran['log/unique_actions'] = np.float32(cov_stats['unique_actions'])
    tran['log/unique_action_bigrams'] = np.float32(cov_stats['unique_action_bigrams'])

  # --------------------------------------------------
  # Replay add with filtering
  # --------------------------------------------------
  @elements.timer.section('replay_add_filtered')
  def replay_add_filtered(tran, worker):
    allowed = set(tester_agent.spaces.keys())
    filtered = {k: v for k, v in tran.items() if k in allowed}
    replay.add(filtered, worker)

  # --------------------------------------------------
  # Driver / replay / stream
  # --------------------------------------------------
  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(shape_reward)
  driver.on_step(replay_add_filtered)
  driver.on_step(logfn)

  stream_train = iter(tester_agent.stream(make_stream(replay, 'train')))
  stream_report = iter(tester_agent.stream(make_stream(replay, 'report')))
  carry_train = [tester_agent.init_train(args.batch_size)]
  carry_report = tester_agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = tester_agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')

  driver.on_step(trainfn)

  # --------------------------------------------------
  # Checkpoints
  # --------------------------------------------------
  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = tester_agent
  cp.replay = replay

  load_regex = args.from_checkpoint_regex if hasattr(args, 'from_checkpoint_regex') else None

  if load_regex is None:
    elements.checkpoint.load(
        args.from_checkpoint,
        dict(agent=tester_agent.load))
  else:
    elements.checkpoint.load(
        args.from_checkpoint,
        dict(agent=bind(tester_agent.load, regex=load_regex)))

  if load_regex is None:
    elements.checkpoint.load(
        ref_checkpoint,
        dict(agent=ref_agent.load))
  else:
    elements.checkpoint.load(
        ref_checkpoint,
        dict(agent=bind(ref_agent.load, regex=load_regex)))

  cp.load_or_save()

  print('Start tester fine-tuning loop')

  driver.reset(init_dual_policy)

  while step < args.steps:
    driver(dual_policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = tester_agent.report(
            carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()

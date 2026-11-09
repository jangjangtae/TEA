import collections
import json
import os
from functools import partial as bind

import elements
import embodied
import numpy as np


def _filter_transition_for_replay(tran):
  # Drop auxiliary/debug keys that the agent does not consume.
  drop_prefixes = ('bug/', 'fault/', 'log/diag_')
  return {
      k: v for k, v in tran.items()
      if not any(k.startswith(prefix) for prefix in drop_prefixes)
  }


def _cfg_get(config, key, default=None):
  if config is None:
    return default
  if hasattr(config, 'get'):
    return config.get(key, default)
  return getattr(config, key, default)


def _fault_enabled(args):
  return bool(_cfg_get(getattr(args, 'fault', None), 'enabled', False))


def _load_agent_checkpoint(path, agent, regex=None):
  if regex:
    elements.checkpoint.load(path, dict(agent=bind(agent.load, regex=regex)))
  else:
    elements.checkpoint.load(path, dict(agent=agent.load))


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  fault_cfg = getattr(args, 'fault', None)
  use_fault = _fault_enabled(args)
  faultlib = None
  ref_agent = None
  use_reference = False
  fault_tracker = None
  fault_stats = {}
  no_first_bonus = None
  if use_fault:
    from dreamerv3 import fault_score as faultlib
    use_reference = faultlib.uses_reference_model(fault_cfg)
    if use_reference:
      ref_agent = make_agent()
    fault_tracker = faultlib.FaultRewardTracker(fault_cfg)
    fault_stats = faultlib.load_norm_stats(_cfg_get(
        fault_cfg, 'norm_stats', ''))
  if os.environ.get('TEA_NO_FIRST_TRANSITION_BONUS') == '1':
    from dreamerv3.tea_no_first_bonus import FirstTransitionBonusMask
    assert use_fault, 'First-transition ablation requires TEA instrumentation.'
    no_first_bonus = FirstTransitionBonusMask(fault_cfg)
  replay = make_replay()
  replay_schedule = None
  if os.environ.get('FATE_INITIAL_REPLAY_MANIFEST'):
    from dreamerv3.fate_replay_restore import ReplayScheduleAudit, restore_initial_replay
    restored = restore_initial_replay(replay, args)
    if restored.get('matched_train_start'):
      replay_schedule = ReplayScheduleAudit(args.logdir, restored['matched_train_start'])
  elif os.environ.get('TEA_AUDIT_FRESH_SCHEDULE') == '1':
    from dreamerv3.fate_replay_restore import ReplayScheduleAudit
    replay_schedule = ReplayScheduleAudit(args.logdir, 2048)
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  fault_trace_path = logdir / _cfg_get(
      fault_cfg, 'trace', 'fault_score_trace.jsonl')
  if use_fault:
    with open(str(fault_trace_path), 'a', encoding='utf-8') as f:
      pass
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()
  fault_episode_id = collections.defaultdict(int)
  fault_episode_step = collections.defaultdict(int)
  fault_prev_score = collections.defaultdict(float)
  diagnostic_recorder = None
  if os.environ.get('FATE_ACTION_DIAGNOSTICS') == '1':
    from dreamerv3 import fate_action_diagnostics as diagnostics
    diagnostic_recorder = diagnostics.Recorder(logdir, True)

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if key == 'image' and value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  @elements.timer.section('fault_score')
  def apply_fault_score(tran, worker):
    if not use_fault:
      return
    if tran['is_first']:
      fault_prev_score[worker] = 0.0
    tran['log/fault_score_prev'] = np.float32(fault_prev_score[worker])
    result = faultlib.compute_transition_fault(
        tran, fault_cfg, fault_stats, force_log_only=False,
        beta_override=fault_tracker.beta)
    if no_first_bonus is not None:
      no_first_bonus.apply(tran, worker, result)
    fault_tracker.apply(tran, worker, result, force_log_only=False)
    faultlib.add_transition_fault_logs(tran, result)
    fault_prev_score[worker] = float(result['fault_score'])
    if not _cfg_get(fault_cfg, 'log_only', True):
      tran['reward'] = np.float32(result['augmented_reward'])

  @elements.timer.section('fault_trace')
  def write_fault_trace(tran, worker):
    if not use_fault:
      return
    if tran['is_first']:
      fault_episode_id[worker] += 1
      fault_episode_step[worker] = 0
    fault_episode_step[worker] += 1
    record = faultlib.trace_record(
        tran, int(step), worker, fault_episode_id[worker],
        fault_episode_step[worker])
    if no_first_bonus is not None:
      record.update(no_first_bonus.trace_fields(worker))
    with open(str(fault_trace_path), 'a', encoding='utf-8') as f:
      f.write(json.dumps(record) + '\n')

  def init_policy(batch_size):
    if not use_fault:
      return agent.init_policy(batch_size)
    if not use_reference:
      return agent.init_policy(batch_size)
    return {
        'agent': agent.init_policy(batch_size),
        'ref': ref_agent.init_policy(batch_size),
    }

  def policy(carry, obs, mode='train'):
    if not use_fault:
      return agent.policy(carry, obs, mode=mode)
    if not use_reference:
      return agent.policy(carry, obs, mode=mode)
    agent_carry, acts, outs = agent.policy(carry['agent'], obs, mode=mode)
    ref_carry, _, ref_outs = ref_agent.policy(carry['ref'], obs, mode='eval')
    # Align the frozen reference prior with the action that actually enters env.
    ref_carry = (*ref_carry[:-1], agent_carry[-1])
    outs = dict(outs)
    faultlib.add_reference_outputs(outs, ref_outs)
    return {'agent': agent_carry, 'ref': ref_carry}, acts, outs

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(apply_fault_score)
  driver.on_step(write_fault_trace)
  if diagnostic_recorder:
    driver.on_step(lambda tran, worker: diagnostic_recorder(tran, worker, int(step)))
  # Each worker must retain its own temporal stream for RSSM sequence learning.
  driver.on_step(lambda tran, worker: replay.add(
      _filter_transition_for_replay(tran), worker))

  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    if replay_schedule and not replay_schedule.ready(int(step)):
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      if replay_schedule:
        replay_schedule.record(int(step))
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  load_regex = args.from_checkpoint_regex if hasattr(args, 'from_checkpoint_regex') else None
  if args.from_checkpoint:
    _load_agent_checkpoint(args.from_checkpoint, agent, load_regex)
  if use_fault and use_reference:
    ref_ckpt = _cfg_get(fault_cfg, 'ref_ckpt', '') or args.from_checkpoint
    assert ref_ckpt, 'fault.ref_ckpt or run.from_checkpoint is required.'
    _load_agent_checkpoint(ref_ckpt, ref_agent, load_regex)
  cp.load_or_save()

  identity_audit = None
  if os.environ.get('FATE_REFERENCE_AUDIT', '0') == '1':
    assert use_reference and ref_agent is not None, 'Reference audit requires a frozen model.'
    from dreamerv3.fate_reference_audit import ReferenceAudit
    identity_audit = ReferenceAudit(
        logdir, agent, ref_agent, args.from_checkpoint, ref_ckpt)

  print('Start training loop')
  if use_fault:
    print('Fault score enabled')
    print('Fault score source:', _cfg_get(fault_cfg, 'score_source', 'latent_reward'))
    if use_reference:
      print('Fault reference checkpoint:', _cfg_get(fault_cfg, 'ref_ckpt', '') or args.from_checkpoint)
    else:
      print('Fault reference checkpoint: disabled (online agent score source)')
    print('Fault trace file:', fault_trace_path)
  train_policy = lambda *args: policy(*args, mode='train')
  driver.reset(init_policy)
  while step < args.steps:

    driver(train_policy, steps=10)
    if diagnostic_recorder:
      diagnostics.save_milestone(cp, step, logdir)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
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

  # Periodic saves are wall-clock based. Persist once more at the exact target
  # step so staged long runs can evaluate and resume deterministic milestones.
  cp.save()
  if replay_schedule:
    replay_schedule.finish(int(step))
  if identity_audit:
    identity_audit.finish(agent, ref_agent)
  if diagnostic_recorder:
    diagnostic_recorder.close()
  logger.add(train_agg.result())
  logger.add(epstats.result(), prefix='epstats')
  logger.add(replay.stats(), prefix='replay')
  logger.add({'fps/policy': policy_fps.result(), 'fps/train': train_fps.result()})
  logger.write()
  logger.close()
  driver.close()

from collections import defaultdict
from functools import partial as bind
import json

import elements
import embodied
import numpy as np


def _to_python(x):
  if isinstance(x, np.ndarray):
    if x.ndim == 0:
      return x.item()
    return x.tolist()
  if isinstance(x, (np.floating, np.integer, np.bool_)):
    return x.item()
  return x


def _cfg_get(config, key, default=None):
  if config is None:
    return default
  if hasattr(config, 'get'):
    return config.get(key, default)
  return getattr(config, key, default)


def _fault_enabled(args):
  cfg = getattr(args, 'fault', None)
  return bool(_cfg_get(cfg, 'enabled', False) or _cfg_get(cfg, 'ref_ckpt', ''))


def eval_only(make_agent, make_env, make_logger, args):
  assert args.from_checkpoint

  agent = make_agent()
  fault_cfg = getattr(args, 'fault', None)
  use_fault = _fault_enabled(args)
  faultlib = None
  ref_agent = None
  use_reference = False
  fault_tracker = None
  fault_stats = {}
  if use_fault:
    from dreamerv3 import fault_score as faultlib
    use_reference = faultlib.uses_reference_model(fault_cfg)
    if use_reference:
      ref_agent = make_agent()
    fault_tracker = faultlib.FaultRewardTracker(fault_cfg)
    fault_stats = faultlib.load_norm_stats(_cfg_get(
        fault_cfg, 'norm_stats', ''))
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)

  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  epstats = elements.Agg()
  episodes = defaultdict(elements.Agg)
  should_log = elements.when.Clock(args.log_every)
  policy_fps = elements.FPS()

  # Always create the bug trace file, even for fault-free runs.
  bug_trace_path = logdir / 'bug_trace.jsonl'
  with open(str(bug_trace_path), 'w', encoding='utf-8') as f:
    pass
  fault_trace_path = logdir / _cfg_get(
      fault_cfg, 'trace', 'fault_score_trace.jsonl')
  if use_fault:
    with open(str(fault_trace_path), 'w', encoding='utf-8') as f:
      pass

  ep_step = defaultdict(int)
  ep_index = defaultdict(int)
  fault_prev_score = defaultdict(float)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]

    if tran['is_first']:
      episode.reset()
      ep_step[worker] = 0
      ep_index[worker] += 1

    ep_step[worker] += 1

    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    for key, value in tran.items():
      isimage = (value.dtype == np.uint8) and (value.ndim == 3)
      if key == 'image' and isimage and worker == 0:
        episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
      elif key.startswith('bug/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
      elif key.startswith('fault/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')

    # Record every step for both clean and faulty environments.
    record = {
        'global_step': int(step),
        'worker': int(worker),
        'episode_index': int(ep_index[worker]),
        'episode_step': int(ep_step[worker]),
        'reward': _to_python(tran['reward']),
        'is_first': _to_python(tran['is_first']),
        'is_last': _to_python(tran['is_last']),
        'is_terminal': _to_python(tran['is_terminal']),
    }

    if 'action' in tran:
      record['action'] = _to_python(tran['action'])

    # Add bug/* and fault/* fields when present.
    for key in tran.keys():
      if key.startswith('bug/'):
        record[key] = _to_python(tran[key])
      if key.startswith('fault/'):
        record[key] = _to_python(tran[key])

    # Add scalar log/* fields as well.
    for key in tran.keys():
      if key.startswith('log/'):
        value = tran[key]
        if getattr(value, 'ndim', 0) == 0:
          record[key] = _to_python(value)

    with open(str(bug_trace_path), 'a', encoding='utf-8') as f:
      f.write(json.dumps(record) + '\n')
    if use_fault:
      fault_record = faultlib.trace_record(
          tran, int(step), worker, ep_index[worker], ep_step[worker])
      with open(str(fault_trace_path), 'a', encoding='utf-8') as f:
        f.write(json.dumps(fault_record) + '\n')

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

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=(not args.debug))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())

  @elements.timer.section('fault_score')
  def apply_fault_score(tran, worker):
    if not use_fault:
      return
    if tran['is_first']:
      fault_prev_score[worker] = 0.0
    tran['log/fault_score_prev'] = np.float32(fault_prev_score[worker])
    result = faultlib.compute_transition_fault(
        tran, fault_cfg, fault_stats, force_log_only=True,
        beta_override=fault_tracker.beta)
    fault_tracker.apply(tran, worker, result, force_log_only=True)
    faultlib.add_transition_fault_logs(tran, result)
    fault_prev_score[worker] = float(result['fault_score'])

  driver.on_step(apply_fault_score)
  driver.on_step(logfn)

  cp = elements.Checkpoint()
  cp.agent = agent
  cp.load(args.from_checkpoint, keys=['agent'])
  if use_fault and use_reference:
    ref_ckpt = _cfg_get(fault_cfg, 'ref_ckpt', '') or args.from_checkpoint
    elements.checkpoint.load(ref_ckpt, dict(agent=ref_agent.load))

  print('Start evaluation')
  print('Bug trace file:', bug_trace_path)
  if use_fault:
    print('Fault score source:', _cfg_get(fault_cfg, 'score_source', 'latent_reward'))
    if use_reference:
      print('Fault reference checkpoint:', _cfg_get(fault_cfg, 'ref_ckpt', '') or args.from_checkpoint)
    else:
      print('Fault reference checkpoint: disabled (online agent score source)')
    print('Fault trace file:', fault_trace_path)

  def init_policy(batch_size):
    if not use_fault:
      return agent.init_policy(batch_size)
    if not use_reference:
      return agent.init_policy(batch_size)
    return {
        'agent': agent.init_policy(batch_size),
        'ref': ref_agent.init_policy(batch_size),
    }

  def policy(carry, obs, mode='eval'):
    if not use_fault:
      return agent.policy(carry, obs, mode=mode)
    if not use_reference:
      return agent.policy(carry, obs, mode=mode)
    agent_carry, acts, outs = agent.policy(carry['agent'], obs, mode=mode)
    ref_carry, _, ref_outs = ref_agent.policy(carry['ref'], obs, mode='eval')
    ref_carry = (*ref_carry[:-1], agent_carry[-1])
    outs = dict(outs)
    faultlib.add_reference_outputs(outs, ref_outs)
    return {'agent': agent_carry, 'ref': ref_carry}, acts, outs

  eval_policy = lambda *args: policy(*args, mode='eval')
  driver.reset(init_policy)

  while step < args.steps:
    driver(eval_policy, steps=10)
    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

  logger.close()

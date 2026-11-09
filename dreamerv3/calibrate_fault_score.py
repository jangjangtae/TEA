import argparse
import json
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import ruamel.yaml as yaml

from . import fault_score
from . import main as dreamer_main


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', default=['crafter'])
  parser.add_argument('--ref_ckpt', default='')
  parser.add_argument('--episodes', type=int, default=None)
  parser.add_argument('--steps', type=int, default=None)
  parser.add_argument('--out', default='')
  parser.add_argument('--trace', default='')
  parser.add_argument('--logdir', default='~/logdir/fault_calibration')
  parser.add_argument('--keep_env_faults', action='store_true')
  args, other = parser.parse_known_args(argv)
  return args, other


def load_config(config_names, other, logdir):
  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  config = elements.Config(configs['defaults'])
  for name in config_names:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  return config.update(logdir=os.path.expanduser(logdir))


def clean_fault_env():
  os.environ['CRAFTER_FAULT'] = '0'
  os.environ['CRAFTER_FAULT_SAMPLER'] = '0'
  os.environ['CRAFTER_SEMANTIC_FAULT_SAMPLER'] = '0'
  os.environ['CRAFTER_TESTER_REWARD'] = '0'
  os.environ['CRAFTER_USE_RND'] = '0'
  os.environ['CRAFTER_RND_UPDATE'] = '0'
  os.environ['CRAFTAX_FAULT'] = '0'
  os.environ['CRAFTAX_FAULT_SAMPLER'] = '0'


def resolve_checkpoint(path):
  path = elements.Path(path)
  if (path / 'done').exists():
    return path
  latest = path / 'latest'
  if latest.exists():
    return path / latest.read_text().strip()
  return path


class Collector:

  def __init__(self, trace_path=None):
    self.episodes = 0
    self.steps = 0
    self.latent_kl = []
    self.reward_error = []
    self.fault_raw = []
    self.context_scores = {}
    self.trace_path = trace_path and pathlib.Path(trace_path).expanduser()
    if self.trace_path:
      self.trace_path.parent.mkdir(parents=True, exist_ok=True)
      self.trace_path.write_text('', encoding='utf-8')
    self.ep_step = {}
    self.ep_id = {}
    self.next_ep_id = 0

  def on_step(self, tran, worker):
    if tran['is_first']:
      self.next_ep_id += 1
      self.ep_id[worker] = self.next_ep_id
      self.ep_step[worker] = 0
    self.ep_step[worker] = self.ep_step.get(worker, 0) + 1

    if not tran['is_first']:
      latent_kl = _scalar(tran.get('fault/latent_kl_surprise', 0.0))
      reward_error = _scalar(tran.get('fault/reward_prediction_error', 0.0))
      fault_raw = _scalar(tran.get('fault/fault_score_raw', 0.0))
      self.latent_kl.append(latent_kl)
      self.reward_error.append(reward_error)
      self.fault_raw.append(fault_raw)
      context = fault_score.context_keys(tran)
      for key in context.values():
        self.context_scores.setdefault(key, []).append(fault_raw)
      self.steps += 1

      if self.trace_path:
        row = {
            'transition_index': self.steps,
            'worker': int(worker),
            'episode_id': int(self.ep_id.get(worker, 0)),
            'episode_step': int(self.ep_step.get(worker, 0)),
            'reward': _scalar(tran.get('reward', 0.0)),
            'action': _to_python(tran.get('action', 0)),
            'latent_kl_surprise': latent_kl,
            'reward_prediction_error': reward_error,
            'fault_score_raw': fault_raw,
            'fault_context_key': context['full'],
            'is_last': bool(tran['is_last']),
            'is_terminal': bool(tran['is_terminal']),
        }
        with self.trace_path.open('a', encoding='utf-8') as f:
          f.write(json.dumps(row) + '\n')

    if tran['is_last']:
      self.episodes += 1

  def summary(self):
    data = {}
    data.update(fault_score.summarize(self.latent_kl, 'latent_kl'))
    data.update(fault_score.summarize(self.reward_error, 'reward_error'))
    data.update(fault_score.summarize(self.fault_raw, 'fault_score'))
    data['context_schema'] = fault_score.CONTEXT_SCHEMA
    data['context_stats'] = {
        key: fault_score.summarize(values, 'fault_score')
        for key, values in sorted(self.context_scores.items())
    }
    data['episodes'] = int(self.episodes)
    data['steps'] = int(self.steps)
    return data


def main(argv=None):
  args, other = parse_args(argv)
  if not args.keep_env_faults:
    clean_fault_env()

  config = load_config(args.configs, other, args.logdir)
  os.environ.setdefault(
      'CRAFTER_OUTPUT_DIR',
      str(elements.Path(config.logdir) / 'env'))
  episodes = args.episodes
  if episodes is None:
    episodes = int(config.calibration.episodes)
  steps = args.steps
  if steps is None:
    steps = int(config.calibration.steps)

  ref_ckpt = args.ref_ckpt or config.fault.ref_ckpt
  if not ref_ckpt:
    raise ValueError('--ref_ckpt or --fault.ref_ckpt is required.')
  ref_ckpt = resolve_checkpoint(ref_ckpt)

  out = args.out or config.calibration.out
  if not out:
    out = elements.Path(config.logdir) / 'clean_fault_stats.json'
  trace = args.trace or config.calibration.trace

  logdir = elements.Path(config.logdir)
  logdir.mkdir()
  print('Calibration logdir:', logdir)
  print('Reference checkpoint:', ref_ckpt)
  print('Output stats:', out)
  if trace:
    print('Trace:', trace)

  agent = dreamer_main.make_agent(config)
  cp = elements.Checkpoint()
  cp.agent = agent
  cp.load(ref_ckpt, keys=['agent'])

  collector = Collector(trace)
  fns = [bind(dreamer_main.make_env, config, i) for i in range(config.run.envs)]
  driver = embodied.Driver(fns, parallel=(not config.run.debug))
  driver.on_step(collector.on_step)

  policy = lambda *xs: agent.policy(*xs, mode='eval')
  driver.reset(agent.init_policy)
  if steps and steps > 0:
    while collector.steps < steps:
      driver(policy, steps=10)
  else:
    driver(policy, episodes=episodes)

  stats = collector.summary()
  fault_score.write_json(out, stats)
  print(json.dumps(stats, indent=2, sort_keys=True))


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


if __name__ == '__main__':
  main()

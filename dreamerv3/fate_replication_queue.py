"""Audited P0 repetition, fault SparseSurprise, and independent P1 comparison."""

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import pickle
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from dreamerv3.analyze_craftax_multiseed import iter_jsonl
from dreamerv3.analyze_fate_corrected_completed import summarize_trace
from dreamerv3.audit_fate_revision_job import audit_job
from dreamerv3.audit_fate_revision_run import (
    check_clean_replay_history, check_trace, close, read_config, require)
from dreamerv3.fate_clean_novelty import calibrate, require_clean
from dreamerv3.fate_diagnostic_followup import (
    SPLITS, audit_eval, check_diagnostic_rows, digest, write_json)


DATA = Path('./analysis_outputs')
P0_CLEAN = DATA / 'fate_clean_corrected_size1m_20260922_v2'
PREVIOUS = DATA / 'fate_clean_novelty_20260929_v2'
PYTHON = './dreamer_cuda/bin/python'
DEPS = './.deps/craftax_pkgs'
METHODS = dict(taskonly='TaskOnly', excess_delta_p95_beta02='FATE',
               sparse_surprise_matched='SparseSurprise')
REPEAT_OFFSET = 10000
P1_PRETRAIN_SEED = 1


def stamp():
  return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def file_map(paths):
  return {str(Path(path).resolve()): digest(path) for path in paths}


def verify_files(files):
  for path, expected in files.items():
    require(Path(path).is_file() and digest(path) == expected, f'Changed or missing file: {path}')


def latest_checkpoint(train):
  directory = Path(train) / 'ckpt'
  name = (directory / 'latest').read_text().strip()
  require(Path(name).name == name, 'Invalid checkpoint pointer')
  path = directory / name
  require((path / 'done').is_file() and (path / 'agent.pkl').is_file(), 'Incomplete checkpoint')
  return path


def budgets(smoke):
  return dict(train=4096 if smoke else 1000000, pretrain=4096 if smoke else 1100000,
              probe=4096 if smoke else 200000, clean=256 if smoke else 30000,
              fault=256 if smoke else 60000)


def safe_environment(source, train_root, output, profile='clean', parameters=None):
  # Allowlist process environment so another project's fault flags cannot leak in.
  env = {k: v for k, v in os.environ.items() if k in (
      'HOME', 'PATH', 'USER', 'LANG', 'LC_ALL', 'TMPDIR')}
  env.update(PROJECT_ROOT=str(source), TRAIN_ROOT=str(train_root), ROOT=str(output),
      PYTHON_BIN=PYTHON, PYTHONPATH=f'{source}:{DEPS}', PYTHONUNBUFFERED='1',
      PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0', JAX_PLATFORMS='cuda,cpu',
      XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='0.75',
      XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/lib/cuda',
      LD_LIBRARY_PATH='/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib/python3/dist-packages/tensorflow',
      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
      MPLCONFIGDIR=str(output / 'matplotlib'), CONFIGS='craftax size1m',
      TRAIN_ENVS='16', TRAIN_RATIO='128', REPLAY_SIZE='100000',
      BATCH_SIZE='16', BATCH_LENGTH='64', REPORT_LENGTH='32', SAVE_EVERY='600',
      JAX_PLATFORM='cuda', JAX_PREALLOC='True', JAX_COMPUTE_DTYPE='float32', EVAL_ENVS='1',
      TRAIN_FAULT_PROFILE=profile, TRAIN_FAULT_EP_PROB='0.5', TRAIN_FAULT_MANIFEST_PROB='1.0',
      TASK_REWARD_SOURCE='delivered', VARIANTS='taskonly',
      FATE_QUEUE_LIBRARY_ONLY='1', AUDIT_REVISION_JOBS='0',
      RESTART_INCOMPLETE_FROM_SCRATCH='1',
      RND_NORM='1', RND_CLIP='5.0', RND_DOWNSAMPLE='4', RND_HIDDEN_DIM='128',
      RND_OUTPUT_DIM='128', RND_LR='0.005')
  if (source / 'legacy_craftax.py').is_file():
    env['FATE_LEGACY_CRAFTAX'] = str(source / 'legacy_craftax.py')
  if parameters and 'sparse' in parameters:
    for key, value in dict(SPARSE_BETA=parameters['sparse']['beta'],
                          SPARSE_THRESHOLD=parameters['sparse']['threshold'],
                          SPARSE_CLIP=parameters['sparse']['clip']).items():
      require(math.isfinite(value) and value > 0, f'Invalid coefficient {key}')
      env[key] = str(value)
  return env


def shell_call(source, function, *arguments):
  require(function in ('run_eval', 'run_train_variant', 'make_norm_stats'), 'Unknown shell function')
  script = source / 'dreamerv3/run_craftax_multiseed_fault_queue.sh'
  body = 'source "$1"\nshift\n' + function + ' "$@"\n'
  return ['bash', '-Eeuo', 'pipefail', '-c', body, 'fate', str(script), *map(str, arguments)]


def pretrain_command(source, output, steps, template):
  args = shlex.split(template.read_text())
  require(args[0] == PYTHON and '--run.steps' in args, 'Unexpected pretrain template')
  args[1] = str(source / 'dreamerv3/main.py')
  for flag, value in {'--seed': P1_PRETRAIN_SEED, '--logdir': output,
                      '--run.steps': steps}.items():
    require(args.count(flag) == 1, f'Ambiguous pretrain argument {flag}')
    args[args.index(flag) + 1] = str(value)
  return args


def check_pretrain(directory, steps):
  cfg = read_config(directory / 'config.yaml')
  require(cfg['seed'] == P1_PRETRAIN_SEED and cfg['run']['steps'] == steps and
          cfg['run']['envs'] == 16 and cfg['run']['train_ratio'] == 512 and
          cfg['replay']['size'] == 200000 and not cfg['run']['from_checkpoint'] and
          not cfg['fault']['enabled'], 'Changed P1 pretraining protocol')
  report = check_clean_replay_history(directory.parent)
  checkpoint = latest_checkpoint(directory)
  with (checkpoint / 'step.pkl').open('rb') as handle:
    step = int(pickle.load(handle))
  require(steps <= step < steps + 16, 'Wrong pretraining checkpoint step')
  with (checkpoint / 'agent.pkl').open('rb') as handle:
    params = pickle.load(handle)['params']
  require(params and all(np.isfinite(v).all() for v in params.values()), 'Nonfinite pretraining model')
  losses = [v for row in iter_jsonl(directory / 'metrics.jsonl') for k, v in row.items()
            if k.startswith('train/loss/')]
  require(losses and all(math.isfinite(v) for v in losses), 'Missing/nonfinite pretraining losses')
  write_json(directory / 'pretrain_audit.json', dict(report, checkpoint=str(checkpoint),
      checkpoint_step=step, agent_sha256=digest(checkpoint / 'agent.pkl'), losses=len(losses)))
  return [directory / 'pretrain_audit.json', directory / 'config.yaml',
          checkpoint / 'agent.pkl', checkpoint / 'step.pkl', checkpoint / 'done']


def check_adaptation(root, seed, variant, steps, profile, expected):
  directory = root / f'seed_{seed}/train_{variant}/train'
  report = audit_job(root, seed, f'train_{variant}')
  cfg = read_config(directory / 'config.yaml')
  f = cfg['fault']
  require(cfg['seed'] == seed and cfg['run']['steps'] == steps and
          cfg['run']['envs'] == 16 and cfg['run']['train_ratio'] == 128 and
          cfg['replay']['size'] == 100000, 'Wrong adaptation budget or training configuration')
  require(f['ref_ckpt'] == expected['reference'] and
          digest(Path(f['ref_ckpt']) / 'agent.pkl') == expected['reference_sha256'] and
          digest(f['norm_stats']) == expected['stats_sha256'], 'Changed reference/calibration')
  if variant == 'sparse_surprise_matched':
    require(not f['log_only'] and f['reward_mode'] == 'excess_threshold', 'Wrong sparse objective')
    for key, config_key in (('beta', 'beta'), ('threshold', 'reward_threshold'), ('clip', 'clip')):
      require(close(f[config_key], expected['sparse'][key]), 'Changed SparseSurprise coefficient')
  elif variant == 'excess_delta_p95_beta02':
    require(not f['log_only'] and f['reward_mode'] == 'excess_delta_threshold' and
            close(f['beta'], .2) and close(f['reward_delta_threshold'], .5) and close(f['clip'], 2),
            'Changed FATE reward')
  else:
    require(variant in ('taskonly', 'reward_scale_probe') and f['log_only'], 'Wrong task-only objective')
  bins, count, faults, active, positive = {}, 0, 0, 0, 0
  for row in iter_jsonl(directory / 'fault_score_trace.jsonl'):
    count += 1
    require(row['global_step'] == count, 'Repeated/restarted trace')
    if profile == 'clean':
      require_clean(row)
    faults += int(row['bug_triggered'])
    active += int(row['log/fault_episode'])
    if variant == 'sparse_surprise_matched':
      value = (row['fault_score'] - f['reward_threshold']) / (f['clip'] - f['reward_threshold'])
    else:
      value = (row['fault_score_delta'] - .5) / 1.5
    bonus = 0 if row['is_first'] or f['log_only'] else f['beta'] * np.clip(value, 0, 1)
    require(close(row['learning_reward'], row['task_reward_delivered'] + bonus), 'Wrong delivered-plus-bonus reward')
    require(row['rnd_reward_bonus'] == 0, 'Unexpected RND reward')
    positive += int(bonus > 0)
    if not row['is_first']:
      bucket = bins.setdefault(str((count - 1) // 100000), dict(steps=0, bonus=0.,
          task=0., absolute_task=0., positive=0))
      bucket['steps'] += 1
      bucket['bonus'] += float(bonus)
      bucket['task'] += row['task_reward_delivered']
      bucket['absolute_task'] += abs(row['task_reward_delivered'])
      bucket['positive'] += int(bonus > 0)
  require(steps <= count < steps + 16, 'Wrong adaptation trace length')
  if not f['log_only']:
    require(positive > 0, 'Shaping never contributed')
  if profile != 'clean':
    require(active > 0, 'Fault training was not enabled')
  report.update(fault_events=faults, fault_episode_steps=active, reward_bins=bins,
                profile=profile, rows=count, config_sha256=digest(directory / 'config.yaml'))
  write_json(directory / 'replication_train_audit.json', report)
  checkpoint = latest_checkpoint(directory)
  return [directory / 'replication_train_audit.json', directory / 'config.yaml',
          checkpoint / 'agent.pkl', checkpoint / 'step.pkl', checkpoint / 'done']


def check_evaluation(directory, spec, raw_calibration=False):
  cfg = read_config(directory / 'config.yaml')
  require(cfg['seed'] == spec['eval_seed'] and cfg['run']['steps'] == spec['steps'], 'Wrong evaluation seed/budget')
  require(cfg['fault']['ref_ckpt'] == spec['reference'] and
          digest(Path(spec['reference']) / 'agent.pkl') == spec['reference_sha256'], 'Wrong evaluation reference')
  require(cfg['run']['from_checkpoint'] == spec['checkpoint'] and
          digest(Path(spec['checkpoint']) / 'agent.pkl') == spec['checkpoint_sha256'], 'Wrong evaluation policy')
  require(cfg['fault']['norm_stats'] == spec['stats'], 'Wrong policy calibration identity')
  if not raw_calibration:
    require(digest(spec['stats']) == spec['stats_sha256'], 'Changed calibration')
    audit_eval(directory, spec['method'], spec['split'], spec['eval_seed'],
               Path(spec['checkpoint']), spec['steps'])
    data = json.loads((directory / 'opportunity_audit.json').read_text())
  else:
    require(cfg['script'] == 'eval_only' and cfg['fault']['log_only'] and
            cfg['run']['envs'] == 1 and not cfg['fault']['norm_stats'], 'Wrong raw calibration eval')
    trace = directory / 'fault_score_trace.jsonl'
    report = check_trace(directory, cfg, {})
    counts, diagnostics = check_diagnostic_rows(iter_jsonl(trace), 'clean')
    scores = list(iter_jsonl(directory / 'scores.jsonl'))
    if not scores and spec['steps'] < 30000:
      result = dict(steps=report['rows'], episodes=0, task_return=None)
    else:
      result, _ = summarize_trace(iter_jsonl(trace), scores, 'clean', report['rows'])
    data = dict(audit=report, result=dict(result, **diagnostics), subtypes=[],
                checkpoint=spec['checkpoint'], trace_sha256=digest(trace))
  data['identity'] = spec
  data['result'].update({k: spec[k] for k in (
      'prior', 'method', 'adapt_seed', 'eval_seed', 'split', 'evaluation_round', 'adaptation_profile')})
  for row in data['subtypes']:
    row.update({k: spec[k] for k in ('prior', 'adapt_seed', 'eval_seed', 'evaluation_round', 'adaptation_profile')})
  write_json(directory / 'replication_eval_audit.json', data)
  return [directory / 'replication_eval_audit.json', directory / 'config.yaml']


class Jobs:
  def __init__(self, root):
    self.root = root
    self.jobs = root / 'jobs'
    self.jobs.mkdir(parents=True, exist_ok=True)

  def event(self, key, status):
    with (self.root / 'status.tsv').open('a') as handle:
      handle.write(f'{stamp()}\t{key}\t{status}\n')
    print(f'{stamp()} {key}: {status}', flush=True)

  def run(self, key, command, env, outputs, audit, timeout=7200):
    directory = self.jobs / key
    directory.mkdir(exist_ok=True)
    spec = dict(command=list(map(str, command)), env=env, outputs=list(map(str, outputs)), timeout=timeout)
    write_json(directory / 'spec.json', spec, immutable=True)
    done = directory / 'DONE.json'
    if done.exists():
      verify_files(json.loads(done.read_text())['artifacts'])
      self.event(key, 'SKIP_VERIFIED')
      return
    require(shutil.disk_usage(self.root).free >= 60 * 2**30, 'Data disk below 60 GiB')
    require(shutil.disk_usage('/').free >= 10 * 2**30, 'Root disk below 10 GiB')
    for attempt in range(len(list(directory.glob('attempt_*.json'))), 2):
      number = attempt + 1
      record = directory / f'attempt_{number}.json'
      for output in outputs:
        require(output.resolve().is_relative_to(self.root.resolve()), 'Refusing external output')
        if output.exists():
          output.rename(output.with_name(output.name + f'.interrupted_{time.time_ns()}'))
      write_json(record, dict(start=stamp(), status='RUNNING'))
      self.event(key, f'START attempt={number}')
      start = time.monotonic()
      with (directory / f'attempt_{number}.log').open('a') as log:
        process = subprocess.Popen(command, cwd=env['PROJECT_ROOT'], env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
          code = process.wait(timeout=timeout)
        except BaseException:
          os.killpg(process.pid, signal.SIGTERM)
          try:
            process.wait(timeout=90)
          except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
          write_json(record, dict(start=stamp(), status='INTERRUPTED_OR_TIMEOUT'))
          raise
      elapsed = time.monotonic() - start
      if code:
        write_json(record, dict(status='PROCESS_FAILED', code=code, seconds=elapsed, end=stamp()))
        self.event(key, f'PROCESS_FAILED:{code}')
        if number < 2:
          time.sleep(10)
          continue
        raise RuntimeError(f'{key}: process failed twice; inspect {directory}')
      # Audit failures are never hidden by retries or marked complete.
      try:
        artifacts = audit()
        require(artifacts, 'No audit artifacts')
      except BaseException:
        write_json(record, dict(status='AUDIT_FAILED', seconds=elapsed, end=stamp()))
        self.event(key, 'AUDIT_FAILED')
        raise
      write_json(done, dict(artifacts=file_map(artifacts), seconds=elapsed, completed=stamp()))
      write_json(record, dict(status='DONE', seconds=elapsed, end=stamp()))
      self.event(key, 'DONE')
      return
    raise RuntimeError(f'{key}: retry budget exhausted')


class Study:
  def __init__(self, root, smoke=False):
    self.root = root
    self.source = root / 'source'
    self.clean_source = root / 'clean_source'
    self.work = root / ('smoke' if smoke else 'experiment')
    self.work.mkdir(exist_ok=True)
    self.smoke = smoke
    self.budget = budgets(smoke)
    self.jobs = Jobs(root)
    self.prefix = 'smoke_' if smoke else 'full_'
    self.inputs = json.loads((root / 'inputs.json').read_text())
    self.cal0 = json.loads((root / 'p0_calibration.json').read_text())

  def evaluation(self, prior, method, adapt_seed, eval_seed, split, checkpoint,
                 train_root, expected, group, profile, raw=False):
    key = f'{group}_{prior}_{method}_a{adapt_seed}_e{eval_seed}_{split}'
    output = self.work / group / f'policy_{adapt_seed}'
    directory = output / f'seed_{eval_seed}/{key}'
    spec = dict(prior=prior, method=method, adapt_seed=adapt_seed, eval_seed=eval_seed,
        split=split, evaluation_round=group, adaptation_profile=profile,
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint / 'agent.pkl'),
        reference=expected['reference'], reference_sha256=expected['reference_sha256'],
        stats='' if raw else expected['stats'], stats_sha256='' if raw else expected['stats_sha256'],
        steps=self.budget['clean' if split == 'clean' else 'fault'])
    env = safe_environment(self.source, train_root, output)
    command = shell_call(self.source, 'run_eval', eval_seed, key, checkpoint, spec['steps'],
        '' if split == 'clean' else f'benchmark_{split}', '.5', '1', spec['stats'],
        'p95', 'latent_reward', 'False', '1.0', 'excess_delta_threshold', '.2', '.5', '2.0')
    self.jobs.run(self.prefix + key, command, env, [directory],
        lambda: check_evaluation(directory, spec, raw), timeout=5400)
    self.summary()
    return directory

  def adaptation(self, prior, seed, variant, train_root, expected, profile, group):
    output = self.work / group
    directory = output / f'seed_{seed}/train_{variant}/train'
    steps = self.budget['probe'] if variant == 'reward_scale_probe' else self.budget['train']
    command = shell_call(self.source, 'run_train_variant', seed, variant, expected['stats'], steps)
    env = safe_environment(self.source, train_root, output, profile, expected)
    key = f'{group}_{prior}_{variant}_a{seed}'
    self.jobs.run(self.prefix + key, command, env, [directory],
        lambda: check_adaptation(output, seed, variant, steps, profile, expected), timeout=8 * 3600)
    checkpoint = latest_checkpoint(directory)
    replay = directory / 'replay'
    if replay.is_dir():
      require(not replay.is_symlink() and replay.resolve().is_relative_to(self.work.resolve()), 'Unsafe replay prune')
      shutil.rmtree(replay)
    return checkpoint

  def four_evals(self, prior, method, seed, checkpoint, train_root, expected, group, profile):
    for split in SPLITS:
      self.evaluation(prior, method, seed, seed, split, checkpoint, train_root, expected, group, profile)

  def p0_policy(self, method, seed):
    return Path(self.inputs['policies'][f'{method}:{seed}']['checkpoint'])

  def p0_repeats(self):
    for seed in ([0] if self.smoke else range(5)):
      for method in ('CleanEval', 'CleanAdaptTaskOnly', 'CleanAdaptFATE', 'CleanAdaptSparseSurprise'):
        for split in ('holdout', 'sparse'):
          self.evaluation('P0', method, seed, REPEAT_OFFSET + seed, split,
              self.p0_policy(method, seed), P0_CLEAN, self.cal0['seeds'][str(seed)],
              'repeat', 'none' if method == 'CleanEval' else 'clean')

  def p0_fault_sparse(self):
    for seed in ([0] if self.smoke else range(5)):
      expected = self.cal0['seeds'][str(seed)]
      checkpoint = self.adaptation('P0', seed, 'sparse_surprise_matched', P0_CLEAN,
          expected, 'benchmark_train', 'fault_adapt')
      self.four_evals('P0', 'FaultAdaptSparseSurprise', seed, checkpoint, P0_CLEAN,
          expected, 'basic', 'benchmark_train')
      for method, policy in [('FaultAdaptSparseSurprise', checkpoint),
                             ('FaultAdaptFATE', self.p0_policy('FATE', seed))]:
        for split in ('holdout', 'sparse'):
          self.evaluation('P0', method, seed, REPEAT_OFFSET + seed, split,
              policy, P0_CLEAN, expected, 'repeat', 'benchmark_train')

  def p1_preparation(self):
    parent = self.work / 'p1'
    train_root = parent / 'clean'
    directory = train_root / 'train'
    env = safe_environment(self.clean_source, train_root, parent)
    env.update(CRAFTAX_FAULT='0', CRAFTAX_FAULT_SAMPLER='0', CRAFTAX_USE_RND='0',
               CRAFTAX_RND_ALPHA='0', CRAFTAX_RND_UPDATE='0')
    command = pretrain_command(self.clean_source, directory, self.budget['pretrain'],
                              self.root / 'pretrain_template.txt')
    self.jobs.run(self.prefix + 'p1_pretrain', command, env, [directory],
        lambda: check_pretrain(directory, self.budget['pretrain']), timeout=14 * 3600)
    checkpoint = latest_checkpoint(directory)
    record = dict(checkpoint=str(checkpoint), agent_sha256=digest(checkpoint / 'agent.pkl'))
    write_json(parent / 'protocol.json', record, immutable=True)
    expected = dict(reference=str(checkpoint), reference_sha256=record['agent_sha256'])
    for seed in range(5):
      evaluation = self.evaluation('P1', 'CleanEval', seed, seed, 'clean', checkpoint,
          train_root, expected, 'calibration_clean', 'none', raw=True)
      stats = parent / f'core/seed_{seed}/craftax_clean_fault_stats.json'
      stats.parent.mkdir(parents=True, exist_ok=True)
      def audit_stats():
        data = json.loads(stats.read_text())
        require(data['fault_score_p95'] > 0 and close(data['fault_score_p95'], data['latent_kl_p95']), 'Invalid raw p95')
        require(self.budget['clean'] <= data['steps'] < self.budget['clean'] + 10, 'Partial p95 calibration')
        return [stats]
      self.jobs.run(self.prefix + f'p1_p95_{seed}', shell_call(self.source, 'make_norm_stats',
          evaluation / 'fault_score_trace.jsonl', stats),
          safe_environment(self.source, train_root, parent), [stats], audit_stats, timeout=600)
    probe_expected = dict(expected, stats=str(parent / 'core/seed_0/craftax_clean_fault_stats.json'))
    probe_expected['stats_sha256'] = digest(probe_expected['stats'])
    self.adaptation('P1', 0, 'reward_scale_probe', train_root, probe_expected, 'clean', 'p1_probe')
    calibration = parent / 'clean_calibration.json'
    probe = self.work / 'p1_probe/seed_0/train_reward_scale_probe/train'
    command = [PYTHON, '-m', 'dreamerv3.fate_clean_novelty', 'calibrate', '--parent', str(parent),
        '--probe', str(probe), '--steps', str(self.budget['probe']), '--out', str(calibration)]
    def audit_calibration():
      data = json.loads(calibration.read_text())
      require(set(data['seeds']) == set(map(str, range(5))), 'Incomplete P1 calibration')
      for row in data['seeds'].values():
        require(row['reference'] == str(checkpoint) and row['reference_sha256'] == record['agent_sha256'],
                'P0 calibration leaked into P1')
        require(digest(row['stats']) == row['stats_sha256'], 'P1 p95 hash mismatch')
      return [calibration]
    self.jobs.run(self.prefix + 'p1_calibrate', command,
        safe_environment(self.source, train_root, parent), [calibration], audit_calibration, timeout=3600)
    return train_root, checkpoint, json.loads(calibration.read_text())

  def p1_controls(self):
    train_root, reference, calibration = self.p1_preparation()
    for seed in ([0] if self.smoke else range(3)):
      expected = calibration['seeds'][str(seed)]
      for split in ('seen', 'holdout', 'sparse'):
        self.evaluation('P1', 'CleanEval', seed, seed, split, reference, train_root, expected, 'basic', 'none')
      for variant, method in METHODS.items():
        checkpoint = self.adaptation('P1', seed, variant, train_root, expected, 'clean', 'p1_adapt')
        self.four_evals('P1', 'CleanAdapt' + method, seed, checkpoint, train_root,
                       expected, 'basic', 'clean')

  def summary(self):
    rows, subtypes = [], []
    for done in sorted(self.jobs.jobs.glob(self.prefix + '*/DONE.json')):
      for path in json.loads(done.read_text())['artifacts']:
        if Path(path).name == 'replication_eval_audit.json':
          data = json.loads(Path(path).read_text())
          rows.append(data['result'])
          subtypes.extend(data['subtypes'])
    output = self.work / 'summary'
    output.mkdir(exist_ok=True)
    groups = {}
    for row in rows:
      key = tuple(row[k] for k in ('prior', 'adaptation_profile', 'evaluation_round', 'method', 'split'))
      groups.setdefault(key, []).append(row)
    means = []
    for key, items in groups.items():
      entry = dict(zip(('prior', 'adaptation_profile', 'evaluation_round', 'method', 'split'), key))
      entry.update(n_evaluations=len(items), n_policy_seeds=(1 if key[3] == 'CleanEval' else
          len({r['adapt_seed'] for r in items})))
      for metric in ('task_return', 'coverage', 'unique_types', 'events_per_10k'):
        values = [r[metric] for r in items if r.get(metric) is not None]
        entry[metric + '_mean'] = statistics.mean(values) if values else None
        entry[metric + '_sd'] = statistics.stdev(values) if len(values) > 1 else None
      hits = [set(r.get('discovered_types', '').split(';')) - {''} for r in items]
      entry['consistent_types'] = len(set.intersection(*hits)) if key[-1] != 'clean' else None
      means.append(entry)
    for row in subtypes:
      row['hit'] = int(row.get('legacy_events', 0) > 0)
    for name, values in [('per_evaluation', rows), ('subtype_opportunities', subtypes),
                         ('group_summary', means)]:
      if not values:
        continue
      temporary = output / f'{name}.csv.tmp'
      with temporary.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in values for k in row)))
        writer.writeheader()
        writer.writerows(values)
      temporary.replace(output / f'{name}.csv')
    write_json(output / 'completion.json', dict(evaluations=len(rows), smoke=self.smoke, updated=stamp()))
    text = ['# Replication results', '',
        'Partial unless the queue status is DONE. Original evaluations are not overwritten or pooled here.',
        'Groups preserve prior, adaptation profile, and evaluation round. SD is descriptive across evaluations.',
        'CleanEval has one policy per prior. Repeated rollouts are not independent trained policies.',
        'Each fault evaluation has its own 60K budget; no cross-rollout subtype union is used as 60K coverage.',
        'Opportunity counts are conditional on the assigned subtype. Monitor bonus is hypothetical, not paid.',
        'The pipeline stops on integrity failure; low performance is not a failure.', '',
        '| Prior | Method | Round | Profile | n | Return | Coverage | Consistent |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for row in means:
      def fmt(value):
        return '-' if value is None else f'{value:.3f}'
      text.append(f"| {row['prior']} | {row['method']} | {row['evaluation_round']} | {row['split']} | "
          f"{row['n_evaluations']} | {fmt(row['task_return_mean'])} | {fmt(row['coverage_mean'])} | {row['consistent_types']} |")
    (output / 'README.md').write_text('\n'.join(text) + '\n')

  def run(self):
    self.p0_repeats()
    self.p0_fault_sparse()
    self.p1_controls()
    self.summary()
    count = json.loads((self.work / 'summary/completion.json').read_text())['evaluations']
    require(count == (36 if self.smoke else 130), f'Incomplete evaluation matrix: {count}')
    self.jobs.event(self.prefix + 'queue', 'DONE')


def prepare(root):
  require(root.resolve().is_relative_to(DATA) and not root.exists(), 'Use a new output directory on data SSD')
  for parent in (PREVIOUS, P0_CLEAN):
    subprocess.run(['sha256sum', '--status', '--check', str(parent / 'source_sha256.txt')],
                   cwd=parent / 'source', check=True)
  root.mkdir(parents=True)
  ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
  for old, new in ((PREVIOUS / 'source', root / 'source'), (P0_CLEAN / 'source', root / 'clean_source')):
    shutil.copytree(old, new, ignore=ignore, symlinks=True)
  local = Path(__file__).resolve().parent
  (root / 'source/dreamerv3').chmod((root / 'source/dreamerv3').stat().st_mode | 0o200)
  for name in ('fate_replication_queue.py', 'test_fate_replication_queue.py'):
    target = root / 'source/dreamerv3' / name
    shutil.copyfile(local / name, target)
  shutil.copyfile(PREVIOUS / 'clean_calibration.json', root / 'p0_calibration.json')
  shutil.copyfile(P0_CLEAN / 'train_command.txt', root / 'pretrain_template.txt')
  manifest = json.loads((PREVIOUS / 'protocol.json').read_text())
  policies = {}
  tracked = [root / 'p0_calibration.json', root / 'pretrain_template.txt']
  for seed in range(5):
    for method in ('CleanEval', 'CleanAdaptTaskOnly', 'CleanAdaptFATE', 'FATE'):
      row = next(x for x in manifest['cases'] if x['method'] == method and x['seed'] == seed and x['split'] == 'clean')
      verify_files({str(Path(row['checkpoint']) / 'agent.pkl'): row['checkpoint_sha256'],
                    row['stats']: row['stats_sha256'], row['source_config']: row['source_config_sha256']})
      policies[f'{method}:{seed}'] = dict(checkpoint=row['checkpoint'])
      tracked.extend([Path(row['checkpoint']) / 'agent.pkl', Path(row['checkpoint']) / 'done',
                      Path(row['source_config']), Path(row['stats'])])
    directory = PREVIOUS / f'clean_novelty/seed_{seed}/train_sparse_surprise_matched/train'
    checkpoint = latest_checkpoint(directory)
    require((directory / 'clean_novelty_audit.json').is_file(), 'Missing original SparseSurprise audit')
    policies[f'CleanAdaptSparseSurprise:{seed}'] = dict(checkpoint=str(checkpoint))
    tracked.extend([checkpoint / 'agent.pkl', checkpoint / 'done', directory / 'clean_novelty_audit.json'])
  write_json(root / 'inputs.json', dict(policies=policies, artifacts=file_map(tracked)))
  write_json(root / 'protocol.json', dict(
      name='P0 replication and P1 initialization control', exploratory=True,
      p0_fault_sparse_seeds=list(range(5)), p1_adaptation_seeds=list(range(3)),
      p1_pretrain_seed=P1_PRETRAIN_SEED, p1_pretrain_steps=1100000,
      adaptation_steps=1000000, repeat_eval_seed_offset=REPEAT_OFFSET,
      clean_steps=30000, fault_eval_steps=60000, p1_probe_steps=200000,
      core_adaptations=14, extra_p0_evaluations=60, full_evaluations=130,
      independent_priors=2, p1_p95_seeds=list(range(5)),
      deadline=None, retries_per_process=1, logical_audit_failure='stop',
      reuse_p0_clean_probe_coefficients_for_fault=True,
      common_eval_monitor='hypothetical FATE; no bonus paid',
      no_outcome_based_selection=True, paused_previous_queue_not_resumed=True,
      note='P1 calibration has five scales for compatibility; adaptations use only seeds 0-2. '
           'Repeated evals are not new policies; do not union 60K results or pool P0/P1 as independent seeds.'))
  tracked_code = [p for top in ('source', 'clean_source') for p in (root / top).rglob('*')
                  if p.is_file() and not p.is_symlink()]
  tracked_code += [root / 'inputs.json', root / 'protocol.json']
  write_json(root / 'frozen_files.json', file_map(tracked_code))
  for path in tracked_code:
    path.chmod(path.stat().st_mode & ~0o222)
  print(f'Prepared, not started: {root}')


def preflight(root):
  verify_files(json.loads((root / 'frozen_files.json').read_text()))
  verify_files(json.loads((root / 'inputs.json').read_text())['artifacts'])
  require(str(Path(__file__).resolve()).startswith(str(root / 'source/') + '/'), 'Run frozen source, not live tree')
  env = safe_environment(root / 'source', P0_CLEAN, root / 'preflight')
  env.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
  suites = ('test_fate_replication_queue', 'test_fate_diagnostic_followup',
            'test_fate_revision_queue', 'test_fate_clean_novelty',
            'test_intrinsic_reward_scale_calibration', 'test_fate_revision_integrity')
  command = [PYTHON, '-m', 'unittest', '-q', *('dreamerv3.' + name for name in suites)]
  (root / 'preflight').mkdir(exist_ok=True)
  with (root / 'preflight/tests.log').open('w') as log:
    subprocess.run(command, cwd=root / 'source', env=env, stdout=log, stderr=subprocess.STDOUT,
                   check=True, timeout=600)
    for module in ('dreamerv3.test_craftax_faults', 'dreamerv3.test_fault_reward_modes'):
      subprocess.run([PYTHON, '-m', module], cwd=root / 'source', env=env,
                     stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
  write_json(root / 'preflight/PASS.json', dict(completed=stamp(), log_sha256=digest(root / 'preflight/tests.log')))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('mode', choices=('prepare', 'preflight', 'smoke', 'run', 'summary'))
  parser.add_argument('--root', type=Path, required=True)
  args = parser.parse_args()
  root = args.root.resolve()
  if args.mode == 'prepare':
    prepare(root)
    return
  with (root / 'queue.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
      if args.mode == 'summary':
        Study(root).summary()
        return
      preflight(root)
      if args.mode == 'preflight':
        return
      Study(root, smoke=True).run()
      if args.mode == 'run':
        Study(root).run()
    except BaseException as error:
      Jobs(root).event('queue', f'STOPPED {type(error).__name__}: {error}')
      write_json(root / 'STOPPED.json', dict(time=stamp(), error=repr(error)))
      raise


if __name__ == '__main__':
  main()

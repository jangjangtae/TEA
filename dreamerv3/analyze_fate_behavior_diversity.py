"""CPU-only descriptive diversity analysis of completed corrected Clean evaluations."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from dreamerv3.analyze_fate_corrected_completed import job_states, save_csv
from dreamerv3.audit_fate_revision_run import read_config, require


CONTEXT_FIELDS = ('inventory_bucket', 'nearby_tile', 'achievement_stage', 'nearby_mob')
METHODS = (
    ('core', 'base_clean_eval', 'CleanEval'),
    ('core', 'eval_taskonly_clean', 'TaskOnly'),
    ('core', 'eval_excess_delta_p95_beta02_clean', 'FATE'),
    ('matched', 'eval_dense_matched_clean', 'DenseSurprise-matched'),
    ('matched', 'eval_rnd_matched_clean', 'RND-matched'),
    ('controls', 'eval_taskonly_clean', 'Clean-adapt TaskOnly'),
    ('controls', 'eval_excess_delta_p95_beta02_clean', 'Clean-adapt FATE'),
)
SCALARS = ('task_return', 'episodes', 'mean_episode_transitions', 'contexts',
           'context_actions', 'transition_patterns', 'context_entropy_bits',
           'action_given_context_entropy_bits', 'mean_achievement_count')


def entropy(counts):
  total = sum(counts.values())
  return -sum((n / total) * math.log2(n / total) for n in counts.values() if n) if total else 0.0


def rarefied_richness(counts, sample):
  """Exact expected distinct categories in a uniform sample without replacement."""
  total = sum(counts.values())
  require(0 <= sample <= total, 'Invalid rarefaction sample')
  if sample == 0:
    return 0.0
  result = 0.0
  for count in counts.values():
    if not count:
      continue
    if total - count < sample:
      result += 1.0
    else:
      log_unseen = (math.lgamma(total - count + 1) - math.lgamma(total - count - sample + 1)
                    - math.lgamma(total + 1) + math.lgamma(total - sample + 1))
      result += -math.expm1(min(0.0, log_unseen))
  return result


def context(row):
  return tuple(int(row[key]) for key in CONTEXT_FIELDS)


def collect(rows):
  contexts, joints, edges, post_joints = Counter(), Counter(), Counter(), set()
  achievements = Counter()
  returns, lengths, achievement_counts = [], [], []
  previous = previous_bug = None
  count = resets = episode_length = 0
  episode_return = 0.0
  last_global = 0
  reported_post = 0
  tile_values = set()
  for row, bug in rows:
    require(row is not None and bug is not None, 'Trace lengths disagree')
    count += 1
    require(row['global_step'] == bug['global_step'] == last_global + 1, 'Unaligned or gapped traces')
    last_global = row['global_step']
    require(row['worker'] == bug['worker'] == 0, 'Expected single-env evaluation')
    require(row['episode_id'] == bug['episode_index'], 'Episode IDs disagree')
    require(row['is_first'] == bug['is_first'] and row['is_last'] == bug['is_last'], 'Boundary mismatch')
    require(not row['bug_triggered'] and not bug['log/fault_exists_episode'], 'Not fault-free Clean')
    require(row['rnd_reward_bonus'] == 0 and bug['log/rnd_update_enabled'] == 0, 'Active RND in evaluation')
    require(math.isfinite(row['learning_reward']) and math.isclose(
        row['learning_reward'], row['task_reward_delivered'], abs_tol=2e-6), 'Shaped/nonfinite evaluation')
    require(all(row[key] == bug[f'log/context_{key}'] for key in CONTEXT_FIELDS), 'Context mismatch')
    tile_values.add(row['unique_tile_coverage_cumulative'])
    if row['is_first']:
      require(previous is None or previous['is_last'], 'Reset before terminal boundary')
      resets += 1
      episode_length, episode_return = 0, 0.0
    else:
      require(previous is not None and not previous['is_last'], 'Missing preceding state')
      require(row['episode_id'] == previous['episode_id'], 'Cross-episode transition')
      # fault_score_trace.action is the incoming requested action, not the
      # outgoing policy action recorded in bug_trace at the same observation.
      action = int(row['action'])
      require(action == int(bug['log/requested_action']) == int(bug['log/env_action'])
              == int(previous_bug['action']), 'Action/state alignment error')
      before, after = context(previous), context(row)
      contexts[before] += 1
      joints[before, action] += 1
      edges[before, action, after] += 1
      post_joints.add((action,) + after)
      reported_post = row['semantic_context_coverage_cumulative']
      episode_length += 1
      episode_return += row['task_reward_delivered']
    if row['is_last']:
      returns.append(episode_return)
      lengths.append(episode_length)
      achievement_counts.append(int(bug['log/achievement_count']))
      for key, value in bug.items():
        if key.startswith('log/achievement_') and key != 'log/achievement_count':
          achievements[key.removeprefix('log/achievement_')] += int(value > 0)
    previous, previous_bug = row, bug
  require(returns, 'No completed episodes')
  require(len(post_joints) == reported_post, 'Logged context coverage does not match raw rows')
  summary = dict(steps=count, non_reset_steps=count - resets, episodes=len(returns),
      task_return=statistics.mean(returns), mean_episode_transitions=statistics.mean(lengths),
      contexts=len(contexts), context_actions=len(joints), transition_patterns=len(edges),
      context_entropy_bits=entropy(contexts),
      action_given_context_entropy_bits=entropy(joints) - entropy(contexts),
      mean_achievement_count=statistics.mean(achievement_counts),
      logged_action_post_context_count=len(post_joints),
      tile_field_all_zero=tile_values == {0})
  return dict(summary=summary, contexts=contexts, joints=joints, edges=edges,
              returns=returns, achievements=achievements)


def read_clean(root, seed, name, method):
  path = root / f'seed_{seed}' / name
  audit_path = root / f'job_audits/seed_{seed}_{name}.json'
  audit = json.loads(audit_path.read_text())
  cfg = read_config(path / 'config.yaml')
  require(cfg['script'] == 'eval_only' and cfg['run']['envs'] == 1 and cfg['run']['steps'] == 30000,
          'Wrong evaluation protocol')
  require(cfg['seed'] == seed and cfg['fault']['log_only']
          and cfg['fault']['task_reward_source'] == 'delivered', 'Wrong seed/reward protocol')
  hashes = [hashlib.sha256(), hashlib.sha256()]

  def rows():
    with (path / 'fault_score_trace.jsonl').open('rb') as left, (path / 'bug_trace.jsonl').open('rb') as right:
      for a, b in itertools.zip_longest(left, right):
        require(a is not None and b is not None, 'Trace file lengths disagree')
        hashes[0].update(a)
        hashes[1].update(b)
        yield json.loads(a), json.loads(b)

  result = collect(rows())
  summary = result['summary']
  require(summary['steps'] == audit['rows'] == 30000, 'Partial evaluation')
  require(summary['non_reset_steps'] == audit['non_reset_transitions'], 'Audit count mismatch')
  scores = [json.loads(line)['episode/score'] for line in (path / 'scores.jsonl').read_text().splitlines()]
  require(len(scores) == len(result['returns']) and all(math.isclose(a, b, abs_tol=2e-5)
      for a, b in zip(scores, result['returns'])), 'Reconstructed returns disagree with scores')
  summary.update(method=method, seed=seed, path=str(path))
  result['provenance'] = dict(method=method, seed=seed, path=str(path),
      config_sha256=hashlib.sha256((path / 'config.yaml').read_bytes()).hexdigest(),
      fault_trace_sha256=hashes[0].hexdigest(), bug_trace_sha256=hashes[1].hexdigest(),
      audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
  return result


def stage_counts(data, stage, kind):
  return Counter({key: value for key, value in data[kind].items()
                  if (key if kind == 'contexts' else key[0])[2] == stage})


def stage_analysis(selected):
  rows = []
  for seed in sorted({item['summary']['seed'] for item in selected}):
    items = [item for item in selected if item['summary']['seed'] == seed]
    stages = sorted({ctx[2] for item in items for ctx in item['contexts']})
    for stage in stages:
      effort = min(1000, min(sum(stage_counts(item, stage, 'contexts').values()) for item in items))
      for item in items:
        ctx = stage_counts(item, stage, 'contexts')
        joint = stage_counts(item, stage, 'joints')
        edge = stage_counts(item, stage, 'edges')
        rows.append(dict(method=item['summary']['method'], seed=seed, stage=stage,
            visits=sum(ctx.values()), unique_contexts=len(ctx), context_actions=len(joint),
            matched_draws=effort, shared_support=effort >= 100,
            expected_contexts=rarefied_richness(ctx, effort) if effort >= 100 else None,
            expected_context_actions=rarefied_richness(joint, effort) if effort >= 100 else None,
            expected_transition_patterns=rarefied_richness(edge, effort) if effort >= 100 else None))
  return rows


def common_context_comparison(left, right):
  common = [ctx for ctx, count in left['contexts'].items()
            if count >= 20 and right['contexts'][ctx] >= 20]
  def stats(item):
    distributions = [Counter({action: item['joints'][ctx, action] for action in range(17)}) for ctx in common]
    return (statistics.mean(entropy(x) for x in distributions),
            statistics.mean(rarefied_richness(x, 20) for x in distributions)) if common else (None, None)
  lh, la = stats(left)
  rh, ra = stats(right)
  return dict(method=left['summary']['method'], baseline=right['summary']['method'],
      seed=left['summary']['seed'], common_contexts=len(common),
      method_support_fraction=sum(left['contexts'][c] for c in common) / sum(left['contexts'].values()),
      baseline_support_fraction=sum(right['contexts'][c] for c in common) / sum(right['contexts'].values()),
      entropy_difference=lh - rh if common else None,
      expected_actions_at_20_difference=la - ra if common else None)


def aggregate(rows, metrics, group_keys=('method',)):
  groups = defaultdict(list)
  for row in rows:
    groups[tuple(row[k] for k in group_keys)].append(row)
  output = []
  for key, items in groups.items():
    row = dict(zip(group_keys, key))
    row.update(n=len(items), seeds=';'.join(str(x['seed']) for x in items))
    for metric in metrics:
      values = [x[metric] for x in items if x[metric] is not None]
      row[f'{metric}_n'] = len(values)
      row[f'{metric}_mean'] = statistics.mean(values) if values else None
      row[f'{metric}_sd'] = statistics.stdev(values) if len(values) > 1 else None
    output.append(row)
  return output


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--parent', type=Path, required=True)
  parser.add_argument('--controls', type=Path, required=True)
  parser.add_argument('--completed-results', type=Path, required=True)
  parser.add_argument('--cohort', choices=('all', 'main'), default='all',
                      help='Use main to analyze completed reward-matched baselines without partial clean controls.')
  parser.add_argument('--out', type=Path, required=True)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)
  roots = dict(core=args.parent / 'core', matched=args.parent / 'matched', controls=args.controls / 'controls')
  states = {key: job_states(root) for key, root in roots.items()}
  methods = tuple(x for x in METHODS if args.cohort == 'all' or x[0] != 'controls')
  # Freeze eligible jobs before parsing; later completions are not mixed in.
  cases = [(stage, name, method, seed) for stage, name, method in methods for seed in range(5)
           if states[stage].get((seed, name)) == 'DONE']
  data = []
  for stage, name, method, seed in cases:
    if not name.startswith('base_'):
      require(states[stage].get((seed, 'train_' + name[5:-6])) == 'DONE', 'Incomplete training')
    data.append(read_clean(roots[stage], seed, name, method))
    print(f'Checked {method} seed={seed}', flush=True)
  index = {(x['summary']['method'], x['summary']['seed']): x for x in data}
  matched_seeds = sorted(set.intersection(*({seed for method, seed in index if method == m} for _, _, m in methods)))
  require(matched_seeds, 'No complete matched Clean cohort')
  matched = [x for x in data if x['summary']['seed'] in matched_seeds]
  all_rows, matched_rows = [x['summary'] for x in data], [x['summary'] for x in matched]
  stage_rows = stage_analysis(matched)
  paired = []
  for method, baseline in (
      ('Clean-adapt FATE', 'Clean-adapt TaskOnly'), ('Clean-adapt FATE', 'CleanEval'),
      ('Clean-adapt FATE', 'FATE'), ('FATE', 'TaskOnly'),
      ('FATE', 'DenseSurprise-matched'), ('FATE', 'RND-matched'),
      ('FATE', 'CleanEval'), ('DenseSurprise-matched', 'TaskOnly'),
      ('RND-matched', 'TaskOnly'), ('DenseSurprise-matched', 'CleanEval'),
      ('RND-matched', 'CleanEval')):
    for seed in matched_seeds:
      if (method, seed) in index and (baseline, seed) in index:
        paired.append(common_context_comparison(index[method, seed], index[baseline, seed]))
  achievement_rows = [dict(method=x['summary']['method'], seed=x['summary']['seed'],
      achievement=key, completed_episodes=x['summary']['episodes'], hit_episodes=count,
      rate=count / x['summary']['episodes']) for x in data for key, count in x['achievements'].items()]
  with args.completed_results.open() as handle:
    downstream = [row for row in csv.DictReader(handle) if row['split'] != 'clean']
  joined = []
  seen = set()
  for row in downstream:
    key = row['method'], int(row['seed'])
    if key not in index:
      continue
    identity = key + (row['split'],)
    require(identity not in seen, 'Duplicate downstream case')
    seen.add(identity)
    cfg_path = Path(row['path']) / 'config.yaml'
    require(hashlib.sha256(cfg_path.read_bytes()).hexdigest() == row['config_sha256'], 'Downstream config changed')
    require(Path(row['audit_path']).is_file(), 'Missing downstream audit')
    clean = index[key]['summary']
    joined.append(dict(method=key[0], seed=key[1], split=row['split'],
        clean_return=clean['task_return'], clean_contexts=clean['contexts'],
        clean_context_actions=clean['context_actions'],
        fault_return=float(row['task_return']), fault_coverage=float(row['coverage']),
        fault_events=int(row['event_count']), fault_types=int(row['unique_types'])))
  summary = aggregate(matched_rows, SCALARS)
  stage_summary = aggregate(stage_rows, ('visits', 'expected_contexts', 'expected_context_actions',
                                       'expected_transition_patterns'), ('method', 'stage'))
  for name, rows in (
      ('clean_per_seed', all_rows), ('clean_all_available_summary', aggregate(all_rows, SCALARS)),
      ('clean_matched_summary', summary), ('stage_matched_per_seed', stage_rows),
      ('stage_matched_summary', stage_summary), ('common_context_paired', paired),
      ('achievement_per_seed', achievement_rows), ('clean_to_fault_per_seed', joined)):
    save_csv(args.out / f'{name}.csv', rows)
  notes = ['# Corrected FATE behavior diversity: exploratory log analysis', '',
      f'Created: {datetime.now().astimezone().isoformat()}',
      f'Cohort: {args.cohort}.',
      f'Matched Clean seeds: {matched_seeds}. Each evaluation has 30,000 driver callbacks.',
      'Incomplete jobs are excluded. One shared clean checkpoint, not independent pretraining seeds.', '',
      '## Matched Clean results', '',
      '| Method | n | Return | Contexts | Context-actions | Transition patterns | H(A|C), bits |',
      '|---|---:|---:|---:|---:|---:|---:|']
  for row in summary:
    notes.append(f"| {row['method']} | {row['n']} | {row['task_return_mean']:.3f} | "
        f"{row['contexts_mean']:.1f} | {row['context_actions_mean']:.1f} | "
        f"{row['transition_patterns_mean']:.1f} | {row['action_given_context_entropy_bits_mean']:.3f} |")
  notes += ['', '## Definitions and limitations', '',
      '- C = (inventory bucket, front tile, achievement stage, nearby-mob flag). These are coarse contexts, not full states.',
      '- Decision pairs use the preceding context and the incoming requested action. Every action is checked against the preceding outgoing policy action and the executed Clean action.',
      '- Reset callbacks are excluded, first actions and terminal transitions retained, and no transitions cross episode boundaries.',
      '- The old logged semantic coverage includes incoming action plus POST-transition context; it is audited but not mislabeled as state coverage.',
      '- Entropy is empirical visitation/conditional-action entropy, not actor-distribution entropy. Random, ineffective actions can increase it.',
      '- Stage matching uses exact expected richness under uniform sampling without replacement, with the same sample count for all included methods within each seed/stage: min(1000, smallest visit count). Counts below 100 are unsupported, not zero results.',
      '- Rarefaction standardizes observational effort, not temporal dependence. No iid-step confidence interval is used.',
      '- Common-context comparisons require >=20 visits in each policy and average contexts equally. Action richness is matched to 20 draws. This conditions on shared support and omits newly reached contexts.',
      '- Return and achievement rates use completed episodes only; diversity uses all observed non-reset transitions, including an unfinished final episode.',
      '- CSV dispersion is sample SD across seeds. Partial clean-control results are not mixed with main n=5 means.',
      '- Fault links come from the specified earlier audited-results CSV; missing newer fault evaluations stay missing. They show associations, not a causal mediation effect.',
      '- Tile coverage is not measured here. The legacy tile-count field is a zero placeholder.',
      '- No new training or evaluation was run. No queue or frozen experiment source was modified.',
      '- Clean-adapt RND/SparseSurprise, benign build changes, and imperfect references are not covered by these logs.', '']
  (args.out / 'README.md').write_text('\n'.join(notes))
  manifest = dict(created=datetime.now().astimezone().isoformat(), matched_clean_seeds=matched_seeds,
      cohort=args.cohort,
      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      completed_results=str(args.completed_results),
      completed_results_sha256=hashlib.sha256(args.completed_results.read_bytes()).hexdigest(),
      minimum_stage_visits=100, maximum_stage_draws=1000, common_context_draws=20,
      files=[x['provenance'] for x in data])
  (args.out / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
  print(f'SAVED {args.out}', flush=True)


if __name__ == '__main__':
  main()

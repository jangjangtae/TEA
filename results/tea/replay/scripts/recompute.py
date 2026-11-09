"""Reproduce nested summaries and exploratory paired intervals using only stdlib."""

import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import statistics as st


METHODS = (
    'FaultAdaptTaskOnly', 'FaultAdaptFATE', 'FaultAdaptSparseSurprise',
    'FaultAdaptSparseMagnitude072', 'FaultRestoredTaskOnly', 'FaultRestoredTEA')
METRICS = ('task_return', 'coverage', 'events_per_10k', 'unique_types',
           'contexts', 'context_actions', 'transition_patterns', 'mean_achievement_count')
T975_DF4 = 2.7764451051977987
CONTRASTS = {
    'fresh_TEA_minus_TaskOnly': {'FaultAdaptFATE': 1, 'FaultAdaptTaskOnly': -1},
    'restored_TEA_minus_TaskOnly': {'FaultRestoredTEA': 1, 'FaultRestoredTaskOnly': -1},
    'TaskOnly_restored_minus_fresh': {'FaultRestoredTaskOnly': 1, 'FaultAdaptTaskOnly': -1},
    'TEA_restored_minus_fresh': {'FaultRestoredTEA': 1, 'FaultAdaptFATE': -1},
    'fresh_TEA_minus_SparseSurprise': {'FaultAdaptFATE': 1, 'FaultAdaptSparseSurprise': -1},
    'fresh_TEA_minus_SparseMagnitude072': {'FaultAdaptFATE': 1, 'FaultAdaptSparseMagnitude072': -1},
    'interaction_restored_gap_minus_fresh_gap': {
        'FaultRestoredTEA': 1, 'FaultRestoredTaskOnly': -1,
        'FaultAdaptFATE': -1, 'FaultAdaptTaskOnly': 1},
}


def require(test, message):
    if not test:
        raise ValueError(message)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = defaultdict(list)
    require(len(rows) == 270, 'Expected 270 evaluation records')
    for row in rows:
        groups[row['method'], row['split'], int(row['adapt_seed'])].append(row)
    expected = {(m, p, s) for m in METHODS for p in ('clean', 'holdout', 'sparse') for s in range(5)}
    require(set(groups) == expected, 'Unexpected or missing method/profile/seed')
    policies, summary = [], []
    for method in METHODS:
        for split in ('clean', 'holdout', 'sparse'):
            for seed in range(5):
                block = groups[method, split, seed]
                require(sorted(int(r['eval_seed']) for r in block) == [seed, seed+10000, seed+20000],
                        'Missing or duplicate repeat')
                require(all(int(r['steps']) == (30000 if split == 'clean' else 60000) for r in block),
                        'Unexpected evaluation budget')
                result = dict(method=method, split=split, adapt_seed=seed, repeats=3)
                for metric in METRICS:
                    vals = [float(r[metric]) for r in block if r.get(metric) not in ('', None)]
                    if vals:
                        require(len(vals) == 3 and all(math.isfinite(x) for x in vals),
                                f'Incomplete or nonfinite metric: {metric}')
                        result[metric] = st.mean(vals)
                policies.append(result)
            result = dict(method=method, split=split, adaptation_seeds=5, repeats_per_policy=3)
            for metric in METRICS:
                vals = [r[metric] for r in policies[-5:] if metric in r]
                if vals:
                    require(len(vals) == 5, 'Missing policy metric')
                    result[metric+'_mean'] = st.mean(vals)
                    result[metric+'_sd'] = st.stdev(vals)
            summary.append(result)
    return policies, summary


def paired_interval(differences):
    require(len(differences) == 5 and all(math.isfinite(x) for x in differences),
            'Five finite policy-seed differences are required')
    mean, sd = st.mean(differences), st.stdev(differences)
    se = sd / math.sqrt(5)
    half = T975_DF4 * se
    # Enumerate the full empirical paired-bootstrap distribution, not new data.
    boot = sorted(st.mean(sample) for sample in itertools.product(differences, repeat=5))

    def quantile(p):
        pos = p * (len(boot)-1)
        lo = int(pos)
        return boot[lo] + (pos-lo)*(boot[min(lo+1, len(boot)-1)]-boot[lo])

    return dict(n_policy_pairs=5, degrees_of_freedom=4, mean_difference=mean,
                sd_difference=sd, standard_error=se, t_critical=T975_DF4,
                t95_low=mean-half, t95_high=mean+half,
                bootstrap95_low=quantile(.025), bootstrap95_high=quantile(.975),
                wins=sum(x > 1e-10 for x in differences),
                ties=sum(abs(x) <= 1e-10 for x in differences),
                losses=sum(x < -1e-10 for x in differences))


def effects(policies):
    lookup = {(r['method'], r['split'], r['adapt_seed']): r for r in policies}
    output, per_seed = [], []
    measurements = [('clean', k) for k in (
        'task_return', 'contexts', 'context_actions', 'transition_patterns', 'mean_achievement_count')]
    measurements += [(p, k) for p in ('holdout', 'sparse') for k in (
        'task_return', 'coverage', 'events_per_10k', 'unique_types')]
    for name, weights in CONTRASTS.items():
        formula = ' + '.join(f'({weight})*{method}' for method, weight in weights.items())
        for split, metric in measurements:
            scale = 100 if metric == 'coverage' else 1
            unit = 'percentage_points' if metric == 'coverage' else 'original_metric_units'
            differences = []
            for seed in range(5):
                values = {m: lookup[m, split, seed][metric] for m in weights}
                difference = sum(weights[m]*v for m, v in values.items()) * scale
                differences.append(difference)
                per_seed.append(dict(contrast=name, split=split, metric=metric, unit=unit,
                                     adapt_seed=seed, difference=difference,
                                     component_values_original_units=json.dumps(values, sort_keys=True)))
            output.append(dict(contrast=name, formula=formula, split=split, metric=metric, unit=unit,
                               **paired_interval(differences)))
    return output, per_seed


def validate_against(generated, original, keys):
    left = {tuple(str(r[k]) for k in keys): r for r in generated}
    right = {tuple(r[k] for k in keys): r for r in original}
    require(len(left) == len(generated) and len(right) == len(original), 'Duplicate aggregate row')
    require(set(left) == set(right), 'Aggregate row mismatch')
    checks = 0
    for key, row in left.items():
        for field in set(row) | set(right[key]):
            if field in keys:
                continue
            a, b = row.get(field), right[key].get(field)
            if a in ('', None) and b in ('', None):
                continue
            require(a not in ('', None) and b not in ('', None), f'Missing value: {key}/{field}')
            require(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10),
                    f'Value mismatch: {key}/{field}: {a} vs {b}')
            checks += 1
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    results = args.bundle/'results'
    output = args.out or args.bundle/'statistics'
    rows = read_csv(results/'evaluations.csv')
    policies, summary = aggregate(rows)
    checks = validate_against(policies, read_csv(results/'per_policy.csv'), ('method', 'split', 'adapt_seed'))
    checks += validate_against(summary, read_csv(results/'summary.csv'), ('method', 'split'))
    contrasts, differences = effects(policies)
    write_csv(output/'paired_effects.csv', contrasts)
    write_csv(output/'paired_seed_differences.csv', differences)
    write_csv(output/'recomputed_per_policy.csv', policies)
    write_csv(output/'recomputed_summary.csv', summary)
    report = dict(passed=True, evaluations=len(rows), policy_profile_rows=len(policies),
                  summary_rows=len(summary), aggregate_values_verified=checks,
                  contrast_metric_rows=len(contrasts), seed_difference_rows=len(differences),
                  inference_unit='five paired adaptation seeds after averaging three evaluations',
                  primary_interval='two-sided pointwise paired t, df=4, no multiplicity correction',
                  bootstrap='all 5**5=3125 empirical resamples; linearly interpolated percentile limits; sensitivity only',
                  scope='P0 Fault adaptation only; not five independent pretrained checkpoints')
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

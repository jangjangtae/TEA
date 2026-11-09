import math
import unittest

from recompute import METHODS, aggregate, effects, paired_interval


class RecomputeTests(unittest.TestCase):
    def rows(self):
        return [dict(method=m, split=p, adapt_seed=str(s), eval_seed=str(s+10000*r),
                     steps=30000 if p == 'clean' else 60000, task_return=s+r)
                for m in METHODS for p in ('clean', 'holdout', 'sparse')
                for s in range(5) for r in range(3)]

    def test_policy_first_aggregation(self):
        policies, summary = aggregate(self.rows())
        self.assertEqual(len(policies), 90)
        self.assertEqual(policies[0]['task_return'], 1)
        self.assertEqual(summary[0]['task_return_mean'], 3)
        self.assertAlmostEqual(summary[0]['task_return_sd'], math.sqrt(2.5))

    def test_missing_or_duplicate_repeat_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            aggregate(rows[:-1])
        rows[0]['eval_seed'] = rows[1]['eval_seed']
        with self.assertRaises(ValueError):
            aggregate(rows)

    def test_constant_difference_and_zero(self):
        x = paired_interval([2]*5)
        self.assertEqual((x['t95_low'], x['t95_high'], x['wins']), (2, 2, 5))
        self.assertEqual(paired_interval([0]*5)['ties'], 5)

    def test_known_t_interval(self):
        x = paired_interval([1, 2, 3, 4, 5])
        self.assertAlmostEqual(x['t95_low'], 1.036756838522439)
        self.assertAlmostEqual(x['t95_high'], 4.963243161477561)

    def test_invalid_inference_unit(self):
        with self.assertRaises(ValueError):
            paired_interval([1]*15)
        with self.assertRaises(ValueError):
            paired_interval([1, 2, 3, 4, float('nan')])

    def test_interaction_and_percentage_points(self):
        policies = []
        means = dict(zip(METHODS, [1, 3, 1, 1, 2, 3]))
        for method in METHODS:
            for split in ('clean', 'holdout', 'sparse'):
                for seed in range(5):
                    row = dict(method=method, split=split, adapt_seed=seed)
                    row.update({k: means[method] for k in (
                        'task_return', 'contexts', 'context_actions', 'transition_patterns',
                        'mean_achievement_count', 'events_per_10k', 'unique_types')})
                    row['coverage'] = means[method]/10
                    policies.append(row)
        summaries, _ = effects(policies)
        row = next(r for r in summaries if r['contrast'].startswith('interaction')
                   and r['split'] == 'sparse' and r['metric'] == 'coverage')
        self.assertAlmostEqual(row['mean_difference'], -10)
        self.assertEqual(row['unit'], 'percentage_points')


if __name__ == '__main__':
    unittest.main()

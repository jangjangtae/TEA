"""Protect nested evaluation aggregation and paired publication intervals."""

import copy
import math
import statistics as st
import unittest

from dreamerv3.plot_tea_final_paper import aggregate_evaluations, paired_interval, T95


def records(core=False):
  rows=[]
  for split in ('clean','holdout','sparse'):
    for seed in range(5):
      repeats=1 if core and split=='clean' else 3
      for rnd in range(repeats):
        row=dict(method='A',split=split,adapt_seed=str(seed),eval_seed=str(seed+10000*rnd),
            steps='30000' if split=='clean' else '60000',task_return=str(seed+2*rnd),
            context_actions=str(100+seed+rnd) if split=='clean' else '',
            coverage='' if split=='clean' else str((seed+rnd)/7),
            unique_types='0' if split=='clean' else str(seed+rnd))
        rows.append(row)
  return rows


class FinalPaperDataTest(unittest.TestCase):
  def test_repetitions_are_averaged_within_five_policies(self):
    output=aggregate_evaluations(records(),['A'])
    values=[output['A','clean',seed]['task_return'] for seed in range(5)]
    self.assertEqual(values,[2,3,4,5,6])
    self.assertEqual(len(output),15)
    self.assertEqual(st.stdev(values),math.sqrt(2.5))

  def test_core_and_additional_clean_budgets_are_not_mixed(self):
    rows=records(True)
    output=aggregate_evaluations(rows,['A'],True)
    self.assertEqual(output['A','clean',0]['repeats'],1)
    self.assertEqual(output['A','sparse',0]['repeats'],3)
    with self.assertRaisesRegex(ValueError,'repeat count'):
      aggregate_evaluations(rows,['A'])

  def test_duplicate_or_missing_evaluation_is_rejected(self):
    rows=records()
    rows[1]['eval_seed']=rows[0]['eval_seed']
    with self.assertRaisesRegex(ValueError,'evaluation seeds'):
      aggregate_evaluations(rows,['A'])
    with self.assertRaisesRegex(ValueError,'repeat count'):
      aggregate_evaluations(records()[:-1],['A'])

  def test_nonfinite_partial_and_wrong_budget_are_rejected(self):
    for key,value in [('task_return','nan'),('task_return',''),('steps','10000')]:
      rows=records()
      rows[1][key]=value
      with self.subTest(key=key,value=value),self.assertRaises(ValueError):
        aggregate_evaluations(rows,['A'])

  def test_seven_subtype_denominator_is_retained(self):
    rows=records()
    for row in rows:
      if row['split']!='clean':
        row['coverage']=str(float(row['unique_types'])/5)
    with self.assertRaisesRegex(ValueError,'seven'):
      aggregate_evaluations(rows,['A'])

  def test_coverage_is_mean_not_union(self):
    output=aggregate_evaluations(records(),['A'])
    self.assertAlmostEqual(output['A','sparse',0]['coverage'],1/7)

  def test_paired_t_interval_uses_five_pairs(self):
    mean,sd,lower,upper=paired_interval([-1,0,1,2,3])
    self.assertEqual(mean,1)
    self.assertAlmostEqual(sd,math.sqrt(2.5))
    self.assertAlmostEqual(upper-mean,T95*sd/math.sqrt(5))
    self.assertAlmostEqual(mean-lower,upper-mean)
    with self.assertRaises(ValueError):
      paired_interval([0]*15)

  def test_zero_difference_is_valid(self):
    self.assertEqual(paired_interval([0]*5),(0,0.,0.,0.))


if __name__=='__main__':
  unittest.main()

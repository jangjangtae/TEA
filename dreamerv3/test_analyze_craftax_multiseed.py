import math
import unittest

from dreamerv3 import analyze_craftax_multiseed as analysis


class CoverageMetricsTest(unittest.TestCase):

  def test_bug_events_are_not_unique_bug_coverage(self):
    rows = [
        {
            "bug_triggered": 1,
            "bug_type": "delay_after_success",
            "unique_bug_count_cumulative": 1,
            "episode_id": 1,
        },
        {
            "bug_triggered": 1,
            "bug_type": "delay_after_success",
            "unique_bug_count_cumulative": 2,
            "episode_id": 1,
        },
        {
            "bug_triggered": 1,
            "bug_type": "sticky_after_repeat_switch",
            "unique_bug_count_cumulative": 3,
            "episode_id": 2,
        },
    ]
    result = analysis.bug_discovery(rows, analysis.binary(rows, "bug_triggered"), "seen")
    self.assertEqual(result["event_count"], 3)
    self.assertEqual(result["types"], {
        "delay_after_success", "sticky_after_repeat_switch"})
    self.assertEqual(result["coverage"], 2 / 8)
    # Cumulative unique types after each row: 1, 1, 2. The discovery AUC is
    # their time average, normalized by the 8 expected seen bugs.
    self.assertAlmostEqual(result["auc"], 4 / 3)
    self.assertAlmostEqual(result["auc_norm"], (4 / 3) / 8)

  def test_empty_bug_type_does_not_inflate_unique_coverage(self):
    rows = [
        {"bug_triggered": 1, "bug_type": "", "episode_id": 1},
        {
            "bug_triggered": 1,
            "bug_type": "delay_after_success",
            "episode_id": 1,
        },
    ]
    result = analysis.bug_discovery(rows, analysis.binary(rows, "bug_triggered"), "seen")
    self.assertEqual(result["event_count"], 2)
    self.assertEqual(result["types"], {"delay_after_success"})
    self.assertEqual(result["coverage"], 1 / 8)
    self.assertAlmostEqual(result["auc"], 0.5)

  def test_missing_craftax_context_is_not_reported_as_zero(self):
    result = analysis.context_diversity([
        {"action": 1, "unique_tile_coverage_cumulative": 0},
        {"action": 2, "unique_tile_coverage_cumulative": 0},
    ])
    self.assertTrue(math.isnan(result["coverage"]))
    self.assertTrue(math.isnan(result["unique_suspicious"]))

  def test_no_bug_is_right_censored(self):
    rows = [
        {"bug_triggered": 0, "episode_id": 1},
        {"bug_triggered": 0, "episode_id": 2},
    ]
    result = analysis.bug_discovery(rows, analysis.binary(rows, "bug_triggered"), "holdout")
    self.assertEqual(result["found"], 0)
    self.assertEqual(result["first_step"], 3)
    self.assertEqual(result["first_episode"], 3)


if __name__ == "__main__":
  unittest.main()

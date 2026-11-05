"""Contract checks for the proposed research protocol, not experiment results."""
import json
import unittest
from pathlib import Path


class ResearchProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(Path('configs/rsi-research-protocol.json').read_text())

    def test_preserves_experiment_and_authorization_boundaries(self):
        p = self.protocol
        self.assertEqual(p['status'], 'design_not_run')
        self.assertFalse(p['human_arc_heuristics'])
        self.assertFalse(p['execution_authorized'])
        self.assertEqual(p['minimum_strict_oracles'], 5)
        self.assertEqual(p['max_generations'], 2)

    def test_benchmark_and_matched_control_are_distinct(self):
        evaluation = self.protocol['evaluation']
        self.assertEqual(evaluation['competitive_target_percent'], 26.0)
        self.assertEqual(evaluation['benchmark'], 'ARC-AGI-2 semi-private')
        self.assertTrue(evaluation['independent_score_required'])
        self.assertTrue(evaluation['matched_untrained_base_required'])
        self.assertEqual(evaluation['attempts_per_test_input'], 2)
        self.assertFalse(evaluation['public_eval_used_for_development'])

    def test_budgets_require_fresh_authorization(self):
        budgets = self.protocol['budgets']
        self.assertEqual(budgets['authorized_live_calls'], 0)
        self.assertEqual(budgets['authorized_training_steps'], 0)
        self.assertEqual(budgets['authorized_external_spend_usd'], 0)
        self.assertTrue(budgets['separate_run_caps_required'])


if __name__ == '__main__':
    unittest.main()

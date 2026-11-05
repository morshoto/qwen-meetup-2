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

    def test_proposed_training_budget_is_bounded_but_not_authorized(self):
        training = self.protocol['proposed_training']
        self.assertEqual(training['preflight_max_steps'], 1)
        self.assertEqual(training['max_steps_per_generation'], 20)
        self.assertEqual(training['max_wall_seconds_per_generation'], 1800)
        self.assertEqual(training['batch_size'], 1)
        self.assertEqual(training['lora_rank'], 8)
        self.assertFalse(self.protocol['execution_authorized'])

    def test_recursive_improvement_compares_each_generation_to_its_parent(self):
        evaluation = self.protocol['evaluation']
        self.assertTrue(evaluation['each_generation_compared_to_parent'])
        self.assertEqual(evaluation['recursive_transitions_required'], 2)

    def test_development_manifest_excludes_known_prior_exposure(self):
        split = self.protocol['development_split']
        self.assertEqual(split['known_prior_exposure_registry'],
                         'configs/p3c.exploration.exclusions.v1.json')
        self.assertEqual(split['task_count'], 200)
        self.assertTrue(split['requires_complete_history_audit'])
        self.assertFalse(split['independence_established'])


if __name__ == '__main__':
    unittest.main()

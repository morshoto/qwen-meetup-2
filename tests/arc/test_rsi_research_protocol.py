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

    def test_training_resource_proposal_has_enforceable_stop_limits(self):
        training = self.protocol['proposed_training']
        self.assertEqual(training['preflight_max_wall_seconds'], 300)
        self.assertEqual(training['min_free_disk_gib'], 20)
        self.assertEqual(training['min_system_memory_free_percent'], 20)
        self.assertEqual(training['max_swap_increase_mib'], 512)
        self.assertEqual(training['max_process_tree_rss_gib'], 32)

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

    def test_proposed_generation_pins_model_and_attempt_schedule(self):
        generation = self.protocol['proposed_generation']
        self.assertEqual(generation['model_revision'], 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4')
        self.assertEqual(generation['slots_per_generation'], 10)
        self.assertEqual(generation['max_repairs_per_slot'], 2)
        self.assertEqual(generation['temperature'], 0.2)
        self.assertEqual(generation['max_new_tokens'], 512)

    def test_evaluation_budget_covers_both_arms(self):
        budget = self.protocol['proposed_evaluation_per_transition']
        self.assertEqual(budget['tasks'], 200)
        self.assertEqual(budget['models'], 2)
        self.assertEqual(budget['calls_per_model_task'], 2)
        self.assertEqual(budget['max_calls'], 800)
        self.assertEqual(budget['max_calls'], budget['tasks'] * budget['models'] * budget['calls_per_model_task'])


if __name__ == '__main__':
    unittest.main()

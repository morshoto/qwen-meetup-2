import copy
import unittest

from src.arc.rsi_preflight_review import review_preflight_observation


def sample(stamp, *, free=70):
    return {'monotonic_seconds': stamp, 'disk_free_bytes': 30 * 1024 ** 3,
            'memory_free_percent': free, 'swap_used_mib': 0, 'process_tree_rss_bytes': 0}


class PreflightReviewTests(unittest.TestCase):
    def setUp(self):
        self.baseline = sample(100)
        self.observation = {'stop_reason': 'process_exited', 'exit_code': 0,
            'samples': [sample(101)], 'training_verified': False, 'group_signals_confirmed': True,
            'escaped_processes_excluded': False, 'stream_complete': True, 'elapsed_seconds': 1.1,
            'loss': {'status': 'ok', 'last_step': 1, 'last_loss': 0.125}}

    def review(self, observation=None):
        return review_preflight_observation(baseline=self.baseline,
            observation=self.observation if observation is None else observation)

    def test_finite_one_step_is_observed_but_not_training_eligible(self):
        result = self.review()
        self.assertEqual(result['status'], 'preflight_observation_consistent_unverified')
        self.assertFalse(result['training_allowed'])
        self.assertIn('process_containment_proof', result['remaining_gates'])
        self.assertIn('independent_adapter_reload', result['remaining_gates'])
        self.assertIn('execution_provenance', result['remaining_gates'])

    def test_review_checks_registered_limits_and_elapsed(self):
        limits = {'timeout_seconds': 1, 'minimum_disk_gib': 20, 'minimum_free_memory_percent': 75,
                  'maximum_swap_growth_mib': 512, 'maximum_process_rss_gib': 32, 'sample_interval_seconds': 1}
        result = review_preflight_observation(baseline=self.baseline, observation=self.observation, resource_limits=limits)
        self.assertEqual(result['status'], 'preflight_observation_rejected')
        self.assertFalse(result['training_allowed'])
        invalid = dict(limits, maximum_process_rss_gib=33)
        self.assertEqual(review_preflight_observation(baseline=self.baseline, observation=self.observation,
                         resource_limits=invalid)['status'], 'preflight_observation_rejected')

    def test_bad_exit_loss_stream_or_resource_is_rejected(self):
        changes = [
            {'stop_reason': 'wall_clock_limit'}, {'exit_code': 1}, {'stream_complete': False},
            {'group_signals_confirmed': False}, {'loss': {'status': 'ok', 'last_step': 0, 'last_loss': None}},
            {'loss': {'status': 'ok', 'last_step': 2, 'last_loss': 0.1}},
            {'loss': {'status': 'ok', 'last_step': 1, 'last_loss': float('nan')}},
            {'loss': {'status': 'nonfinite_loss', 'last_step': 0, 'last_loss': None}},
            {'samples': []}, {'samples': [sample(101, free=19)]}, {'samples': [sample(401)]},
            {'elapsed_seconds': 301}, {'elapsed_seconds': True},
        ]
        for change in changes:
            value = copy.deepcopy(self.observation)
            value.update(change)
            with self.subTest(change=change):
                result = self.review(value)
                self.assertEqual(result['status'], 'preflight_observation_rejected')
                self.assertFalse(result['training_allowed'])


if __name__ == '__main__':
    unittest.main()

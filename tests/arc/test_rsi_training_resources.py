"""The proposed training limits must fail closed on missing resource evidence."""
import unittest

from src.arc.rsi_training_resources import training_resource_decision


def sample(t=0, *, disk=30 * 1024**3, free=70, swap=0, rss=8 * 1024**3):
    return {'monotonic_seconds': t, 'disk_free_bytes': disk, 'memory_free_percent': free,
            'swap_used_mib': swap, 'process_tree_rss_bytes': rss}


class TrainingResourceTests(unittest.TestCase):
    def test_valid_samples_stay_within_proposed_limits(self):
        self.assertIsNone(training_resource_decision(sample(), [sample(0), sample(5)]))

    def test_each_registered_threshold_stops(self):
        cases = [(sample(5, disk=20 * 1024**3), 'insufficient_disk'),
                 (sample(5, free=20), 'memory_pressure_limit'),
                 (sample(5, swap=512), 'swap_increase_limit'),
                 (sample(5, rss=32 * 1024**3), 'process_rss_limit')]
        for state, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(training_resource_decision(sample(), [sample(0), state]), reason)

    def test_sustained_swap_growth_stops_before_absolute_limit(self):
        states = [sample(0), sample(5, swap=1), sample(10, swap=2)]
        self.assertEqual(training_resource_decision(sample(), states), 'swap_growth_trend')

    def test_missing_invalid_or_late_monitor_samples_stop(self):
        invalid = [[], [sample(6)], [sample(0), sample(6)],
                   [sample(0), dict(sample(5), process_tree_rss_bytes=None)],
                   [sample(0), dict(sample(5), memory_free_percent=float('nan'))],
                   [sample(0), sample(-1)]]
        for states in invalid:
            with self.subTest(states=states):
                self.assertEqual(training_resource_decision(sample(), states), 'monitor_evidence_invalid')


if __name__ == '__main__':
    unittest.main()

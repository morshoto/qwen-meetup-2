"""Fixture-only integration of conditioned prompts, probes, and SFT drafts."""
import unittest

from src.arc.data import Task
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS
from src.arc.rsi_fixture import run_fixture_slot


class FixtureSlotTests(unittest.TestCase):
    def setUp(self):
        self.task = Task('fixture-parent', (([[1, 2]], [[2, 1]]),), ([[3, 4]],))
        self.public, self.private = _quality_grids(42, 5), _quality_grids(43, 5)

    def test_repair_then_draft_is_fake_and_never_training_eligible(self):
        result = run_fixture_slot(self.task, [FAKE_PROGRAMS[2], FAKE_PROGRAMS[1]],
                                  public_probes=self.public, private_probes=self.private)
        self.assertEqual(result['status'], 'fixture_draft')
        self.assertEqual(len(result['prompts']), 2)
        self.assertIn('visible_demonstration_mismatch', result['prompts'][1])
        self.assertEqual(result['draft']['provenance']['execution_mode'], 'fake_fixture')
        self.assertFalse(result['draft']['eligible_for_training'])
        self.assertNotIn(str(self.private[0]), repr(result['prompts']))

    def test_private_rejection_stops_without_repair_feedback(self):
        repeated = [[[7, 8]]] * 5
        result = run_fixture_slot(self.task, [FAKE_PROGRAMS[1], FAKE_PROGRAMS[1]],
                                  public_probes=self.public, private_probes=repeated)
        self.assertEqual(result['status'], 'private_acceptance_rejected')
        self.assertEqual(len(result['prompts']), 1)
        self.assertNotIn('draft', result)

    def test_repeated_failed_source_stops_without_third_request(self):
        result = run_fixture_slot(self.task, [FAKE_PROGRAMS[2]] * 3,
                                  public_probes=self.public, private_probes=self.private)
        self.assertEqual(result['status'], 'repeated_code')
        self.assertEqual(len(result['prompts']), 2)

    def test_rejects_more_than_three_or_nonfixture_sources(self):
        with self.assertRaises(ValueError):
            run_fixture_slot(self.task, [FAKE_PROGRAMS[1]] * 4,
                             public_probes=self.public, private_probes=self.private)
        with self.assertRaises(ValueError):
            run_fixture_slot(self.task, ['def transform(grid): return grid'],
                             public_probes=self.public, private_probes=self.private)


if __name__ == '__main__':
    unittest.main()

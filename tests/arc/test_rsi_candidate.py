"""Task-conditioned verification remains provisional until provenance audit."""
import unittest

from src.arc.data import Task
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_candidate import inspect_candidate


class RecordingSandbox(FakeE6Sandbox):
    def __init__(self):
        self.inputs = []

    def run(self, code, grid):
        self.inputs.append(grid)
        return super().run(code, grid)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.code = FAKE_PROGRAMS[1]
        self.task = Task('parent', (([[1, 2]], [[2, 1]]),), ([[3, 4]],))
        self.public = _quality_grids(42, 5)
        self.private = _quality_grids(43, 5)
        self.sandbox = RecordingSandbox()

    def test_generates_labels_only_through_sandbox_and_stays_provisional(self):
        result = inspect_candidate(self.task, self.code, public_probes=self.public,
                                   private_probes=self.private, sandbox=self.sandbox)
        self.assertEqual(result['status'], 'verified_candidate')
        self.assertFalse(result['eligible_for_training'])
        self.assertEqual(result['source_type'], 'generator_distillation')
        self.assertEqual(result['synthetic_task']['train'][0]['output'], [[2, 1]])
        self.assertEqual(result['synthetic_task']['query_outputs'], [[[4, 3]]])
        self.assertIn([[3, 4]], self.sandbox.inputs)
        self.assertNotIn('private_probes', repr(result))
        self.assertNotIn(str(self.private[0]), repr(result))

    def test_visible_mismatch_rejects_before_private_probes(self):
        task = Task('parent', (([[1, 2]], [[8, 8]]),), ([[3, 4]],))
        result = inspect_candidate(task, self.code, public_probes=self.public,
                                   private_probes=self.private, sandbox=self.sandbox)
        self.assertEqual(result['status'], 'visible_demonstration_mismatch')
        self.assertFalse(result['eligible_for_training'])
        self.assertFalse(any(grid in self.private for grid in self.sandbox.inputs))

    def test_overlapping_private_inputs_rejected_before_sandbox_execution(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            inspect_candidate(self.task, self.code, public_probes=self.public,
                              private_probes=self.public, sandbox=self.sandbox)
        self.assertEqual(self.sandbox.inputs, [])

    def test_private_failure_exposes_no_probe_details(self):
        class HiddenFailure(RecordingSandbox):
            def run(inner, code, grid):
                if grid in self.private:
                    raise ValueError('private secret output')
                return super().run(code, grid)
        result = inspect_candidate(self.task, self.code, public_probes=self.public,
                                   private_probes=self.private, sandbox=HiddenFailure())
        self.assertEqual(result['status'], 'private_acceptance_rejected')
        self.assertNotIn('secret', repr(result))
        self.assertFalse(result['eligible_for_training'])


if __name__ == '__main__':
    unittest.main()

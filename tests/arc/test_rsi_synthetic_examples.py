import copy
import unittest

from src.arc.data import Task
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_candidate import inspect_candidate
from src.arc.rsi_synthetic_examples import retain_synthetic_examples


class SyntheticExampleTests(unittest.TestCase):
    def setUp(self):
        self.task = Task('parent', (([[1, 2]], [[2, 1]]),), ([[3, 4]],))
        self.code = FAKE_PROGRAMS[1]
        self.candidate = inspect_candidate(self.task, self.code, public_probes=_quality_grids(42, 5),
                                           private_probes=_quality_grids(43, 5), sandbox=FakeE6Sandbox())

    def test_generated_query_examples_are_retained_with_source_identity(self):
        record = retain_synthetic_examples(self.task, self.code, self.candidate)
        self.assertEqual(record['examples'], [{'input': [[3, 4]], 'output': [[4, 3]]}])
        self.assertEqual(record['source_sha256'], self.candidate['source_sha256'])
        self.assertEqual(record['label_origin'], 'model_generated_program')
        self.assertFalse(record['training_allowed'])
        self.assertNotIn('private_behavior_sha256', record)
        self.assertNotIn('private_probe_set_sha256', record)

    def test_changed_source_parent_or_outputs_are_rejected(self):
        cases = []
        for key, value in [('query_inputs', [[[9]]]), ('query_outputs', []),
                           ('query_outputs', [[[99]]]), ('train', [])]:
            candidate = copy.deepcopy(self.candidate)
            candidate['synthetic_task'][key] = value
            cases.append(candidate)
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                retain_synthetic_examples(self.task, self.code, candidate)
        with self.assertRaises(ValueError):
            retain_synthetic_examples(self.task, FAKE_PROGRAMS[2], self.candidate)


if __name__ == '__main__':
    unittest.main()

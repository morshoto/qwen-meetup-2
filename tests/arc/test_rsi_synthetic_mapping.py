import copy
import unittest

from src.arc.data import Task
from src.arc.e5_selfplay import solve_prompt
from src.arc.rsi_synthetic_examples import retain_synthetic_examples
from src.arc.rsi_synthetic_mapping import map_synthetic_row
from src.arc.rsi_slot_audit import task_sha256
from tests.arc.test_rsi_sft_draft import SftDraftTests
from src.arc.rsi_sft_draft import make_sft_draft


class SyntheticMappingTests(unittest.TestCase):
    def setUp(self):
        fixture = SftDraftTests('test_draft_contains_no_query_outputs_or_private_probe_payload')
        fixture.setUp()
        self.task, self.code = fixture.task, fixture.code
        provenance = dict(fixture.provenance, execution_mode='unverified_capture',
                          slot=0, slot_evidence_sha256='b' * 64)
        self.draft = make_sft_draft(self.task, self.code, fixture.candidate, fixture.novelty,
                                    provenance, generation=0, parent_adapter='none')
        self.examples = retain_synthetic_examples(self.task, self.code, fixture.candidate)
        self.examples.update(request_id=provenance['request_id'], slot_evidence_sha256='b' * 64)

    def test_query_pair_becomes_demo_and_original_demo_is_held_out(self):
        result = map_synthetic_row(task=self.task, draft=self.draft, generated=self.examples)
        expected = Task(self.task.id, (([[3, 4]], [[4, 3]]),), ([[1, 2]],))
        self.assertEqual(result['messages'][1]['content'], solve_prompt(expected))
        self.assertEqual(result['messages'][2]['content'], self.code)
        self.assertEqual(result['mapping'], 'generated_demos_holdout_first_visible_v1')
        self.assertFalse(result['training_allowed'])

    def test_changed_binding_or_invalid_examples_are_rejected(self):
        for key, value in [('parent_task_sha256', 'a' * 64), ('source_sha256', 'a' * 64),
                           ('request_id', 'other'), ('slot_evidence_sha256', 'c' * 64),
                           ('label_origin', 'ground_truth')]:
            changed = copy.deepcopy(self.examples)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                map_synthetic_row(task=self.task, draft=self.draft, generated=changed)
        changed = copy.deepcopy(self.examples)
        changed['examples'][0]['input'] = [[9]]
        with self.assertRaises(ValueError):
            map_synthetic_row(task=self.task, draft=self.draft, generated=changed)
        changed = copy.deepcopy(self.examples)
        changed['examples'][0]['output'] = [[99]]
        with self.assertRaises(ValueError):
            map_synthetic_row(task=self.task, draft=self.draft, generated=changed)

    def test_repeated_inputs_cannot_leak_heldout_query(self):
        task = Task('parent', (([[1, 2]], [[2, 1]]),), ([[1, 2]],))
        draft = copy.deepcopy(self.draft)
        draft['messages'][1]['content'] = solve_prompt(task)
        examples = copy.deepcopy(self.examples)
        examples['parent_task_sha256'] = task_sha256(task)
        examples['examples'][0]['input'] = [[1, 2]]
        with self.assertRaises(ValueError):
            map_synthetic_row(task=task, draft=draft, generated=examples)


if __name__ == '__main__':
    unittest.main()

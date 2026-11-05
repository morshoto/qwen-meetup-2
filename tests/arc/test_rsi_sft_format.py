import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.arc.data import Task
from src.arc.e5_selfplay import solve_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.rsi_sft_format import audit_sft_format
from src.arc.rsi_training_preparation import prepare_training_inputs
from tests.arc.test_rsi_batch_precheck import row


class SFTFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tasks = {f't{i}': Task(f't{i}', (([[1, 2]], [[2, 1]]),), ([[3, 4]],)) for i in range(5)}
        self.drafts = [row(i) for i in range(5)]
        for draft in self.drafts:
            draft['messages'][0]['content'] = FINAL_ANSWER_SYSTEM_PROMPT
            draft['messages'][1]['content'] = solve_prompt(self.tasks[draft['parent_task_id']])
        prepare_training_inputs(root=self.root, drafts=self.drafts, tasks=self.tasks, generation=0,
                                parent_adapter='none', registration_sha256='a' * 64, seed=42)
        self.plan = json.loads((self.root / 'training.preparation.json').read_text())
        self.tokenizer = Mock()
        self.tokenizer.apply_chat_template.return_value = [10, 11, 12]
        self.render = Mock(return_value=([10, 11, 12, 13, 14], 3))

    def audit(self):
        return audit_sft_format(self.plan, tasks=self.tasks, tokenizer=self.tokenizer, render=self.render)

    def test_rows_match_inference_prefix_and_mask(self):
        result = self.audit()
        self.assertEqual(result['status'], 'format_consistent_unverified')
        self.assertEqual(result['row_count'], 5)
        self.assertFalse(result['training_allowed'])
        self.assertEqual([json.loads(line).keys() for line in result['dataset_jsonl'].splitlines()],
                         [{'messages'}] * 5)
        self.assertTrue(all(c.kwargs['enable_thinking'] is False for c in self.tokenizer.apply_chat_template.call_args_list))

    def test_mask_mismatch_empty_target_and_truncation_fail(self):
        for result in (([10, 11, 12, 13], 2), ([10, 11, 12], 3), ([10, 11, 12] + [13] * 2046, 3)):
            self.render.return_value = result
            with self.subTest(result=result[:1]), self.assertRaises(ValueError):
                self.audit()

    def test_mutated_rows_or_task_fail_before_render(self):
        self.plan['generation_training']['records'][0]['messages'][1]['content'] += ' hidden'
        with self.assertRaises(ValueError):
            self.audit()
        self.render.assert_not_called()

    def test_preparation_hash_and_mask_must_match(self):
        self.plan['dataset_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            self.audit()
        self.plan['generation_training']['mask_prompt'] = False
        with self.assertRaises(ValueError):
            self.audit()

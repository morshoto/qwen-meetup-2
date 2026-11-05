import json
import tempfile
import unittest
from pathlib import Path

from src.arc.data import Task
from src.arc.e5_selfplay import solve_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.rsi_training_preparation import prepare_training_inputs
from tests.arc.test_rsi_batch_precheck import row


class TrainingPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tasks = {f't{i}': Task(f't{i}', (([[1, 2]], [[2, 1]]),), ([[3, 4]],)) for i in range(5)}
        self.drafts = [row(i) for i in range(5)]
        for draft in self.drafts:
            draft['messages'][0]['content'] = FINAL_ANSWER_SYSTEM_PROMPT
            draft['messages'][1]['content'] = solve_prompt(self.tasks[draft['parent_task_id']])

    def prepare(self, **changes):
        args = dict(root=self.root, drafts=self.drafts, tasks=self.tasks, generation=0,
                    parent_adapter='none', registration_sha256='a' * 64, seed=42)
        args.update(changes)
        return prepare_training_inputs(**args)

    def test_five_drafts_create_quarantined_separate_plans(self):
        result = self.prepare()
        self.assertEqual(result['status'], 'prepared_quarantined')
        self.assertFalse(result['training_allowed'])
        plan = json.loads((self.root / 'training.preparation.json').read_text())
        self.assertEqual(plan['preflight']['max_steps'], 1)
        self.assertEqual(plan['preflight']['timeout_seconds'], 300)
        self.assertEqual(plan['generation_training']['max_steps'], 20)
        self.assertEqual(plan['generation_training']['timeout_seconds'], 1800)
        self.assertEqual(plan['generation_training']['lora_rank'], 8)
        self.assertTrue(plan['generation_training']['mask_prompt'])
        self.assertNotIn('records', plan['preflight'])
        self.assertIn('independent_batch_audit', result['blockers'])
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_too_few_drafts_cannot_prepare_batch(self):
        result = self.prepare(drafts=self.drafts[:4])
        self.assertEqual(result['status'], 'insufficient_records')
        self.assertFalse(result['training_allowed'])
        self.assertFalse((self.root / 'training.preparation.json').exists())

    def test_mismatched_or_leaking_prompt_is_rejected(self):
        self.drafts[0]['messages'][1]['content'] += '\n[[9]]'
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_grid_is_not_frozen_as_training_data(self):
        invalid = Task('t0', (([[99]], [[99]]),), ([[3, 4]],))
        self.tasks['t0'] = invalid
        self.drafts[0]['messages'][1]['content'] = solve_prompt(invalid)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_seed_or_registration_is_rejected(self):
        for changes in ({'seed': True}, {'registration_sha256': 'bad'}, {'seed': -1}):
            with self.assertRaises(ValueError):
                self.prepare(**changes)

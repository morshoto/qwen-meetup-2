import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.arc.data import Task
from src.arc.e5_selfplay import solve_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.rsi_local_tokenizer import audit_local_tokenizer
from src.arc.rsi_training_preparation import prepare_training_inputs
from tests.arc.test_rsi_batch_precheck import row

REV = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


class LocalTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / REV
        self.model.mkdir()
        (self.model / 'tokenizer.json').write_text('tokenizer')
        (self.model / 'tokenizer_config.json').write_text('config')
        self.token_hash = hashlib.sha256(b'tokenizer').hexdigest()
        self.config_hash = hashlib.sha256(b'config').hexdigest()
        self.tasks = {f't{i}': Task(f't{i}', (([[1, 2]], [[2, 1]]),), ([[3, 4]],)) for i in range(5)}
        drafts = [row(i) for i in range(5)]
        for draft in drafts:
            draft['messages'][0]['content'] = FINAL_ANSWER_SYSTEM_PROMPT
            draft['messages'][1]['content'] = solve_prompt(self.tasks[draft['parent_task_id']])
        prepare_training_inputs(root=self.root, drafts=drafts, tasks=self.tasks, generation=0,
                                parent_adapter='none', registration_sha256='a' * 64, seed=42)
        self.plan = json.loads((self.root / 'training.preparation.json').read_text())
        self.tokenizer = Mock()
        self.tokenizer.apply_chat_template.return_value = [10, 11]
        self.render = Mock(return_value=([10, 11, 12], 2))

    def audit(self):
        return audit_local_tokenizer(self.plan, tasks=self.tasks, model_dir=self.model,
            expected_tokenizer_sha256=self.token_hash, expected_config_sha256=self.config_hash)

    @patch('src.arc.rsi_local_tokenizer._load_components')
    def test_valid_components_bind_hashes_without_training_authorization(self, load):
        load.return_value = (self.tokenizer, self.render, '0.30.7')
        report = self.audit()
        self.assertEqual(report['status'], 'local_tokenizer_format_consistent')
        self.assertEqual(report['row_count'], 5)
        self.assertFalse(report['training_allowed'])
        self.assertNotIn('dataset_jsonl', report)

    @patch('src.arc.rsi_local_tokenizer._load_components')
    def test_changed_tokenizer_or_wrong_revision_fails_before_loading(self, load):
        (self.model / 'tokenizer.json').write_text('changed')
        with self.assertRaises(ValueError):
            self.audit()
        load.assert_not_called()

    @patch('src.arc.rsi_local_tokenizer._load_components')
    def test_wrong_runtime_and_mid_audit_drift_are_rejected(self, load):
        load.return_value = (self.tokenizer, self.render, 'wrong')
        with self.assertRaises(ValueError):
            self.audit()
        def mutate(_row):
            (self.model / 'tokenizer_config.json').write_text('changed')
            return ([10, 11, 12], 2)
        self.render.side_effect = mutate
        load.return_value = (self.tokenizer, self.render, '0.30.7')
        with self.assertRaises(ValueError):
            self.audit()

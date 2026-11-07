"""Synthetic offline integration; no real model or ARC protected data."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.arc.data import Task, Verifier
from src.arc.lt_batch_journal import BatchJournal
from src.arc.lt_live_backend import PROMPT
from src.arc.lt_phase_b_dataset import build_dataset
from src.arc.lt_phase_b_paired import score_paired
from src.arc.lt_phase_b_public_worker import execute_public_worker
from src.arc.lt_public import build_public_export


class FakeBackend:
    adapter_state, adapter_sha256 = 'off', None

    def generate(self, prompt, *, kind, candidate):
        payload = json.loads(prompt)
        assert kind == 'direct'
        assert all('output' not in entry for entry in payload['task']['test'])
        payload['instruction'] = PROMPT
        self.prompt = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        self.raw = '[[[1]]]'
        self.stats = {'prompt_tokens': 10, 'completion_tokens': 4, 'peak_memory_gb': 1, 'finish_reason': 'stop'}
        return self.raw


class FakeEndToEnd(unittest.TestCase):
    def test_journal_resume_verified_dataset_and_paired_score(self):
        ids = ['a', 'b', 'c']
        tasks = {x: Task(x, (([[ord(x) % 10]], [[1]]),), ([[ord(x) % 10]],)) for x in ids}
        verifier = Verifier({x: [[[1]]] for x in ids})
        backend = FakeBackend()
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        challenges = {task_id: task.prompt_data() for task_id, task in tasks.items()}
        receipt = {'comparison_sha256': 'c' * 64,
                   'selected_task_sha256': {task_id: digest(value) for task_id, value in challenges.items()}}
        export = build_public_export(challenges, receipt)
        with tempfile.TemporaryDirectory() as tmp:
            journal = BatchJournal(Path(tmp), ids, 'fixture-identity')
            for task_id in journal.pending(10):
                journal.start(task_id)
                result = execute_public_worker(export=export, receipt=receipt, task_id=task_id,
                    expected_run_sha256='c' * 64, source_code_sha256='a' * 64,
                    adapter_state='off', backend_factory=lambda: backend)
                result['cleanup'] = {'leader_reaped': True, 'group_absent': True}
                journal.finish(task_id, result)
            resumed = BatchJournal(Path(tmp), ids, 'fixture-identity')
            self.assertEqual(resumed.pending(10), [])
            rows = [json.loads((Path(tmp) / (x + '.result.json')).read_text()) for x in ids]
            output = build_dataset(tasks, rows, verifier, comparison_tasks={'x': Task('x', (([[0]], [[1]]),), ([[0]],))}, source={
                'model_id': 'mlx-community/Qwen3-14B-4bit',
                'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
                'source_code_sha256': 'a' * 64, 'run_sha256': 'c' * 64, 'run_id': 'fixture'})
            self.assertEqual(output['yield_status'], 'disjoint_split')
            self.assertEqual(output['eligible_task_count'], 3)
            self.assertEqual(output['train'][-1]['messages'][0]['content'], backend.prompt)
            paired = [dict(r, adapter_state=s) for r in rows for s in ('off', 'on')]
            score = score_paired(ids, paired, verifier)
            self.assertEqual(score['paired']['pass_at_1']['unchanged_right'], 3)
            self.assertEqual(score['identical_candidates'], {'off': 3, 'on': 3})


if __name__ == '__main__': unittest.main()

"""Synthetic offline integration; no real model or ARC protected data."""
import json
import tempfile
import unittest
from pathlib import Path

from src.arc.data import Task, Verifier
from src.arc.lt_batch_journal import BatchJournal
from src.arc.lt_live_backend import PROMPT
from src.arc.lt_phase_b_dataset import build_dataset
from src.arc.lt_phase_b_paired import score_paired
from src.arc.lt_worker import execute_task


class FakeBackend:
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
        with tempfile.TemporaryDirectory() as tmp:
            journal = BatchJournal(Path(tmp), ids, 'fixture-identity')
            for task_id in journal.pending(10):
                journal.start(task_id)
                result = execute_task(tasks[task_id], backend, '14b')
                result['cleanup'] = {'leader_reaped': True, 'group_absent': True}
                journal.finish(task_id, result)
            resumed = BatchJournal(Path(tmp), ids, 'fixture-identity')
            self.assertEqual(resumed.pending(10), [])
            rows = [json.loads((Path(tmp) / (x + '.result.json')).read_text()) for x in ids]
            output = build_dataset(tasks, rows, verifier, comparison_tasks={'x': Task('x', (([[0]], [[1]]),), ([[0]],))}, source={
                'model_id': 'mlx-community/Qwen3-14B-4bit',
                'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
                'source_code_sha256': 'a' * 64, 'run_id': 'fixture'})
            self.assertEqual(output['yield_status'], 'disjoint_split')
            self.assertEqual(output['eligible_task_count'], 3)
            self.assertEqual(output['train'][0]['messages'][0]['content'], backend.prompt)
            paired = [dict(r, adapter_state=s) for r in rows for s in ('off', 'on')]
            score = score_paired(ids, paired, verifier)
            self.assertEqual(score['paired']['pass_at_1']['unchanged_right'], 3)
            self.assertEqual(score['identical_candidates'], {'off': 3, 'on': 3})


if __name__ == '__main__': unittest.main()

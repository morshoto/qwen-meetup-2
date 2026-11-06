import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.arc.http_backend import BackendError
from src.arc.lt_comparison import freeze_manifest, make_manifest
from src.arc.lt_journal import run_fixture_model, audit_fixture_run
from src.arc.lt_public import build_public_export
from src.arc.sandbox import FakeSandbox, IDENTITY_CODE


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = [f't{i:03}' for i in range(200)]
        self.tasks = {x: {'train': [{'input': [[1]], 'output': [[1]]}],
                          'test': [{'input': [[2]]}]} for x in self.ids}
        proposal = make_manifest(self.ids, seed=1, development=set(), training=set(), exposures=set(), dataset_sha256='a' * 64)
        config = {'prompt_sha256': 'b' * 64, 'code_sha256': 'c' * 64,
                  'models': {'14b': 'rev14', '32b': 'rev32'}, 'thinking': 'off',
                  'temperature': 0.7, 'candidates': 2, 'max_tokens': 1000, 'timeout_seconds': 60,
                  'seed': 42, 'max_repairs': 0, 'prompt_variant': 'legacy',
                  'model_files_sha256': {'14b': 'd' * 64, '32b': 'e' * 64}, 'runtime_sha256': 'f' * 64,
                  'budget': {'max_calls': 2000, 'max_reserved_tokens': 20000000, 'max_sandbox_starts': 8000,
                             'max_seconds': 36000, 'max_calls_per_task': 10, 'max_reserved_tokens_per_task': 100000,
                             'max_sandbox_per_task': 40, 'max_seconds_per_task': 120}}
        self.frozen = freeze_manifest(proposal, config)
        self.receipt = {'comparison_sha256': self.frozen['sha256'], 'selected_task_sha256': {
            x: hashlib.sha256(json.dumps(self.tasks[x], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            for x in self.ids}}
        self.export = build_public_export(self.tasks, self.receipt)

    def test_records_completed_and_aborted_tasks_without_retry(self):
        class Backend:
            def __init__(self): self.calls = 0
            def generate(self, prompt, *, kind, candidate):
                self.calls += 1
                if self.calls == 5: raise BackendError('unavailable', 'fixture')
                return 'copy input' if kind == 'analysis' else IDENTITY_CODE
        backend = Backend()
        result = run_fixture_model(self.root, self.frozen, self.receipt, self.export,
                                   '14b', backend, FakeSandbox())
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['attempted'], 2)
        rows = [json.loads(line) for line in (self.root / '14b.results.jsonl').read_text().splitlines()]
        self.assertEqual([row['status'] for row in rows], ['complete', 'error'])
        self.assertEqual(rows[0]['model'], '14b')
        self.assertEqual(rows[0]['comparison_sha256'], self.frozen['sha256'])
        self.assertIsNone(rows[0]['tokens'])
        self.assertFalse(result['run_identity_independently_verified'])
        with self.assertRaises(FileExistsError):
            run_fixture_model(self.root, self.frozen, self.receipt, self.export, '14b', backend, FakeSandbox())
        self.assertEqual(backend.calls, 5)

    def test_model_manifest_binds_journal_and_detects_tampering(self):
        class Backend:
            def generate(self, prompt, *, kind, candidate):
                raise BackendError('unavailable', 'fixture')
        terminal = run_fixture_model(self.root, self.frozen, self.receipt, self.export,
                                     '14b', Backend(), FakeSandbox())
        self.assertEqual(terminal['model_revision'], 'rev14')
        self.assertEqual(terminal['model_files_sha256'], 'd' * 64)
        self.assertEqual(terminal['runtime_sha256'], 'f' * 64)
        self.assertEqual(terminal['results_sha256'], hashlib.sha256((self.root / '14b.results.jsonl').read_bytes()).hexdigest())
        audit = audit_fixture_run(self.root, self.frozen, self.receipt, '14b')
        self.assertEqual(audit['status'], 'consistent_unverified')
        self.assertFalse(audit['run_identity_independently_verified'])
        marker = self.root / '14b.task-000.started.json'
        original_marker = marker.read_text()
        marker.write_text(original_marker.replace(self.ids[0], 'unexpected'))
        with self.assertRaises(ValueError): audit_fixture_run(self.root, self.frozen, self.receipt, '14b')
        marker.write_text(original_marker)
        with (self.root / '14b.results.jsonl').open('a') as stream: stream.write('changed\n')
        with self.assertRaises(ValueError): audit_fixture_run(self.root, self.frozen, self.receipt, '14b')

    def test_fixture_journal_feeds_two_model_offline_scorer(self):
        from src.arc.data import Verifier
        from src.arc.lt_scoring import score_comparison
        class Backend:
            def __init__(self): self.calls = 0
            def generate(self, prompt, *, kind, candidate):
                self.calls += 1
                if self.calls == 5: raise BackendError('unavailable', 'fixture')
                return 'copy input' if kind == 'analysis' else IDENTITY_CODE
        rows = []
        for model in ('14b', '32b'):
            run_fixture_model(self.root, self.frozen, self.receipt, self.export, model, Backend(), FakeSandbox())
            rows.extend(json.loads(line) for line in (self.root / f'{model}.results.jsonl').read_text().splitlines())
        verifier = Verifier({task_id: [[[2]]] for task_id in self.ids})
        report = score_comparison(self.ids, rows, verifier)
        for model in ('14b', '32b'):
            self.assertEqual(report['models'][model]['status'], 'incomplete')
            self.assertEqual(report['models'][model]['selected'], 200)
            self.assertEqual(report['models'][model]['attempted'], 2)
            self.assertEqual(report['models'][model]['pass_at_1'], 1)
            self.assertEqual(report['models'][model]['pass_at_2'], 1)

    def test_interrupt_leaves_started_and_incomplete_records(self):
        class Backend:
            def generate(self, prompt, *, kind, candidate):
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run_fixture_model(self.root, self.frozen, self.receipt, self.export,
                              '14b', Backend(), FakeSandbox())
        self.assertTrue((self.root / '14b.task-000.started.json').exists())
        failure = json.loads((self.root / '14b.incomplete.json').read_text())
        self.assertEqual(failure['failure_type'], 'KeyboardInterrupt')
        self.assertEqual(failure['attempted'], 0)
        self.assertFalse((self.root / '14b.terminal.json').exists())
        self.assertEqual((self.root / '14b.results.jsonl').read_text(), '')

    def test_changed_public_task_is_rejected_before_any_call(self):
        class Backend:
            def generate(self, *args, **kwargs): raise AssertionError('must not call')
        self.export['tasks'][self.ids[0]]['test'][0]['input'] = [[9]]
        with self.assertRaises(ValueError):
            run_fixture_model(self.root, self.frozen, self.receipt, self.export,
                              '14b', Backend(), FakeSandbox())
        self.assertFalse((self.root / '14b.started.json').exists())


if __name__ == '__main__': unittest.main()

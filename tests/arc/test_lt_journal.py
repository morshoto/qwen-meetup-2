import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_comparison import freeze_manifest, make_manifest
from src.arc.lt_journal import FixtureBackend, run_fixture_model, audit_fixture_run, _usage_total
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
        proposal = make_manifest(self.ids, seed=1, development=set(), training=set(), exposures=set(), protected=set(), dataset_sha256='a' * 64)
        config = {'prompt_sha256': 'b' * 64, 'code_sha256': 'c' * 64,
                  'models': {'14b': '1' * 40, '32b': '2' * 40},
                  'model_names': {'14b': 'model14', '32b': 'model32'}, 'backend_url': 'http://127.0.0.1:8080/v1',
                  'sandbox': {'image': 'sha256:' + '9' * 64, 'timeout': 3, 'memory_mb': 128, 'cpus': 1, 'pids': 32}, 'thinking': 'off',
                  'temperature': 0.7, 'candidates': 2, 'max_tokens': 1000, 'max_prompt_tokens': 5000, 'timeout_seconds': 60,
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
        backend = FixtureBackend(fail_at=5)
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

    def test_token_accounting_requires_usage_for_every_reserved_request(self):
        events = [{'usage': {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7}},
                  {'usage': {'total_tokens': 3}}]
        self.assertEqual(_usage_total(events, 2), 10)
        self.assertIsNone(_usage_total(events, 3))
        self.assertIsNone(_usage_total([{'usage': None}], 1))
        self.assertIsNone(_usage_total([{'usage': {'prompt_tokens': 2, 'completion_tokens': 2, 'total_tokens': 3}}], 1))
        self.assertIsNone(_usage_total([{'usage': {'total_tokens': True}}], 1))
        self.assertIsNone(_usage_total([{'usage': {'prompt_tokens': 8, 'total_tokens': 3}}], 1))

    def test_journal_records_observed_per_task_usage(self):
        result = run_fixture_model(self.root, self.frozen, self.receipt, self.export,
                                   '14b', FixtureBackend(fail_at=5, tokens_per_call=7), FakeSandbox())
        rows = [json.loads(line) for line in (self.root / '14b.results.jsonl').read_text().splitlines()]
        self.assertEqual([r['tokens'] for r in rows], [28, 7])
        self.assertEqual(result['status'], 'incomplete')

    def test_model_manifest_binds_journal_and_detects_tampering(self):
        terminal = run_fixture_model(self.root, self.frozen, self.receipt, self.export,
                                     '14b', FixtureBackend(fail_at=1), FakeSandbox())
        self.assertEqual(terminal['model_revision'], '1' * 40)
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
        rows = []
        for model in ('14b', '32b'):
            run_fixture_model(self.root, self.frozen, self.receipt, self.export, model, FixtureBackend(fail_at=5), FakeSandbox())
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
        with self.assertRaises(KeyboardInterrupt):
            run_fixture_model(self.root, self.frozen, self.receipt, self.export,
                              '14b', FixtureBackend(interrupt_at=1), FakeSandbox())
        self.assertTrue((self.root / '14b.task-000.started.json').exists())
        failure = json.loads((self.root / '14b.incomplete.json').read_text())
        self.assertEqual(failure['failure_type'], 'KeyboardInterrupt')
        self.assertEqual(failure['attempted'], 0)
        self.assertFalse((self.root / '14b.terminal.json').exists())
        self.assertEqual((self.root / '14b.results.jsonl').read_text(), '')

    def test_fixture_entrypoint_rejects_real_transport_before_dispatch(self):
        from unittest.mock import patch
        from src.arc.lt_components import prepare_components
        backend, sandbox = prepare_components(self.frozen, '14b')
        with patch.object(backend, 'generate') as generate, patch.object(sandbox, 'run') as run:
            with self.assertRaises(ValueError):
                run_fixture_model(self.root, self.frozen, self.receipt, self.export,
                                  '14b', backend, sandbox)
        generate.assert_not_called()
        run.assert_not_called()
        self.assertFalse((self.root / '14b.started.json').exists())
        class UnknownBackend:
            def generate(self, *args, **kwargs): raise AssertionError('must not run')
        with self.assertRaises(ValueError):
            run_fixture_model(self.root, self.frozen, self.receipt, self.export,
                              '14b', UnknownBackend(), FakeSandbox())
        with self.assertRaises(ValueError):
            run_fixture_model(self.root, self.frozen, self.receipt, self.export,
                              '14b', FixtureBackend(), sandbox)

    def test_changed_public_task_is_rejected_before_any_call(self):
        self.export['tasks'][self.ids[0]]['test'][0]['input'] = [[9]]
        with self.assertRaises(ValueError):
            run_fixture_model(self.root, self.frozen, self.receipt, self.export,
                              '14b', FixtureBackend(interrupt_at=1), FakeSandbox())
        self.assertFalse((self.root / '14b.started.json').exists())


if __name__ == '__main__': unittest.main()

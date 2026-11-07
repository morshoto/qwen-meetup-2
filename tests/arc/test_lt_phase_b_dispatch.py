import fcntl
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_dispatch import run_collection
from tests.arc import test_lt_phase_b_registered_worker as fixtures
from tests.arc.test_rsi_process_monitor import state


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'artifacts').mkdir()
        self.run = self.root / 'artifacts/lt-phase-b-fixture'; self.run.mkdir(mode=0o700)
        self.fixture = fixtures.RegisteredWorkerTests('test_frozen_public_worker_binds_identity_before_and_after')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.sha = fixtures.sha(self.fixture.spec)
        receipt = {'comparison_sha256': self.sha, 'selected_task_sha256': self.fixture.hashes}
        from src.arc.lt_public import build_public_export
        data = {'spec': self.fixture.spec, 'inventory': self.fixture.inventory,
                'allocation': self.fixture.allocation, 'receipt': receipt,
                'export': build_public_export(self.fixture.challenges, receipt)}
        for name, value in data.items():
            (self.run / (name + '.json')).write_text(json.dumps(value))
        self.challenge = self.root / 'challenges.json'
        self.challenge.write_text(json.dumps(self.fixture.challenges))
        self.archive = self.root / 'official.zip'
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('arc-agi_evaluation_challenges.json', '{}')
        self.preflight = patch('src.arc.lt_phase_b_dispatch.preflight_collection',
                               return_value={'status': 'preflight_consistent_unverified', 'execution_ready': False,
                                             'run_sha256': self.sha, 'selected_tasks': 100,
                                             'evidence_type': 'local_files_and_caller_supplied_registration'})
        self.check = self.preflight.start(); self.addCleanup(self.preflight.stop)
        self.calls = []
        def worker(argv, *, root, timeout, on_start, **kwargs):
            self.calls.append(argv)
            on_start(1234)
            output = Path(argv[argv.index('--output') + 1])
            task = argv[argv.index('--task') + 1]
            _create(output, {'task_id': task, 'model': '14b', 'status': 'complete',
                             'candidates': [[[[1]]], [[[1]]]], 'calls': []})
            return {'status': 'complete', 'leader_reaped': True, 'group_absent': True,
                    'returncode': 0, 'elapsed_seconds': 1., 'stdout': '', 'stderr': None,
                    'baseline': state(), 'observation': {'stop_reason': 'process_exited', 'exit_code': 0,
                    'stream_complete': True, 'cleanup_observation': {'group_cleanup_verified': False},
                    'post_reap_group_absent': True, 'samples': [state()], 'escaped_processes_excluded': False}}
        self.worker = patch('src.arc.lt_phase_b_dispatch.run_monitored_worker', side_effect=worker)
        self.worker.start(); self.addCleanup(self.worker.stop)

    def invoke(self):
        return run_collection(source_root=self.root, run_dir=self.run, expected_run_sha256=self.sha,
            challenge_path=self.challenge, official_archive=self.archive, python_executable='/fixture/python',
            model_path='/fixture/model', limit=1)

    def test_preflight_then_dispatch_public_only_worker(self):
        result = self.invoke()
        self.assertEqual(result, {'completed': 1, 'stop_reason': 'batch_limit'})
        self.check.assert_called_once()
        self.assertEqual(len(self.calls), 1)
        argv = self.calls[0]
        self.assertEqual(argv[:3], ['/fixture/python', '-I', '-B'])
        self.assertNotIn(str(self.challenge), argv)
        self.assertNotIn(str(self.archive), argv)
        self.assertTrue((self.run / 'collection/00000000.result.json').exists())

    def test_failed_preflight_never_dispatches(self):
        self.check.side_effect = ValueError('fixture mismatch')
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.run / 'collection').exists())

    def test_unexpected_preflight_result_never_dispatches(self):
        self.check.return_value = {'status': 'other', 'execution_ready': True}
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.run / 'collection').exists())

    def test_concurrent_lock_never_dispatches(self):
        fd = os.open(self.run / 'run.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): self.invoke()
            self.assertEqual(self.calls, [])
        finally:
            os.close(fd)

    def test_nonprivate_directory_is_rejected(self):
        self.run.chmod(0o755)
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.calls, [])


if __name__ == '__main__': unittest.main()

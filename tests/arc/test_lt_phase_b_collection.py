import json
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_batch_journal import BatchJournal, _create
from src.arc.lt_phase_b_collection import collect_batch
from tests.arc.test_rsi_process_monitor import state


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.journal = BatchJournal(self.root, ['a', 'b', 'c'], 'fixture')
        self.calls = []

    def worker(self, task, output, on_start):
        self.calls.append(task)
        on_start(1234)
        _create(output, {'task_id': task, 'model': '14b', 'status': 'complete',
                         'candidates': [[[[1]]], [[[1]]]], 'calls': []})
        return {'status': 'complete', 'leader_reaped': True, 'group_absent': True,
                'returncode': 0, 'elapsed_seconds': 1., 'stdout': '', 'stderr': None,
                'baseline': state(), 'observation': {'stop_reason': 'process_exited', 'exit_code': 0,
                'group_signals_confirmed': False, 'stream_complete': True,
                'cleanup_observation': {'group_cleanup_verified': False}, 'post_reap_group_absent': True,
                'samples': [state()], 'escaped_processes_excluded': False}}

    def test_success_persists_observation_and_resumes(self):
        result = collect_batch(self.journal, limit=2, worker=self.worker)
        self.assertEqual(result, {'completed': 2, 'stop_reason': 'batch_limit'})
        self.assertEqual(self.calls, ['a', 'b'])
        recorded = json.loads((self.root / 'a.resources.json').read_text())
        self.assertEqual(recorded['observation']['stop_reason'], 'process_exited')
        self.assertFalse(recorded['observation']['escaped_processes_excluded'])
        self.assertEqual(json.loads((self.root / 'a.process.json').read_text())['pid'], 1234)
        self.assertEqual(self.journal.pending(2), ['c'])
        self.assertEqual(collect_batch(self.journal, limit=2, worker=self.worker)['stop_reason'], 'complete')

    def test_failure_is_journaled_and_stops_subsequent_dispatch(self):
        def worker(task, output, on_start):
            on_start(1234)
            return {'status': 'timeout', 'leader_reaped': True, 'group_absent': True,
                    'returncode': -9, 'elapsed_seconds': 2., 'stdout': '', 'stderr': None,
                    'baseline': {'fixture': True}, 'observation': {'stop_reason': 'wall_clock_limit'}}
        result = collect_batch(self.journal, limit=3, worker=worker)
        self.assertEqual(result, {'completed': 1, 'stop_reason': 'worker_timeout'})
        self.assertEqual(json.loads((self.root / 'a.result.json').read_text())['status'], 'timeout')
        self.assertEqual(self.journal.pending(3), ['b', 'c'])
        self.assertFalse((self.root / 'b.started.json').exists())
        self.assertEqual(collect_batch(self.journal, limit=3, worker=self.worker),
                         {'completed': 0, 'stop_reason': 'worker_timeout'})
        self.assertEqual(self.calls, [])

    def test_unverified_cleanup_never_commits_and_requires_reconciliation(self):
        def worker(task, output, on_start):
            result = self.worker(task, output, on_start)
            result['group_absent'] = False
            return result
        with self.assertRaises(ValueError): collect_batch(self.journal, limit=2, worker=worker)
        self.assertTrue((self.root / 'a.resources.json').exists())
        self.assertFalse((self.root / 'a.result.json').exists())
        with self.assertRaises(ValueError): self.journal.pending(1)

    def test_success_claim_with_unsafe_samples_is_not_committed(self):
        def worker(task, output, on_start):
            value = self.worker(task, output, on_start)
            value['observation']['samples'] = [state(free=10)]
            return value
        with self.assertRaises(ValueError): collect_batch(self.journal, limit=2, worker=worker)
        self.assertFalse((self.root / 'a.result.json').exists())
        self.assertTrue((self.root / 'a.resources.json').exists())
        with self.assertRaises(ValueError): self.journal.pending(1)

    def test_preflight_refusal_is_not_a_started_attempt_result(self):
        def worker(task, output, on_start):
            return {'status': 'preflight_refused', 'reason': 'insufficient_disk', 'baseline': {}}
        with self.assertRaises(ValueError): collect_batch(self.journal, limit=2, worker=worker)
        self.assertFalse((self.root / 'a.result.json').exists())
        self.assertEqual(json.loads((self.root / 'a.resources.json').read_text())['reason'], 'insufficient_disk')
        with self.assertRaises(ValueError): self.journal.pending(1)


if __name__ == '__main__': unittest.main()

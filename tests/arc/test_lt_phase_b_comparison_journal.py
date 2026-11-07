import json
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_batch_journal import BatchJournal, _create
from src.arc.lt_phase_b_comparison_journal import compare_batch
from tests.arc.test_rsi_process_monitor import state


class ComparisonJournalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.ids = ['00000001', '00000002']
        self.run = 'a' * 64; self.source = 'b' * 64; self.adapter = 'c' * 64
        self.registration = {'run_sha256': self.run, 'source_code_sha256': self.source,
            'adapter_sha256': self.adapter, 'selected_task_sha256': {x: 'd'*64 for x in self.ids},
            'selected_prompt_sha256': {x: 'e'*64 for x in self.ids}}
        self.off = BatchJournal(self.root / 'off', self.ids, self.run + ':14b:off')
        self.on = BatchJournal(self.root / 'on', self.ids, self.run + ':14b:on')
        self.limits = {'timeout_seconds': 300, 'minimum_disk_gib': 20, 'minimum_free_memory_percent': 20,
                       'maximum_swap_growth_mib': 512, 'maximum_process_rss_gib': 32, 'sample_interval_seconds': 5}
        self.calls = []
    def worker(self, task, adapter_state, output, on_start):
        self.calls.append((task, adapter_state)); on_start(1234)
        _create(output, {'task_id': task, 'model': '14b', 'status': 'complete', 'candidates': [[[[1]]], [[[1]]]],
            'adapter_state': adapter_state, 'adapter_sha256': self.adapter if adapter_state == 'on' else None,
            'run_sha256': self.run, 'source_code_sha256': self.source, 'public_task_sha256': 'd'*64,
            'prompt_sha256': 'e'*64})
        return {'status': 'complete', 'leader_reaped': True, 'group_absent': True, 'returncode': 0,
            'elapsed_seconds': 1., 'stdout': '', 'stderr': None, 'baseline': state(),
            'observation': {'stop_reason': 'process_exited', 'exit_code': 0, 'elapsed_seconds': 1.,
                'stream_complete': True, 'post_reap_group_absent': True,
                'cleanup_observation': {'group_cleanup_verified': False}, 'samples': [state()]}}
    def run_batch(self, limit=2, worker=None):
        return compare_batch(off_journal=self.off, on_journal=self.on, registration=self.registration,
                             limit=limit, worker=worker or self.worker, resource_limits=self.limits)

    def test_off_then_on_sequential_resume(self):
        self.assertEqual(self.run_batch(1), {'completed': 1, 'stop_reason': 'batch_limit', 'state': 'off'})
        self.assertEqual(self.run_batch(2), {'completed': 1, 'stop_reason': 'state_complete', 'state': 'off'})
        self.assertEqual(self.run_batch(2), {'completed': 2, 'stop_reason': 'complete', 'state': 'on'})
        self.assertEqual(self.calls, [(self.ids[0], 'off'), (self.ids[1], 'off'),
                                      (self.ids[0], 'on'), (self.ids[1], 'on')])
        self.assertEqual(self.run_batch(), {'completed': 0, 'stop_reason': 'complete', 'state': 'on'})

    def test_wrong_worker_identity_never_commits(self):
        def bad(task, adapter_state, output, on_start):
            result = self.worker(task, adapter_state, output, on_start)
            row = json.loads(output.read_text()); row['adapter_state'] = 'on'; output.write_text(json.dumps(row))
            return result
        with self.assertRaises(ValueError): self.run_batch(worker=bad)
        self.assertFalse((self.off.root / (self.ids[0] + '.result.json')).exists())
        with self.assertRaises(ValueError): self.run_batch()

    def test_tampered_prior_resources_fail_before_resume(self):
        self.run_batch(1)
        path = self.off.root / (self.ids[0] + '.resources.json')
        value = json.loads(path.read_text()); value['observation']['samples'][0]['memory_free_percent'] = 1
        path.write_text(json.dumps(value)); path.chmod(0o600)
        with self.assertRaises(ValueError): self.run_batch()
        self.assertEqual(len(self.calls), 1)

    def test_failure_stops_and_does_not_start_next_state(self):
        def fail(task, adapter_state, output, on_start):
            on_start(1234)
            return {'status': 'timeout', 'leader_reaped': True, 'group_absent': True, 'returncode': -9,
                    'elapsed_seconds': 2., 'stdout': '', 'stderr': None,
                    'baseline': state(), 'observation': {'stop_reason': 'wall_clock_limit'}}
        result = self.run_batch(worker=fail)
        self.assertEqual(result['stop_reason'], 'worker_timeout')
        self.assertEqual(self.run_batch()['stop_reason'], 'worker_timeout')
        self.assertFalse((self.on.root / (self.ids[0] + '.started.json')).exists())


if __name__ == '__main__': unittest.main()

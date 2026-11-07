import os
import sys
import unittest
from unittest.mock import patch

from src.arc.lt_phase_b_worker_supervisor import run_monitored_worker
from tests.arc.test_rsi_process_monitor import state


class WorkerSupervisorTests(unittest.TestCase):
    def run_worker(self, code, *, sampler=None, on_start=None):
        return run_monitored_worker([sys.executable, '-u', '-c', code], root='.', timeout=3,
            on_start=on_start, baseline_sampler=lambda _: state(),
            process_sampler=sampler or (lambda _root, _pid: state()),
            interval_seconds=0.05, sample_timeout_seconds=0.2, terminate_grace_seconds=0.1)

    def test_finite_worker_keeps_process_and_resource_evidence(self):
        pids = []
        result = self.run_worker('print("fixture", flush=True)', on_start=pids.append)
        self.assertEqual(result['observation']['stop_reason'], 'process_exited')
        self.assertEqual(result['observation']['exit_code'], 0)
        self.assertEqual(result['observation']['output'], 'fixture\n')
        self.assertTrue(result['leader_reaped'])
        self.assertEqual(len(pids), 1)
        self.assertFalse(result['observation']['escaped_processes_excluded'])
        self.assertEqual(result['status'], 'complete' if result['group_absent'] else 'error')

    def test_live_resource_failure_terminates_child(self):
        result = self.run_worker('import time; time.sleep(5)', sampler=lambda _root, _pid: state(free=10))
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['observation']['stop_reason'], 'memory_pressure_limit')
        self.assertTrue(result['leader_reaped'])
        self.assertTrue(result['observation']['samples'])

    def test_unsafe_baseline_refuses_to_spawn(self):
        with patch('src.arc.lt_phase_b_worker_supervisor.subprocess.Popen') as popen:
            result = run_monitored_worker([sys.executable, '-c', 'pass'], root='.', timeout=3,
                baseline_sampler=lambda _: state(free=10))
        popen.assert_not_called()
        self.assertEqual(result['status'], 'preflight_refused')
        self.assertEqual(result['reason'], 'memory_pressure_limit')

    def test_failed_start_callback_terminates_owned_child(self):
        pids = []
        def fail(pid):
            pids.append(pid)
            raise RuntimeError('fixture failure')
        with self.assertRaises(RuntimeError):
            self.run_worker('import time; time.sleep(5)', on_start=fail)
        self.assertEqual(len(pids), 1)
        with self.assertRaises(ProcessLookupError): os.kill(pids[0], 0)


if __name__ == '__main__': unittest.main()

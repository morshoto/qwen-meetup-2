import os
import sys
import tempfile
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
        self.assertTrue(result['observation']['post_reap_group_absent'])
        self.assertTrue(result['group_absent'])
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['observation']['cleanup_observation']['group_cleanup_verified'])

    def test_post_reap_group_probe_fails_closed_if_inaccessible(self):
        from src.arc.lt_phase_b_worker_supervisor import _post_reap_group_absent
        with patch('src.arc.lt_phase_b_worker_supervisor.os.killpg', side_effect=PermissionError):
            self.assertIsNone(_post_reap_group_absent(1234))
        with patch('src.arc.lt_phase_b_worker_supervisor.os.killpg', return_value=None):
            self.assertFalse(_post_reap_group_absent(1234))
        with patch('src.arc.lt_phase_b_worker_supervisor.os.killpg', side_effect=ProcessLookupError):
            self.assertTrue(_post_reap_group_absent(1234))

    def test_worker_uses_requested_source_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_monitored_worker([sys.executable, '-u', '-c', 'import os; print(os.getcwd())'],
                root=tmp, cwd=tmp, timeout=3, baseline_sampler=lambda _: state(),
                process_sampler=lambda _root, _pid: state(), interval_seconds=.05,
                sample_timeout_seconds=.2, terminate_grace_seconds=.1)
            self.assertEqual(os.path.realpath(result['observation']['output'].strip()), os.path.realpath(tmp))

    def test_live_resource_failure_terminates_child(self):
        result = self.run_worker('import time; time.sleep(5)', sampler=lambda _root, _pid: state(free=10))
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['observation']['stop_reason'], 'memory_pressure_limit')
        self.assertTrue(result['leader_reaped'])
        self.assertTrue(result['observation']['samples'])

    def test_training_mode_requires_one_finite_loss(self):
        def run(code):
            return run_monitored_worker([sys.executable, '-u', '-c', code], root='.', timeout=3,
                baseline_sampler=lambda _: state(), process_sampler=lambda _root, _pid: state(),
                interval_seconds=.05, sample_timeout_seconds=.2, terminate_grace_seconds=.1, training=True)
        good = run('print("Iter 1: Train loss 0.125, x", flush=True)')
        self.assertEqual(good['status'], 'complete')
        self.assertEqual(good['observation']['loss']['last_step'], 1)
        self.assertIn('Train loss 0.125', good['stdout'])
        empty = run('print("finished", flush=True)')
        self.assertEqual(empty['status'], 'error')
        nonfinite = run('import time; print("Iter 1: Train loss nan, x", flush=True); time.sleep(5)')
        self.assertEqual(nonfinite['status'], 'error')
        self.assertEqual(nonfinite['observation']['stop_reason'], 'nonfinite_loss')

    def test_real_training_requires_registered_final_step(self):
        def run(last, **kwargs):
            code = f'print("Iter {last}: Train loss 0.125, x", flush=True)'
            return run_monitored_worker([sys.executable, '-u', '-c', code], root='.', timeout=3,
                baseline_sampler=lambda _: state(), process_sampler=lambda _root, _pid: state(),
                interval_seconds=.05, sample_timeout_seconds=.2, terminate_grace_seconds=.1,
                training=True, **kwargs)
        self.assertEqual(run(20, expected_training_steps=20)['status'], 'complete')
        self.assertEqual(run(19, expected_training_steps=20)['status'], 'error')
        self.assertEqual(run(21, expected_training_steps=20)['status'], 'error')
        self.assertEqual(run(20)['status'], 'error')

    def test_invalid_expected_training_steps_fail_before_spawn(self):
        with patch('src.arc.lt_phase_b_worker_supervisor.subprocess.Popen') as popen:
            for bad in (True, 0, 2, 19, 21, 20.0):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    run_monitored_worker([sys.executable, '-c', 'pass'], root='.', timeout=3,
                        training=True, expected_training_steps=bad)
            with self.assertRaises(ValueError):
                run_monitored_worker([sys.executable, '-c', 'pass'], root='.', timeout=3,
                    training=False, expected_training_steps=20)
            popen.assert_not_called()

    def test_invalid_monitor_settings_are_rejected_before_spawning(self):
        with patch('src.arc.lt_phase_b_worker_supervisor.subprocess.Popen') as popen:
            for settings in ({'interval_seconds': 0}, {'sample_timeout_seconds': float('nan')},
                             {'terminate_grace_seconds': -1}, {'interval_seconds': 6}):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    run_monitored_worker([sys.executable, '-c', 'pass'], root='.', timeout=3,
                                         baseline_sampler=lambda _: state(), **settings)
        popen.assert_not_called()

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

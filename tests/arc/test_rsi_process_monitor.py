"""A watchdog may signal only its explicitly owned fixture process group."""
import os
import subprocess
import sys
import time
import unittest

from src.arc.rsi_process_monitor import watch_owned_process


def state(*, free=70):
    return {'monotonic_seconds': time.monotonic(), 'disk_free_bytes': 30 * 1024**3,
            'memory_free_percent': free, 'swap_used_mib': 0, 'process_tree_rss_bytes': 0}


def child(seconds=30, *, session=True):
    return subprocess.Popen([sys.executable, '-c', f'import time; time.sleep({seconds})'],
                            start_new_session=session, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class OwnedMonitorTests(unittest.TestCase):
    def test_resource_violation_terminates_owned_fixture_process(self):
        process = child()
        try:
            result = watch_owned_process(process, baseline=state(), sampler=lambda _: state(free=19),
                                         timeout_seconds=3, interval_seconds=0.05,
                                         sample_timeout_seconds=0.2, terminate_grace_seconds=0.1)
            self.assertEqual(result['stop_reason'], 'memory_pressure_limit')
            self.assertIsNotNone(process.returncode)
            self.assertFalse(result['training_verified'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()

    def test_monitor_timeout_stops_owned_fixture(self):
        process = child()
        def slow(_):
            time.sleep(0.4)
            return state()
        try:
            result = watch_owned_process(process, baseline=state(), sampler=slow,
                                         timeout_seconds=3, interval_seconds=0.05,
                                         sample_timeout_seconds=0.05, terminate_grace_seconds=0.1)
            self.assertEqual(result['stop_reason'], 'monitor_unavailable')
            self.assertIsNotNone(process.returncode)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()

    def test_nonowned_process_group_is_rejected_without_signal(self):
        process = child(session=False)
        try:
            with self.assertRaisesRegex(ValueError, 'owned'):
                watch_owned_process(process, baseline=state(), sampler=lambda _: state(),
                                    timeout_seconds=3, interval_seconds=0.05,
                                    sample_timeout_seconds=0.2, terminate_grace_seconds=0.1)
            self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=3)

    def test_fixture_exit_is_not_reported_as_verified_training(self):
        process = child(0.2)
        try:
            result = watch_owned_process(process, baseline=state(), sampler=lambda _: state(),
                                         timeout_seconds=3, interval_seconds=0.05,
                                         sample_timeout_seconds=0.2, terminate_grace_seconds=0.1)
            self.assertEqual(result['stop_reason'], 'process_exited')
            self.assertEqual(result['exit_code'], 0)
            self.assertFalse(result['training_verified'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()

    def test_failed_baseline_stops_without_sampling(self):
        process = child()
        try:
            def unexpected(_):
                raise AssertionError('sample should not run')
            result = watch_owned_process(process, baseline=state(free=19), sampler=unexpected,
                                         timeout_seconds=3, interval_seconds=0.05,
                                         sample_timeout_seconds=0.2, terminate_grace_seconds=0.1)
            self.assertEqual(result['stop_reason'], 'memory_pressure_limit')
            self.assertIsNotNone(process.returncode)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()

    def test_escalates_when_owned_child_ignores_termination(self):
        code = ('import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                'print("ready", flush=True); time.sleep(30)')
        process = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), 'ready')
            result = watch_owned_process(process, baseline=state(), sampler=lambda _: state(free=19),
                                         timeout_seconds=3, interval_seconds=0.05,
                                         sample_timeout_seconds=0.2, terminate_grace_seconds=0.1)
            self.assertEqual(result['stop_reason'], 'memory_pressure_limit')
            self.assertEqual(process.returncode, -9)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()
            process.stdout.close()


if __name__ == '__main__':
    unittest.main()

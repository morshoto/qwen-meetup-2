"""A watchdog may signal only its explicitly owned fixture process group."""
import os
import subprocess
import sys
import threading
import time
import unittest

from src.arc.rsi_process_monitor import watch_owned_process
from src.arc.rsi_loss_watch import TrainingLossWatch


def state(*, free=70):
    return {'monotonic_seconds': time.monotonic(), 'disk_free_bytes': 30 * 1024**3,
            'memory_free_percent': free, 'swap_used_mib': 0, 'process_tree_rss_bytes': 0}


def child(seconds=30, *, session=True):
    return subprocess.Popen([sys.executable, '-c', f'import time; time.sleep({seconds})'],
                            start_new_session=session, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class OwnedMonitorTests(unittest.TestCase):
    def test_timed_out_raw_identity_sample_never_updates_tracker(self):
        process = child()
        finished = threading.Event()
        applied = []
        def slow_raw(_):
            time.sleep(0.3)
            finished.set()
            return ['late rows']
        try:
            result = watch_owned_process(process, baseline=state(), sampler=lambda _: state(),
                identity_sampler=slow_raw, identity_processor=lambda rows: applied.append(rows),
                timeout_seconds=3, interval_seconds=0.05, sample_timeout_seconds=0.05,
                terminate_grace_seconds=0.1)
            self.assertEqual(result['stop_reason'], 'identity_monitor_unavailable')
            self.assertTrue(finished.wait(2))
            self.assertEqual(applied, [])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()

    def test_owned_fixture_process_is_sampled_by_tracker(self):
        from src.arc.rsi_descendant_tracking import begin_owned_observation, sample_process_rows
        process = child(0.8)
        try:
            tracker, initial = begin_owned_observation(process)
            result = watch_owned_process(process, baseline=state(), sampler=lambda _: state(),
                identity_sampler=lambda _: sample_process_rows(), identity_processor=tracker.observe, timeout_seconds=3,
                interval_seconds=0.05, sample_timeout_seconds=1, terminate_grace_seconds=0.1)
            self.assertIn(process.pid, initial['live_observed_pids'])
            self.assertIn(result['stop_reason'], ('process_exited', 'identity_monitor_unavailable'))
            self.assertGreater(len(result['identity_observations']), 0)
            self.assertFalse(result['cleanup_observation']['group_cleanup_verified'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()

    def test_identity_sampler_failure_stops_owned_fixture(self):
        process = child()
        try:
            result = watch_owned_process(process, baseline=state(), sampler=lambda _: state(),
                identity_sampler=lambda _: None, timeout_seconds=3, interval_seconds=0.05,
                sample_timeout_seconds=0.2, terminate_grace_seconds=0.1)
            self.assertEqual(result['stop_reason'], 'identity_monitor_unavailable')
            self.assertIsNotNone(process.returncode)
            self.assertFalse(result['escaped_processes_excluded'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()

    def test_identity_sampler_records_evidence_but_cannot_claim_cleanup(self):
        process = child(0.2)
        observation = {'live_observed_pids': [process.pid], 'zombie_observed_pids': [],
                       'missing_observed_pids': [], 'reused_pids': [],
                       'cleanup_verified': False, 'escaped_processes_excluded': False}
        try:
            result = watch_owned_process(process, baseline=state(), sampler=lambda _: state(),
                identity_sampler=lambda _: observation, timeout_seconds=3, interval_seconds=0.05,
                sample_timeout_seconds=0.2, terminate_grace_seconds=0.1)
            self.assertEqual(result['stop_reason'], 'process_exited')
            self.assertGreater(len(result['identity_observations']), 0)
            self.assertFalse(result['identity_observations'][0]['cleanup_verified'])
            self.assertFalse(result['escaped_processes_excluded'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()

    def test_streamed_nonfinite_loss_stops_running_fixture(self):
        script = 'import time; print("Iter 1: Train loss nan, Learning Rate 1e-4", flush=True); time.sleep(30)'
        process = subprocess.Popen([sys.executable, '-c', script], start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        watch, ready = TrainingLossWatch(), threading.Event()
        def consume():
            watch.feed(process.stdout.readline())
            ready.set()
        worker = threading.Thread(target=consume)
        worker.start()
        try:
            self.assertTrue(ready.wait(3))
            result = watch_owned_process(process, baseline=state(), sampler=lambda _: state(),
                health_sampler=lambda _: watch.snapshot(), timeout_seconds=3, interval_seconds=0.05,
                sample_timeout_seconds=0.2, terminate_grace_seconds=0.1)
            self.assertEqual(result['stop_reason'], 'nonfinite_loss')
            self.assertIsNotNone(process.returncode)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()
            worker.join(timeout=3)
            process.stdout.close()

    def test_loss_health_stops_fixture_and_unavailable_fails_closed(self):
        for health, expected in ((lambda _: {'status': 'nonfinite_loss'}, 'nonfinite_loss'),
                                 (lambda _: {'status': 'unavailable'}, 'health_monitor_unavailable'),
                                 (lambda _: {'status': 'unknown'}, 'health_monitor_unavailable')):
            process = child()
            try:
                result = watch_owned_process(process, baseline=state(), sampler=lambda _: state(),
                    health_sampler=health, timeout_seconds=3, interval_seconds=0.05,
                    sample_timeout_seconds=0.2, terminate_grace_seconds=0.1)
                self.assertEqual(result['stop_reason'], expected)
                self.assertIsNotNone(process.returncode)
                self.assertFalse(result['training_verified'])
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, 9)
                    process.wait()

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

    def test_registered_model_limits_stop_fixture(self):
        limits = {'timeout_seconds': 3, 'minimum_disk_gib': 20, 'minimum_free_memory_percent': 60,
                  'maximum_swap_growth_mib': 512, 'maximum_process_rss_gib': 32, 'sample_interval_seconds': 1}
        process = child()
        try:
            result = watch_owned_process(process, baseline=state(), sampler=lambda _: state(free=55),
                                         timeout_seconds=3, interval_seconds=0.05,
                                         sample_timeout_seconds=0.2, terminate_grace_seconds=0.1,
                                         resource_limits=limits)
            self.assertEqual(result['stop_reason'], 'memory_pressure_limit')
            self.assertIsNotNone(process.returncode)
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
            self.assertGreaterEqual(result['elapsed_seconds'], 0)
            self.assertLess(result['elapsed_seconds'], 3)
            self.assertFalse(result['training_verified'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()

    def test_exit_records_group_listing_without_claiming_cleanup(self):
        process = child(0.2)
        try:
            result = watch_owned_process(process, baseline=state(), sampler=lambda _: state(),
                timeout_seconds=3, interval_seconds=0.05, sample_timeout_seconds=0.2,
                terminate_grace_seconds=0.1)
            evidence = result['cleanup_observation']
            self.assertIn(evidence['status'], ('process_group_listing_unverified', 'group_observation_unavailable'))
            self.assertFalse(evidence['group_cleanup_verified'])
            self.assertFalse(evidence['escaped_processes_excluded'])
            self.assertFalse(result['escaped_processes_excluded'])
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

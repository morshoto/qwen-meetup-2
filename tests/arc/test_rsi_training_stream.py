import os
import platform
from pathlib import Path
import subprocess
import sys
import unittest

from src.arc.rsi_training_stream import watch_training_stream
from src.arc.rsi_preflight_review import review_preflight_observation
from src.arc.rsi_process_sampling import sample_baseline, sample_process
from tests.arc.test_rsi_process_monitor import state


def child(script):
    return subprocess.Popen([sys.executable, '-u', '-c', script], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


class TrainingStreamTests(unittest.TestCase):
    def run_child(self, script):
        process = child(script)
        try:
            self.baseline = state()
            return watch_training_stream(process, baseline=self.baseline, sampler=lambda _: state(),
                timeout_seconds=3, interval_seconds=0.05, sample_timeout_seconds=0.2,
                terminate_grace_seconds=0.1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()
            process.stdout.close()

    def test_stream_watch_applies_registered_model_limits(self):
        limits = {'timeout_seconds': 3, 'minimum_disk_gib': 20, 'minimum_free_memory_percent': 60,
                  'maximum_swap_growth_mib': 512, 'maximum_process_rss_gib': 32, 'sample_interval_seconds': 1}
        process = child('import time; time.sleep(10)')
        try:
            result = watch_training_stream(process, baseline=state(), sampler=lambda _: state(free=55),
                timeout_seconds=3, interval_seconds=0.05, sample_timeout_seconds=0.2,
                terminate_grace_seconds=0.1, resource_limits=limits)
            self.assertEqual(result['stop_reason'], 'memory_pressure_limit')
            self.assertFalse(result['training_verified'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()
            process.stdout.close()

    @unittest.skipUnless(platform.system() == 'Darwin', 'macOS sampler required')
    def test_real_local_sampler_observes_synthetic_fixture(self):
        baseline = sample_baseline(Path('.'))
        process = child('import time; print("Iter 1: Train loss 0.125, x", flush=True); time.sleep(1.5)')
        try:
            result = watch_training_stream(process, baseline=baseline, sampler=lambda pid: sample_process(Path('.'), pid),
                timeout_seconds=4, interval_seconds=0.2, sample_timeout_seconds=1, terminate_grace_seconds=0.1)
            self.assertEqual(result['stop_reason'], 'process_exited')
            self.assertGreater(len(result['samples']), 0)
            self.assertTrue(any(item['process_tree_rss_bytes'] > 0 for item in result['samples']))
            self.assertTrue(result['stream_complete'])
            self.assertFalse(result['training_verified'])
            self.assertFalse(result['escaped_processes_excluded'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()
            process.stdout.close()

    def test_nonfinite_and_malformed_output_stop_child(self):
        for loss, expected in [('nan', 'nonfinite_loss'), ('???', 'health_monitor_unavailable')]:
            script = f'import time; print("Iter 1: Train loss {loss}, x", flush=True); time.sleep(30)'
            with self.subTest(loss=loss):
                result = self.run_child(script)
                self.assertEqual(result['stop_reason'], expected)
                self.assertFalse(result['training_verified'])

    def test_finite_output_is_observed_without_training_claim(self):
        result = self.run_child('print("Iter 1: Train loss 0.125, x", flush=True)')
        self.assertEqual(result['loss']['last_step'], 1)
        self.assertEqual(result['loss']['last_loss'], 0.125)
        self.assertTrue(result['stream_complete'])
        self.assertFalse(result['training_verified'])
        review = review_preflight_observation(baseline=self.baseline, observation=result)
        expected = ('preflight_observation_consistent_unverified' if result['group_signals_confirmed']
                    else 'preflight_observation_rejected')
        self.assertEqual(review['status'], expected)
        self.assertFalse(review['training_allowed'])

    def test_zero_exit_without_loss_is_not_a_preflight(self):
        result = self.run_child('print("finished", flush=True)')
        review = review_preflight_observation(baseline=self.baseline, observation=result)
        self.assertEqual(review['status'], 'preflight_observation_rejected')
        self.assertFalse(review['training_allowed'])

    def test_inherited_open_stream_is_not_reported_complete(self):
        script = ('import os,time; pid=os.fork(); '
                  '(os.setsid(),time.sleep(0.4),os._exit(0)) if pid == 0 else None')
        result = self.run_child(script)
        self.assertFalse(result['stream_complete'])
        self.assertFalse(result['escaped_processes_excluded'])
        self.assertFalse(result['training_verified'])


if __name__ == '__main__':
    unittest.main()

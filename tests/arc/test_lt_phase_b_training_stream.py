import os
import subprocess
import sys
import unittest

from src.arc.lt_phase_b_training_stream import watch_phase_b_training
from tests.arc.test_rsi_process_monitor import state


class TrainingStreamTests(unittest.TestCase):
    def watch(self, script, *, sampler=None, max_log_bytes=4 * 1024 * 1024):
        process = subprocess.Popen([sys.executable, '-u', '-c', script],
            start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        try:
            return watch_phase_b_training(process, baseline=state(), sampler=sampler or (lambda _: state()),
                timeout_seconds=3, interval_seconds=0.05, sample_timeout_seconds=0.2,
                terminate_grace_seconds=0.1, max_log_bytes=max_log_bytes)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()
            process.stdout.close()

    def test_finite_loss_and_full_log_remain_unverified_observations(self):
        result = self.watch('print("Iter 1: Train loss 0.125, ok", flush=True)')
        self.assertEqual(result['stop_reason'], 'process_exited')
        self.assertEqual(result['loss']['last_loss'], 0.125)
        self.assertEqual(result['loss']['last_step'], 1)
        self.assertTrue(result['stream_complete'])
        self.assertIn('Iter 1: Train loss 0.125', result['log'])
        self.assertFalse(result['training_verified'])
        self.assertFalse(result['escaped_processes_excluded'])

    def test_nonfinite_or_malformed_loss_stops_worker(self):
        for value, reason in [('nan', 'nonfinite_loss'), ('???', 'health_monitor_unavailable')]:
            with self.subTest(value=value):
                result = self.watch(f'import time; print("Iter 1: Train loss {value}, x", flush=True); time.sleep(5)')
                self.assertEqual(result['stop_reason'], reason)
                self.assertIsNotNone(result['exit_code'])
                self.assertIn(value, result['log'])

    def test_overflow_and_resource_pressure_stop_worker(self):
        result = self.watch('import time; print("x" * 512, flush=True); time.sleep(5)', max_log_bytes=128)
        self.assertEqual(result['stop_reason'], 'health_monitor_unavailable')
        self.assertFalse(result['stream_complete'])
        self.assertLessEqual(len(result['log'].encode()), 128)
        result = self.watch('import time; time.sleep(5)', sampler=lambda _: state(free=10))
        self.assertEqual(result['stop_reason'], 'memory_pressure_limit')


if __name__ == '__main__': unittest.main()

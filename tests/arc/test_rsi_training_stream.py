import os
import subprocess
import sys
import unittest

from src.arc.rsi_training_stream import watch_training_stream
from tests.arc.test_rsi_process_monitor import state


def child(script):
    return subprocess.Popen([sys.executable, '-u', '-c', script], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


class TrainingStreamTests(unittest.TestCase):
    def run_child(self, script):
        process = child(script)
        try:
            return watch_training_stream(process, baseline=state(), sampler=lambda _: state(),
                timeout_seconds=3, interval_seconds=0.05, sample_timeout_seconds=0.2,
                terminate_grace_seconds=0.1)
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

    def test_inherited_open_stream_is_not_reported_complete(self):
        script = ('import os,time; pid=os.fork(); '
                  '(os.setsid(),time.sleep(0.4),os._exit(0)) if pid == 0 else None')
        result = self.run_child(script)
        self.assertFalse(result['stream_complete'])
        self.assertFalse(result['escaped_processes_excluded'])
        self.assertFalse(result['training_verified'])


if __name__ == '__main__':
    unittest.main()

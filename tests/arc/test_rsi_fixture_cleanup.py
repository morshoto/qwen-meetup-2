import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from src.arc.rsi_process_monitor import _exited_unreaped
from src.arc.rsi_tiny_pipeline import cleanup_fixture_process


class FixtureCleanupTests(unittest.TestCase):
    def test_exited_child_is_reaped_if_group_observation_fails_without_signal(self):
        process = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
        deadline = time.monotonic() + 3
        while _exited_unreaped(process) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(_exited_unreaped(process))
        with patch('src.arc.rsi_group_observation.observe_exited_group', side_effect=ValueError('unavailable')), \
             patch('src.arc.rsi_tiny_pipeline._stop_group', side_effect=AssertionError('unverified group signalled')):
            result = cleanup_fixture_process(process)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(result['status'], 'leader_reaped_group_unverified')
        self.assertFalse(result['cleanup_verified'])
        self.assertFalse(result['escaped_processes_excluded'])

    def test_live_owned_session_is_reaped_without_poll(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
                                   start_new_session=True)
        with patch.object(process, 'poll', side_effect=AssertionError('poll reaps')):
            result = cleanup_fixture_process(process)
        self.assertIsNotNone(process.returncode)
        self.assertFalse(result['cleanup_verified'])
        self.assertFalse(result['escaped_processes_excluded'])
        self.assertEqual(cleanup_fixture_process(process)['status'], 'already_reaped_unverified')

    def test_exited_unreaped_leader_is_handled_without_poll(self):
        process = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
        deadline = time.monotonic() + 3
        while _exited_unreaped(process) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(_exited_unreaped(process))
        with patch.object(process, 'poll', side_effect=AssertionError('poll reaps')):
            result = cleanup_fixture_process(process)
        self.assertEqual(process.returncode, 0)
        self.assertFalse(result['cleanup_verified'])

    def test_non_session_child_is_not_signalled(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
        try:
            with self.assertRaises(ValueError): cleanup_fixture_process(process)
            self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=3)

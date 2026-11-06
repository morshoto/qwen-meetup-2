import subprocess
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from src.arc.http_backend import BackendError
from src.arc.lt_checked_sandbox import CheckedSandbox

class CheckedSandboxTests(unittest.TestCase):
    def test_records_absence(self):
        s = CheckedSandbox()
        with patch.object(s.inner, 'run', return_value=[[1]]), patch('src.arc.lt_checked_sandbox.subprocess.run', return_value=SimpleNamespace(stdout='')):
            self.assertEqual(s.run('code', [[1]]), [[1]])
        self.assertEqual(s.observations[0]['cleanup'], 'absent')
    def test_daemon_failure_stops(self):
        s = CheckedSandbox()
        with patch.object(s.inner, 'run', return_value=[[1]]), patch('src.arc.lt_checked_sandbox.subprocess.run', side_effect=subprocess.TimeoutExpired('docker', 1)):
            with self.assertRaises(BackendError): s.run('code', [[1]])
        self.assertEqual(s.observations[0]['cleanup'], 'unverified')

"""Darwin fixture checks for the proposed fork restriction; no model execution."""
import os
import platform
import subprocess
import sys
import unittest

from src.arc.rsi_canary_containment import fork_restricted_command


class CanaryContainmentTests(unittest.TestCase):
    def test_command_is_explicit_and_does_not_authorize_a_run(self):
        value = fork_restricted_command([sys.executable, '-c', 'pass'])
        self.assertEqual(value['argv'][:2], ['/usr/bin/sandbox-exec', '-p'])
        self.assertFalse(value['execution_ready'])
        self.assertFalse(value['training_allowed'])
        self.assertFalse(value['containment_verified'])
        for invalid in ([], ['relative'], [sys.executable, '\x00'], [sys.executable, 1]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): fork_restricted_command(invalid)

    @unittest.skipUnless(platform.system() == 'Darwin' and os.path.exists('/usr/bin/sandbox-exec'), 'Darwin sandbox required')
    def test_fixture_threads_work_but_fork_and_spawn_fail(self):
        script = '''import os, subprocess, threading
threading.Thread(target=lambda: None).start()
for fn in (lambda: os.fork(), lambda: subprocess.run(['/usr/bin/true'], check=True)):
    try:
        fn()
    except PermissionError:
        continue
    raise SystemExit('child creation unexpectedly succeeded')
'''
        command = fork_restricted_command([sys.executable, '-c', script])
        result = subprocess.run(command['argv'], timeout=10, capture_output=True, text=True, start_new_session=True)
        self.assertEqual(result.returncode, 0, result.stderr)

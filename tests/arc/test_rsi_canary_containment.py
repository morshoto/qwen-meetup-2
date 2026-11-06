"""Darwin fixture checks for the proposed fork restriction; no model execution."""
import os
import platform
import json
import socket
import subprocess
import sys
import tempfile
import unittest

from src.arc.rsi_canary_containment import fork_restricted_command


class CanaryContainmentTests(unittest.TestCase):
    @unittest.skipUnless(platform.system() == 'Darwin' and os.path.exists('/usr/bin/sandbox-exec'), 'Darwin sandbox required')
    def test_inherited_socket_bypasses_network_deny_but_closed_fd_does_not(self):
        for inherited in (True, False):
            receiver, sender = socket.socketpair()
            with self.subTest(inherited=inherited), receiver, sender:
                receiver.settimeout(0.2)
                script = 'import os,sys; os.write(int(sys.argv[1]), b"fixture")'
                command = fork_restricted_command([sys.executable, '-c', script, str(sender.fileno())])
                result = subprocess.run(command['argv'], close_fds=True,
                    pass_fds=(sender.fileno(),) if inherited else (), capture_output=True,
                    text=True, timeout=5, start_new_session=True)
                if inherited:
                    self.assertEqual(result.returncode, 0)
                    self.assertEqual(receiver.recv(32), b'fixture')
                else:
                    self.assertNotEqual(result.returncode, 0)
                    with self.assertRaises(socket.timeout): receiver.recv(32)

    @unittest.skipUnless(platform.system() == 'Darwin' and os.path.exists('/usr/bin/sandbox-exec'), 'Darwin sandbox required')
    def test_fixture_cannot_contact_owned_unix_or_tcp_listener(self):
        script = ('import json,socket,sys; family=socket.AF_UNIX if sys.argv[1]=="unix" else socket.AF_INET; '
                  'address=json.loads(sys.argv[2]); address=tuple(address) if family==socket.AF_INET else address; '
                  's=socket.socket(family,socket.SOCK_STREAM); s.connect(address); s.sendall(b"fixture")')
        with tempfile.TemporaryDirectory(prefix='rsi-containment-') as root:
            for kind in ('unix', 'tcp'):
                with self.subTest(kind=kind), socket.socket(socket.AF_UNIX if kind == 'unix' else socket.AF_INET,
                                                           socket.SOCK_STREAM) as listener:
                    listener.bind(os.path.join(root, 'listener.sock') if kind == 'unix' else ('127.0.0.1', 0))
                    listener.listen(1)
                    listener.settimeout(0.2)
                    command = fork_restricted_command([sys.executable, '-c', script, kind,
                                                       json.dumps(listener.getsockname())])
                    result = subprocess.run(command['argv'], timeout=5, capture_output=True, text=True,
                                            start_new_session=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('PermissionError', result.stderr)
                    with self.assertRaises(socket.timeout):
                        listener.accept()

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

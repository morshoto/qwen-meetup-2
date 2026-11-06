import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_macos_probe import build_probe_profile


class ProfileTests(unittest.TestCase):
    def test_profile_is_explicitly_non_authorizing(self):
        profile = build_probe_profile(['/tmp/runtime'], ['/tmp/output'])
        self.assertIn('(literal "/")', profile['profile'])
        self.assertIn('(allow iokit-open)', profile['profile'])
        self.assertIn('(deny process-fork)', profile['profile'])
        self.assertFalse(profile['execution_ready'])
        self.assertFalse(profile['isolation_verified'])
        for paths in ([], ['/'], ['relative'], ['/tmp/unsafe\x00name']):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                build_probe_profile(paths, ['/tmp/output'])

    @unittest.skipUnless(platform.system() == 'Darwin' and Path('/usr/bin/sandbox-exec').exists(),
                         'Darwin sandbox required')
    def test_synthetic_aliases_and_inherited_descriptors_remain_visible(self):
        script = '''import json,os,sys
paths=json.loads(sys.argv[1]); result={}
for label,path in paths.items():
 try:
  with open(path,'rb') as stream: stream.read()
  result[label]='readable'
 except OSError: result[label]='denied'
try: os.read(int(sys.argv[2]),32); result['fd']='readable'
except OSError: result['fd']='denied'
try: os.write(int(sys.argv[3]),b'x'); result['socket']='writable'
except OSError: result['socket']='denied'
print(json.dumps(result))
'''
        with tempfile.TemporaryDirectory(prefix='lt-profile-') as directory:
            root = Path(os.path.realpath(directory))
            allowed = root / 'allowed'
            allowed.mkdir()
            secret = root / 'secret'
            secret.write_text('synthetic secret')
            public = allowed / 'public'
            public.write_text('synthetic public')
            outside = root / 'outside-hardlink'
            inside = allowed / 'inside-hardlink'
            symlink = allowed / 'secret-symlink'
            os.link(secret, outside)
            os.link(secret, inside)
            symlink.symlink_to(secret)
            paths = {'secret': str(secret), 'outside': str(outside), 'inside': str(inside),
                     'symlink': str(symlink), 'public': str(public)}
            python = '/opt/homebrew/bin/python3.12' if Path('/opt/homebrew/bin/python3.12').exists() else sys.executable
            dirs = [str(allowed), sys.prefix, sys.base_prefix,
                    str(Path(os.path.realpath(python)).parent.parent)]
            profile = build_probe_profile(dirs, [str(allowed)])['profile']
            for inherited in (False, True):
                receiver, sender = socket.socketpair()
                with self.subTest(inherited=inherited), secret.open('rb') as handle, receiver, sender:
                    command = ['/usr/bin/sandbox-exec', '-p', profile, python, '-I', '-B', '-c',
                               script, json.dumps(paths), str(handle.fileno()), str(sender.fileno())]
                    result = subprocess.run(command, pass_fds=(handle.fileno(), sender.fileno()) if inherited else (),
                                            close_fds=True, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    actual = json.loads(result.stdout)
                    self.assertEqual(actual['public'], 'readable')
                    self.assertEqual(actual['secret'], 'denied')
                    self.assertEqual(actual['outside'], 'denied')
                    self.assertEqual(actual['symlink'], 'denied')
                    self.assertEqual(actual['inside'], 'readable')
                    self.assertEqual(actual['fd'], 'readable' if inherited else 'denied')
                    self.assertEqual(actual['socket'], 'writable' if inherited else 'denied')

    @unittest.skipUnless(platform.system() == 'Darwin' and Path('/usr/bin/sandbox-exec').exists(),
                         'Darwin sandbox required')
    def test_synthetic_fork_and_new_tcp_connection_are_denied(self):
        script = """import json,os,socket,sys
result={}
try:
 pid=os.fork()
 if pid==0: os._exit(0)
 os.waitpid(pid,0); result['fork']='allowed'
except OSError: result['fork']='denied'
try:
 s=socket.socket(); s.settimeout(.2); s.connect(('127.0.0.1',int(sys.argv[1]))); result['tcp']='connected'
except OSError: result['tcp']='denied'
print(json.dumps(result))
"""
        with tempfile.TemporaryDirectory(prefix='lt-profile-') as directory, socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            listener.settimeout(.2)
            python = '/opt/homebrew/bin/python3.12' if Path('/opt/homebrew/bin/python3.12').exists() else sys.executable
            dirs = [os.path.realpath(directory), sys.prefix, sys.base_prefix,
                    str(Path(os.path.realpath(python)).parent.parent)]
            profile = build_probe_profile(dirs, [os.path.realpath(directory)])['profile']
            result = subprocess.run(['/usr/bin/sandbox-exec', '-p', profile, python, '-I', '-B', '-c',
                                     script, str(listener.getsockname()[1])], capture_output=True, text=True,
                                    close_fds=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {'fork': 'denied', 'tcp': 'denied'})
            with self.assertRaises(socket.timeout): listener.accept()

    @unittest.skipUnless(platform.system() == 'Darwin' and Path('/usr/bin/sandbox-exec').exists(),
                         'Darwin sandbox required')
    def test_synthetic_listener_rule_allows_only_registered_port(self):
        script = """import json,socket,sys
result={}
for label,port in [('registered',int(sys.argv[1])),('other',int(sys.argv[2]))]:
 try:
  s=socket.socket(); s.bind(('127.0.0.1',port)); s.listen(1); result[label]='bound'; s.close()
 except OSError: result[label]='denied'
try:
 s=socket.socket(); s.settimeout(.2); s.connect(('127.0.0.1',int(sys.argv[3]))); result['outbound']='connected'
except OSError: result['outbound']='denied'
print(json.dumps(result))
"""
        with tempfile.TemporaryDirectory(prefix='lt-profile-') as directory, socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen(1)
            listener.settimeout(.2)
            with socket.socket() as first, socket.socket() as second:
                first.bind(('127.0.0.1', 0))
                second.bind(('127.0.0.1', 0))
                ports = [first.getsockname()[1], second.getsockname()[1]]
            python = '/opt/homebrew/bin/python3.12' if Path('/opt/homebrew/bin/python3.12').exists() else sys.executable
            dirs = [os.path.realpath(directory), sys.prefix, sys.base_prefix,
                    str(Path(os.path.realpath(python)).parent.parent)]
            profile = build_probe_profile(dirs, [os.path.realpath(directory)])['profile']
            endpoint = f'localhost:{ports[0]}'
            profile += f' (allow network-bind (local ip "{endpoint}")) (allow network-inbound (local ip "{endpoint}"))'
            result = subprocess.run(['/usr/bin/sandbox-exec', '-p', profile, python, '-I', '-B', '-c', script,
                                     str(ports[0]), str(ports[1]), str(listener.getsockname()[1])],
                                    capture_output=True, text=True, close_fds=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {'registered': 'bound', 'other': 'denied', 'outbound': 'denied'})
            with self.assertRaises(socket.timeout): listener.accept()

    @unittest.skipUnless(platform.system() == 'Darwin' and os.environ.get('LT_PROBE_MLX') == '1',
                         'requires explicit offline MLX probe')
    def test_synthetic_mlx_gpu_works_under_proposed_profile(self):
        script = ('import json,mlx.core as mx; x=mx.array([1,2,3]); y=x+2; mx.eval(y); '
                  'print(json.dumps({"device":str(mx.default_device()),"result":y.tolist()}))')
        with tempfile.TemporaryDirectory(prefix='lt-mlx-probe-') as directory:
            root = os.path.realpath(directory)
            roots = [root, sys.prefix, sys.base_prefix, os.path.expanduser('~/.cache/uv/archive-v0')]
            profile = build_probe_profile(roots, [root])['profile']
            env = {'PATH': '/usr/bin:/bin', 'TMPDIR': root, 'HOME': root}
            for sandboxed in (False, True):
                command = [sys.executable, '-I', '-B', '-c', script]
                if sandboxed:
                    command = ['/usr/bin/sandbox-exec', '-p', profile, *command]
                with self.subTest(sandboxed=sandboxed):
                    result = subprocess.run(command, env=env, close_fds=True, capture_output=True,
                                            text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    actual = json.loads(result.stdout)
                    self.assertEqual(actual['result'], [3, 4, 5])
                    self.assertIn('gpu', actual['device'])


if __name__ == '__main__': unittest.main()

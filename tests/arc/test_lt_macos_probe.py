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
    def test_synthetic_listener_rule_allows_registered_and_rejects_other(self):
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

    @unittest.skipUnless(platform.system() == 'Darwin' and Path('/usr/bin/sandbox-exec').exists(),
                         'Darwin sandbox required')
    def test_synthetic_listener_exchanges_bytes_on_registered_port(self):
        import select
        script = ("import socket,sys; s=socket.socket(); s.settimeout(5); "
                  "s.bind(('127.0.0.1',int(sys.argv[1]))); s.listen(1); print('ready',flush=True); "
                  "c,_=s.accept(); c.recv(1); c.sendall(b'y'); c.close(); s.close()")
        with tempfile.TemporaryDirectory(prefix='lt-profile-') as directory:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
            python = '/opt/homebrew/bin/python3.12' if Path('/opt/homebrew/bin/python3.12').exists() else sys.executable
            dirs = [os.path.realpath(directory), sys.prefix, sys.base_prefix,
                    str(Path(os.path.realpath(python)).parent.parent)]
            profile = build_probe_profile(dirs, [os.path.realpath(directory)])['profile']
            endpoint = f'localhost:{port}'
            profile += f' (allow network-bind (local ip "{endpoint}")) (allow network-inbound (local ip "{endpoint}"))'
            child = subprocess.Popen(['/usr/bin/sandbox-exec', '-p', profile, python, '-I', '-B', '-c',
                                      script, str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, close_fds=True)
            try:
                self.assertTrue(select.select([child.stdout], [], [], 5)[0], 'listener did not become ready')
                self.assertEqual(child.stdout.readline().strip(), 'ready')
                with socket.create_connection(('127.0.0.1', port), timeout=3) as client:
                    client.sendall(b'x')
                    self.assertEqual(client.recv(1), b'y')
                _, stderr = child.communicate(timeout=6)
                self.assertEqual(child.returncode, 0, stderr)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                if child.stdout: child.stdout.close()
                if child.stderr: child.stderr.close()

    @unittest.skipUnless(platform.system() == 'Darwin' and os.environ.get('LT_PROBE_MLX') == '1',
                         'requires explicit offline MLX probe')
    def test_synthetic_mlx_gpu_works_under_proposed_profile(self):
        script = ('import json,mlx_lm,mlx.core as mx; from mlx_lm.models.qwen3 import Model,ModelArgs; '
                  'x=mx.array([1,2,3]); y=x+2; '
                  'a=ModelArgs(model_type="qwen3",hidden_size=16,num_hidden_layers=1,intermediate_size=32,'
                  'num_attention_heads=2,rms_norm_eps=1e-6,vocab_size=32,num_key_value_heads=2,'
                  'max_position_embeddings=32,rope_theta=10000,head_dim=8,tie_word_embeddings=True); '
                  'logits=Model(a)(mx.array([[1,2]])); mx.eval(y,logits); '
                  'print(json.dumps({"device":str(mx.default_device()),"result":y.tolist(),'
                  '"shape":list(logits.shape),"finite":bool(mx.all(mx.isfinite(logits)).item())}))')
        with tempfile.TemporaryDirectory(prefix='lt-mlx-probe-') as directory:
            root = os.path.realpath(directory)
            cache_base = subprocess.check_output(['/usr/bin/getconf', 'DARWIN_USER_CACHE_DIR'], text=True).strip()
            metal_cache = os.path.realpath(os.path.join(cache_base, 'org.python.python', 'com.apple.metalfe'))
            roots = [root, sys.prefix, sys.base_prefix, os.path.expanduser('~/.cache/uv/archive-v0'), metal_cache]
            profile = build_probe_profile(roots, [root, metal_cache])['profile']
            profile += ' (allow file-issue-extension (subpath ' + json.dumps(metal_cache) + '))'
            env = {'PATH': '/usr/bin:/bin', 'TMPDIR': root, 'HOME': root}
            for sandboxed in (False, True):
                command = [sys.executable, '-I', '-B', '-c', script]
                if sandboxed:
                    command = ['/usr/bin/sandbox-exec', '-p', profile, *command]
                with self.subTest(sandboxed=sandboxed):
                    result = subprocess.run(command, env=env, cwd=root, close_fds=True, capture_output=True,
                                            text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    actual = json.loads(result.stdout)
                    self.assertEqual(actual['result'], [3, 4, 5])
                    self.assertEqual(actual['shape'], [1, 2, 32])
                    self.assertTrue(actual['finite'])
                    self.assertIn('gpu', actual['device'])


if __name__ == '__main__': unittest.main()


@unittest.skipUnless(platform.system() == 'Darwin' and Path('/usr/bin/sandbox-exec').exists(),
                     'Darwin sandbox required')
class MachServiceProbeTests(unittest.TestCase):
    def test_synthetic_profile_can_message_independent_mach_service(self):
        import ctypes
        import uuid

        system = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        uint = ctypes.c_uint32
        task = uint.in_dll(system, 'mach_task_self_').value
        bootstrap = uint.in_dll(system, 'bootstrap_port').value
        port = uint()
        self.assertEqual(system.mach_port_allocate(task, 1, ctypes.byref(port)), 0)
        try:
            self.assertEqual(system.mach_port_insert_right(task, port.value, port.value, 20), 0)
            name = 'com.lt-probe.' + uuid.uuid4().hex
            self.assertEqual(system.bootstrap_register(bootstrap, name.encode(), port.value), 0)
            script = '''import ctypes,json,sys
L=ctypes.CDLL('/usr/lib/libSystem.B.dylib'); u=ctypes.c_uint32
b=u.in_dll(L,'bootstrap_port').value; p=u()
r=L.bootstrap_look_up(b,sys.argv[1].encode(),ctypes.byref(p))
class H(ctypes.Structure):
 _fields_=[(n,u) for n in ('bits','size','remote','local','voucher','id')]
h=H(19,ctypes.sizeof(H),p.value,0,0,424242)
s=L.mach_msg(ctypes.byref(h),1,ctypes.sizeof(h),0,0,0,0) if r==0 else None
print(json.dumps({'lookup':r,'send':s}))
'''
            with tempfile.TemporaryDirectory(prefix='lt-mach-') as directory:
                python = '/opt/homebrew/bin/python3.12' if Path('/opt/homebrew/bin/python3.12').exists() else sys.executable
                dirs = [os.path.realpath(directory), sys.prefix, sys.base_prefix,
                        str(Path(os.path.realpath(python)).parent.parent)]
                profile = build_probe_profile(dirs, [os.path.realpath(directory)])['profile']
                result = subprocess.run(['/usr/bin/sandbox-exec', '-p', profile, python, '-I', '-B',
                                         '-c', script, name], capture_output=True, text=True,
                                        close_fds=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), {'lookup': 0, 'send': 0})
                buffer = ctypes.create_string_buffer(1024)
                self.assertEqual(system.mach_msg(buffer, 2 | 0x100, 0, len(buffer), port.value, 300, 0), 0)
                self.assertEqual(ctypes.cast(buffer, ctypes.POINTER(uint))[5], 424242)
        finally:
            system.mach_port_destroy(task, port.value)

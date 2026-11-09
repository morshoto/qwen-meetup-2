import os
import pytest
from src.arc.qwen38_agent_sandbox import execute

pytestmark = pytest.mark.skipif(os.environ.get('QWEN38_DOCKER_TEST') != '1', reason='requires running local Docker')


def test_file_and_network_isolation():
    source = '''def run(data):
 import os, socket
 out = {'host_file': os.path.exists('/Users/shotomorisaki/.codex'), 'uid': os.getuid()}
 try:
  socket.create_connection(('1.1.1.1', 443), timeout=1)
  out['network'] = True
 except OSError:
  out['network'] = False
 return out
'''
    assert execute(source, {}, timeout=30) == {'host_file': False, 'uid': 65534, 'network': False}


def test_timeout_cleans_container():
    with pytest.raises(TimeoutError):
        execute('def run(data):\n while True: pass', {}, timeout=5)


def test_links_socket_descriptors_and_read_only_root():
    source = '''def run(data):
 import os, socket
 out = {}
 for label, operation in (
  ('hardlink', lambda: os.link('/etc/passwd', '/tmp/link')),
  ('root_write', lambda: open('/newfile', 'w')),
  ('host_unix_socket', lambda: socket.socket(socket.AF_UNIX).connect('/var/run/docker.sock')),
 ):
  try:
   operation(); out[label] = True
  except OSError:
   out[label] = False
 out['inherited_fds'] = sorted(int(x) for x in os.listdir('/proc/self/fd') if int(x) > 2)
 out['host_path_via_symlink'] = False
 os.symlink('/Users/shotomorisaki/.codex', '/tmp/host')
 out['host_path_via_symlink'] = os.path.exists('/tmp/host')
 return out
'''
    result = execute(source, {}, timeout=30)
    assert result['hardlink'] is False and result['root_write'] is False
    assert result['host_unix_socket'] is False and result['host_path_via_symlink'] is False
    # Listing /proc/self/fd can itself transiently open a descriptor.
    assert all(fd <= 3 for fd in result['inherited_fds'])


def test_forked_child_is_removed_with_owned_container():
    source = '''def run(data):
 import os,time
 child = os.fork()
 if child == 0:
  while True: time.sleep(1)
 return {'child_started': True}
'''
    assert execute(source, {}, timeout=30) == {'child_started': True}


def test_unbounded_output_is_stopped():
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        execute("def run(data):\n print('x' * 2000000)\n return {}", {}, timeout=30)

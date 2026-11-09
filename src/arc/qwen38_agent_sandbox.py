"""Disposable no-network container for a model-authored ARC agent module.

The model runs on the host, never inside this container. Do not pass solutions,
credentials, source directories, host sockets, or inherited descriptors as input.
"""
from __future__ import annotations
import json
import resource
import subprocess
import tempfile
import uuid

IMAGE = 'sha256:a3013c6339a68a667aa9a5f26377693a1f4357a7f7a12db6470a8f06732b91f1'
WRAPPER = r'''
import json, sys
request = json.loads(sys.stdin.read())
namespace = {'__name__': 'model_agent'}
exec(compile(request['source'], '<agent>', 'exec'), namespace)
response = namespace['run'](request['input'])
print(json.dumps(response, separators=(',', ':'), allow_nan=False))
'''


def execute(source, payload, timeout=10):
    if not isinstance(source, str) or len(source) > 20000:
        raise ValueError('invalid module source')
    name = 'qwen38-agent-' + uuid.uuid4().hex
    command = ['docker', 'run', '--name', name, '--rm', '--interactive', '--network', 'none',
               '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--pids-limit', '16', '--memory', '256m', '--cpus', '1', '--user', '65534:65534',
               '--tmpfs', '/tmp:rw,noexec,nosuid,size=8m', '--entrypoint', 'python',
               IMAGE, '-I', '-B', '-c', WRAPPER]
    def limit_output():
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    try:
        with tempfile.TemporaryFile(mode='w+t') as stdout, tempfile.TemporaryFile(mode='w+t') as stderr:
            try:
                completed = subprocess.run(command, input=json.dumps({'source': source, 'input': payload}),
                                           stdout=stdout, stderr=stderr, text=True, timeout=timeout,
                                           check=False, preexec_fn=limit_output)
            except subprocess.TimeoutExpired as exc:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=10, check=False)
                raise TimeoutError('agent exceeded time limit') from exc
            stdout.seek(0); output = stdout.read(100001)
            stderr.seek(0); errors = stderr.read(100001)
    finally:
        # --rm should remove a finished container; remove only this uniquely named owned container if necessary.
        check = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=10, check=False)
        if check.returncode == 0:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=10, check=False)
            check = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=10, check=False)
        if check.returncode == 0 or 'No such' not in check.stderr:
            raise RuntimeError('owned container cleanup unverified: ' + check.stderr[-300:])
    if completed.returncode != 0:
        raise RuntimeError('agent process failed: ' + errors[-2000:])
    if len(output) > 100000:
        raise ValueError('agent output too large')
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError('agent returned invalid JSON') from exc

"""Observe a synthetic Metal workload; never launch a real model or ARC task."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

from src.arc.lt_macos_probe import build_probe_profile
from src.arc.lt_metal_observation import (compare_observations, merge_process_identity, observe_process_identity,
                                           observe_responsibility, parse_metal_processes)

_SCRIPT = (
    'import time,mlx.core as mx; from mlx_lm.models.qwen3 import Model,ModelArgs; '
    'a=ModelArgs(model_type="qwen3",hidden_size=16,num_hidden_layers=1,intermediate_size=32,'
    'num_attention_heads=2,rms_norm_eps=1e-6,vocab_size=32,num_key_value_heads=2,'
    'max_position_embeddings=32,rope_theta=10000,head_dim=8,tie_word_embeddings=True); '
    'y=Model(a)(mx.array([[1,2]])); mx.eval(y); '
    'print("synthetic-forward-complete",flush=True); time.sleep(.5)'
)


def _snapshot() -> dict:
    listing = subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,command='],
                                      text=True, timeout=5)
    return parse_metal_processes(listing)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix='lt-metal-lifecycle-') as tmp:
        root = os.path.realpath(tmp)
        cache_base = subprocess.check_output(['/usr/bin/getconf', 'DARWIN_USER_CACHE_DIR'], text=True, timeout=5).strip()
        cache = os.path.realpath(os.path.join(cache_base, 'org.python.python', 'com.apple.metalfe'))
        reads = [root, sys.prefix, sys.base_prefix, os.path.expanduser('~/.cache/uv/archive-v0'), cache]
        profile = build_probe_profile(reads, [root, cache])['profile']
        profile += ' (allow file-issue-extension (subpath ' + json.dumps(cache) + '))'
        before = _snapshot()
        observed = dict(before)
        child = subprocess.Popen(['/usr/bin/sandbox-exec', '-p', profile, sys.executable, '-I', '-B', '-c', _SCRIPT],
            cwd=root, env={'PATH': '/usr/bin:/bin', 'TMPDIR': root, 'HOME': root},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, close_fds=True, start_new_session=True)
        samples = 0
        responsibilities = {}
        identities = {}
        try:
            child_identity = observe_process_identity(child.pid)
            child_responsibility = observe_responsibility(child.pid)
            deadline = time.monotonic() + 20
            while child.poll() is None and time.monotonic() < deadline:
                current = _snapshot()
                observed.update(current)
                for pid in set(current) - set(before):
                    if pid not in responsibilities or responsibilities[pid]['status'] == 'unavailable':
                        responsibilities[pid] = observe_responsibility(pid)
                    identities[pid] = merge_process_identity(identities.get(pid), observe_process_identity(pid))
                samples += 1
                time.sleep(.05)
            if child.poll() is None:
                child.kill()
            stdout, _ = child.communicate(timeout=5)
            after = _snapshot()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
        report = compare_observations(before, observed, after)
        report.update(child_status='completed' if child.returncode == 0 and stdout.strip() == 'synthetic-forward-complete'
                      else 'failed_or_timed_out', child_pid=child.pid, samples=samples,
                      observed_processes={str(pid): observed[pid] for pid in report['newly_observed']},
                      child_identity=child_identity, child_responsibility=child_responsibility,
                      service_identities={str(pid): identities[pid] for pid in report['newly_observed']},
                      service_responsibilities={str(pid): responsibilities[pid] for pid in report['newly_observed']})
        print(json.dumps(report, sort_keys=True))
        return 0 if report['child_status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

import json
import os
import sys

import pytest

from src.arc.qwen38_weight_supervisor import run_supervised


def good_probe():
    return 80, 0


def test_supervisor_captures_private_streams_and_reaps_child(tmp_path):
    script = 'import json,sys; x=json.load(sys.stdin); print(json.dumps({"received":x["value"]})); print("diagnostic",file=sys.stderr)'
    result = run_supervised([sys.executable, '-c', script], {'value': 'visible'}, tmp_path / 'attempt',
                            timeout_seconds=5, probe=good_probe, poll_seconds=.02)
    assert result['status'] == 'complete' and result['exit_code'] == 0
    assert result['process_group_gone'] is True
    assert json.loads((tmp_path / 'attempt/stdout.log').read_text()) == {'received': 'visible'}
    assert 'diagnostic' in (tmp_path / 'attempt/stderr.log').read_text()
    assert os.stat(tmp_path / 'attempt/request.json').st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        run_supervised([sys.executable, '-c', script], {}, tmp_path / 'attempt', timeout_seconds=5, probe=good_probe)


def test_supervisor_times_out_and_cleans_process_group(tmp_path):
    result = run_supervised([sys.executable, '-c', 'import time; time.sleep(10)'], {}, tmp_path / 'timeout',
                            timeout_seconds=.1, probe=good_probe, poll_seconds=.02)
    assert result['status'] == 'error' and result['reason'] == 'timeout'
    assert result['process_group_gone'] is True
    assert result['elapsed_seconds'] < 5


def test_supervisor_fails_closed_on_resource_probe_failure(tmp_path):
    result = run_supervised([sys.executable, '-c', 'import time; time.sleep(10)'], {}, tmp_path / 'probe',
                            timeout_seconds=5, probe=lambda: (None, None), poll_seconds=.02)
    assert result['status'] == 'error' and result['reason'] == 'memory_probe_failed'
    assert result['process_group_gone'] is True


def test_supervisor_rejects_fast_oversized_worker_output(tmp_path):
    result = run_supervised([sys.executable, '-c', 'import sys; sys.stdout.write("x" * 100000)'], {},
                            tmp_path / 'output', timeout_seconds=5, probe=good_probe,
                            poll_seconds=.1, max_output_bytes=1024)
    assert result['status'] == 'error' and result['reason'] == 'output_limit'
    assert result['process_group_gone'] is True


def test_supervisor_cleans_descendants_when_leader_exits(tmp_path):
    script = 'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); print("spawned",flush=True)'
    result = run_supervised([sys.executable, '-c', script], {}, tmp_path / 'descendant',
                            timeout_seconds=5, probe=good_probe, poll_seconds=.02)
    assert result['status'] == 'error' and result['reason'] == 'remaining_process_group'
    assert result['process_group_gone'] is True

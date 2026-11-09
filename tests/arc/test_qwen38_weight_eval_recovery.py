import hashlib
import json
from pathlib import Path

import pytest


def test_recovery_requires_exact_pre_inference_failure(tmp_path, monkeypatch):
    from src.cli import qwen38_weight_evaluate_recovery as recovery
    root = tmp_path / 'evaluation'
    worker = root / 'adapter-v1/workers/000000'
    journal = root / 'adapter-v1/journal'
    worker.mkdir(parents=True); journal.mkdir(parents=True)
    files = {
        'evaluation-v1.json': b'old descriptor',
        'adapter-v1/termination.json': b'old stop',
        'adapter-v1/journal/000000.start.json': b'old start',
        'adapter-v1/journal/000000.result.json': b'old result',
        'adapter-v1/workers/000000/execution.json': b'old execution',
        'adapter-v1/workers/000000/stderr.log': b'old error',
        'adapter-v1/workers/000000/stdout.log': b'',
    }
    for name, content in files.items(): (root / name).write_bytes(content)
    monkeypatch.setattr(recovery, 'FAILURE_HASHES', {name: hashlib.sha256(content).hexdigest() for name, content in files.items()})
    evidence = recovery.verify_preinference_failure(root)
    assert evidence['status'] == 'preinference_failure_verified'
    (worker / 'response.json').write_text('{}')
    with pytest.raises(ValueError): recovery.verify_preinference_failure(root)
    (worker / 'response.json').unlink()
    (journal / '000001.start.json').write_text('{}')
    with pytest.raises(ValueError): recovery.verify_preinference_failure(root)

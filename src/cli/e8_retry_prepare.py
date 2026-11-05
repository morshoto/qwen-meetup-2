"""Explicitly authorized 14B or 32B download retry; never starts inference."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from src.arc.e8_config import E8Settings, MODELS
from src.arc.e8_resource import check_resources, snapshot
from src.pipelines.e5 import _verify_mutable_file
from src.pipelines.e6g import _seal_g, _verify_g
from src.pipelines.e6f import _sha
from src.pipelines.e8 import (_base, _check_no_other_model_server, _model_dir,
                              _model_manifest, _source_pool)


def prepare(source: Path, attempt: Path, campaign: Path, *, live: bool, seconds: int,
            approve_retry: bool, stage: str = '14b') -> dict:
    if not live or not approve_retry or seconds not in (1200, 3600):
        raise ValueError('retry requires --live, --approve-retry-download and an approved download limit')
    if stage not in ('14b', '32b'):
        raise ValueError('retry stage must be 14b or 32b')
    namespace = Path('artifacts/e8').resolve()
    source, attempt, campaign = source.resolve(), attempt.resolve(), campaign.resolve()
    if not all(p.is_relative_to(namespace) for p in (source, attempt, campaign)):
        raise ValueError('all paths must be within artifacts/e8')
    if attempt.exists() or campaign.exists():
        raise FileExistsError('retry paths must be new')
    lock = source / 'protocol.lock.yml'
    _verify_mutable_file(lock); _verify_g(source / '8b'); _verify_g(source / '14b')
    if stage == '32b': _verify_g(source / '32b')
    if (source / 'protocol.sha256').read_text().strip() != _sha(lock):
        raise ValueError('original protocol hash mismatch')
    protocol = json.loads(lock.read_text())
    _verify_mutable_file(source / 'initial_pool.json')
    settings = E8Settings(**protocol['settings'])
    pool, _, _, _, verifier_identity = _source_pool(settings, _base(settings), protocol['sandbox_image_id'])
    if (json.loads((source / 'initial_pool.json').read_text()) != pool or
            protocol['verifier_identity'] != verifier_identity):
        raise ValueError('original pool or verifier identity mismatch')
    if protocol['settings']['profile'] != 'smoke': raise ValueError('retry only applies to Smoke')
    if json.loads((source / stage / 'result.json').read_text())['status'] == 'completed':
        raise ValueError(f'{stage} stage already completed')
    if stage == '32b' and any(json.loads((source / prior / 'result.json').read_text())['status'] != 'completed' for prior in ('8b', '14b')):
        raise ValueError('32b retry requires completed 8b and 14b stages')
    for name, digest in protocol['dependencies'].items():
        if _sha(Path(name)) != digest: raise ValueError('original protocol dependency changed')
    _check_no_other_model_server()
    model, revision, size = MODELS[stage]
    current = snapshot(source)
    limit = protocol['settings']
    reason = check_resources(current, current, min_free_percent=limit['min_memory_free_percent'],
        max_swap_increase_mib=limit['max_swap_increase_mib'],
        min_disk_bytes=limit['min_free_disk_gib'] * 1024**3 + size)
    if reason: raise RuntimeError(reason)
    attempt.mkdir(parents=True, mode=0o700)
    start = time.monotonic()
    command = ['/tmp/qwen-p3a-mlx-venv/bin/python', '-c',
               'from huggingface_hub import snapshot_download; import sys; '
               'print(snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2]))', model, revision]
    result = {'model': model, 'revision': revision, 'original_protocol_sha256': _sha(lock),
              'download_timeout_seconds': seconds, 'inference_started': False, 'status': 'started'}
    process = subprocess.Popen(command)
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - start
            state = snapshot(source)
            reason = check_resources(current, state, min_free_percent=limit['min_memory_free_percent'],
                max_swap_increase_mib=limit['max_swap_increase_mib'],
                min_disk_bytes=limit['min_free_disk_gib'] * 1024**3)
            if reason or elapsed >= seconds:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise TimeoutError(reason or 'download_timeout')
            time.sleep(3)
        if process.returncode != 0: raise RuntimeError('download process exited unsuccessfully')
        result['model_manifest'] = _model_manifest(_model_dir(model, revision), revision)
        result['status'] = 'download_complete'
        campaign.mkdir(parents=True, mode=0o700)
        for name in ('protocol.lock.yml', 'protocol.lock.yml.sha256.json', 'protocol.sha256',
                     'initial_pool.json', 'initial_pool.json.sha256.json',
                     'split_manifest.json', 'split_manifest.json.sha256.json'):
            shutil.copy2(source / name, campaign / name)
        shutil.copytree(source / '8b', campaign / '8b')
        _verify_g(campaign / '8b')
        if stage == '32b':
            shutil.copytree(source / '14b', campaign / '14b')
            _verify_g(campaign / '14b')
        (campaign / 'retry-provenance.json').write_text(json.dumps({
            'source_campaign': str(source), 'source_8b_seal_sha256': _sha(source / '8b/seal.json'),
            'source_14b_seal_sha256': _sha(source / '14b/seal.json'),
            'source_32b_seal_sha256': _sha(source / '32b/seal.json') if stage == '32b' else None,
            'retry_stage': stage,
            'acquisition_attempt': str(attempt)}, indent=2) + '\n')
        (campaign / 'retry-provenance.json').chmod(0o600)
    except Exception as exc:
        result.update(status='download_incomplete', stop_reason=type(exc).__name__ + ': ' + str(exc))
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            if result['status'] == 'started': result['status'] = 'interrupted'
        result['elapsed_seconds'] = time.monotonic() - start
        result['after'] = snapshot(source)
        (attempt / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        (attempt / 'result.json').chmod(0o600)
        _seal_g(attempt)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-campaign', type=Path, required=True)
    p.add_argument('--attempt-dir', type=Path, required=True)
    p.add_argument('--new-campaign', type=Path, required=True)
    p.add_argument('--live', action='store_true')
    p.add_argument('--approve-retry-download', action='store_true')
    p.add_argument('--download-seconds', type=int, required=True)
    p.add_argument('--stage', choices=('14b', '32b'), default='14b')
    a = p.parse_args()
    print(json.dumps(prepare(a.source_campaign, a.attempt_dir, a.new_campaign,
        live=a.live, seconds=a.download_seconds, approve_retry=a.approve_retry_download, stage=a.stage), indent=2))


if __name__ == '__main__': main()

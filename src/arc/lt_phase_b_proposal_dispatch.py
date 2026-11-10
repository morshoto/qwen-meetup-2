"""Save a private, draft-only collection proposal without starting a run."""
import fcntl
import json
import os
import re
import zipfile
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_dispatch import _private_run_directory
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_registration_proposal import propose_collection_registration


def save_collection_proposal(*, source_root, run_dir, pool_path, challenge_path, official_archive,
                             python_executable, model_path):
    root, directory = _private_run_directory(source_root, run_dir)
    match = re.fullmatch(r'lt-phase-b-([a-zA-Z0-9_-]{1,128})', directory.name)
    if not match: raise ValueError('invalid proposed run directory')
    lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = directory / 'registration-proposal.json'
        if output.exists() or output.is_symlink(): raise ValueError('proposal already exists')
        def inputs():
            with zipfile.ZipFile(official_archive) as archive:
                info = archive.getinfo('arc-agi_evaluation_challenges.json')
                if info.file_size > 16 * 1024**2: raise ValueError('official challenge inventory too large')
                official = archive.read(info)
            return Path(pool_path).read_bytes(), Path(challenge_path).read_bytes(), official
        pool, challenge, official = inputs()
        try:
            ids = json.loads(pool)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError('invalid pool JSON') from exc
        result = propose_collection_registration(source_root=root, challenge_bytes=challenge,
            official_challenge_bytes=official, training_task_ids=ids, run_id=match.group(1),
            python_executable=python_executable, model_path=model_path)
        spec = result.get('spec') if isinstance(result, dict) else None
        if (not isinstance(spec, dict) or result.get('status') != 'collection_registration_proposed_unverified' or
                result.get('execution_ready') is not False or spec.get('status') != 'draft' or
                spec.get('run_id') != match.group(1) or result.get('spec_sha256') != _digest(spec) or
                not isinstance(result.get('export'), dict) or not isinstance(result['export'].get('tasks'), dict) or
                len(result['export']['tasks']) != 100):
            raise ValueError('invalid draft proposal')
        if inputs() != (pool, challenge, official): raise ValueError('proposal inputs changed')
        _create(output, result)
        return {'status': result['status'], 'execution_ready': False, 'spec_sha256': result['spec_sha256'],
                'task_count': 100, 'proposal_path': str(output)}
    finally:
        os.close(lockfd)

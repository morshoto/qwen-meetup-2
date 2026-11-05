"""Replay saved comparison programs; does not authenticate execution origin."""
import hashlib
import json
import os
import re
import stat

from src.arc.data import Task
from src.arc.rsi_comparison_freeze import _hash, _validate
from src.arc.rsi_evaluation import evaluate_task, score_task
from src.arc.rsi_slot_lifecycle import _directory, _exists
from src.arc.rsi_slot_audit import task_sha256


def _read(directory, name):
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError('comparison evidence must be owner-only')
            raw = stream.read(16 * 1024 * 1024 + 1)
    except OSError as exc:
        raise ValueError('comparison evidence unavailable') from exc
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError('comparison evidence too large')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate comparison evidence key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('invalid comparison evidence') from exc
    if not isinstance(value, dict):
        raise ValueError('comparison evidence must be an object')
    return raw, value


def replay_comparison(root, *, expected_sha256, tasks, sandbox, verifier):
    if not _hash(expected_sha256):
        raise ValueError('invalid comparison digest')
    with _directory(root) as directory:
        if _exists(directory, 'comparison.incomplete.json'):
            raise ValueError('incomplete comparison cannot be replayed as complete')
        _, started = _read(directory, 'comparison.started.json')
        raw, frozen = _read(directory, 'comparison.frozen.json')
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('frozen comparison digest differs')
        _validate(frozen)
        if (set(started) != {'registration_sha256', 'generation'} or
                started['registration_sha256'] != frozen['registration_sha256'] or
                type(started['generation']) is not int or started['generation'] not in (0, 1)):
            raise ValueError('comparison start differs from frozen record')
        ids = frozen['task_ids']
        if (not isinstance(tasks, dict) or set(tasks) != set(ids) or
                any(not isinstance(tasks[key], Task) or tasks[key].id != key or
                    task_sha256(tasks[key]) != frozen['task_hashes'][key] for key in ids)):
            raise ValueError('comparison tasks differ')
        seen_requests = set()
        for index, task_id in enumerate(ids):
            for arm in ('parent', 'candidate'):
                _, record = _read(directory, f'comparison-{index:03d}-{arm}.json')
                if set(record) != {'task_id', 'result', 'score'} or record['task_id'] != task_id:
                    raise ValueError('task record identity differs')
                result = record['result']
                if (not isinstance(result, dict) or not isinstance(result.get('sample_sources'), list) or
                        len(result['sample_sources']) != 2):
                    raise ValueError('sample sources unavailable')
                requests = result.get('sample_request_ids')
                if not isinstance(requests, list) or len(requests) != 2:
                    raise ValueError('sample request identifiers unavailable')
                for request_id in requests:
                    if request_id is not None:
                        if (not isinstance(request_id, str) or not re.fullmatch(r'[0-9a-f]{32}', request_id)
                                or request_id in seen_requests):
                            raise ValueError('invalid or duplicate sample request ID')
                        seen_requests.add(request_id)
                sources = result['sample_sources']
                def generate(_prompt, *, sample):
                    return sources[sample]
                replayed = evaluate_task(tasks[task_id], generate=generate, sandbox=sandbox)
                replayed['sample_request_ids'] = requests
                if replayed != result:
                    raise ValueError('task execution differs from replay')
                score = score_task(tasks[task_id], replayed, verifier)
                if score != record['score'] or score['solved'] != frozen[arm][task_id]['solved']:
                    raise ValueError('task score differs from frozen comparison')
    return {'status': 'comparison_replay_consistent', 'task_count': len(ids),
            'independently_verified': False, 'improvement_claim_supported': False,
            'evidence_type': 'caller_supplied_sandbox_and_verifier'}

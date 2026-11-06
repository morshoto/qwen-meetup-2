"""Freeze and summarize supplied matched outcomes, without authenticating a run."""
import hashlib
import json
import os
import re
import stat

from src.arc.rsi_paired import paired_summary
from src.arc.rsi_slot_lifecycle import _create, _directory

_HASH = re.compile(r'[0-9a-f]{64}')
_NAME = 'comparison.frozen.json'
_SETTINGS = {'model_id': 'mlx-community/Qwen3-14B-4bit',
             'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
             'temperature': 0.2, 'max_tokens': 512, 'thinking': 'off', 'samples_per_task': 2}


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


_BUDGET_LIMITS = {'model_calls': 800, 'reserved_tokens': 16000000, 'sandbox_runs': 8000}


def _counters(state):
    return {key: state[key] for key in _BUDGET_LIMITS}


def _budget_delta(before, after):
    for item in (before, after):
        if (not isinstance(item, dict) or set(item) != set(_BUDGET_LIMITS) or
                any(type(item[key]) is not int or not 0 <= item[key] <= limit
                    for key, limit in _BUDGET_LIMITS.items())):
            raise ValueError('invalid comparison budget counters')
    delta = {key: after[key] - before[key] for key in _BUDGET_LIMITS}
    if any(value < 0 for value in delta.values()):
        raise ValueError('comparison budget counters regressed')
    return delta


def _validate(record):
    if (not isinstance(record, dict) or set(record) != {'registration_sha256', 'settings', 'task_ids',
            'seeds', 'task_hashes', 'parent_adapter', 'candidate_adapter', 'parent', 'candidate',
            'budget_before', 'budget_after', 'evidence_type'} or
            not _hash(record['registration_sha256']) or record['settings'] != _SETTINGS or
            record['evidence_type'] != 'caller_supplied_outcomes' or
            not _hash(record['candidate_adapter']) or
            (record['parent_adapter'] != 'none' and not _hash(record['parent_adapter'])) or
            record['parent_adapter'] == record['candidate_adapter']):
        raise ValueError('invalid comparison identity or settings')
    usage = _budget_delta(record['budget_before'], record['budget_after'])
    if usage['model_calls'] != 800 or usage['reserved_tokens'] <= 0:
        raise ValueError('completed comparison budget does not cover 800 calls')
    ids, seeds = record['task_ids'], record['seeds']
    if (not isinstance(ids, list) or len(ids) != 200 or
            any(not isinstance(task, str) or not task for task in ids) or len(set(ids)) != 200 or
            not isinstance(seeds, dict) or set(seeds) != set(ids)):
        raise ValueError('comparison requires 200 distinct matched tasks')
    task_hashes = record['task_hashes']
    if not isinstance(task_hashes, dict) or set(task_hashes) != set(ids) or not all(_hash(v) for v in task_hashes.values()):
        raise ValueError('invalid comparison task hashes')
    all_seeds = []
    for task in ids:
        pair = seeds[task]
        if (not isinstance(pair, list) or len(pair) != 2 or
                any(type(seed) is not int or seed < 0 for seed in pair)):
            raise ValueError('invalid sample seeds')
        all_seeds.extend(pair)
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError('sample seeds must be distinct')
    for arm in ('parent', 'candidate'):
        values = record[arm]
        if not isinstance(values, dict) or set(values) != set(ids):
            raise ValueError('comparison arm is incomplete')
        for task in ids:
            value = values[task]
            if (not isinstance(value, dict) or set(value) != {'solved', 'seeds'} or
                    type(value['solved']) is not bool or value['seeds'] != seeds[task]):
                raise ValueError('comparison outcomes or seeds differ')


def comparison_coverage(task_ids, *, parent, candidate):
    if (not isinstance(task_ids, list) or len(task_ids) != 200 or
            any(not isinstance(task, str) or not task for task in task_ids) or len(set(task_ids)) != 200 or
            not isinstance(parent, dict) or not isinstance(candidate, dict) or
            not set(parent) <= set(task_ids) or not set(candidate) <= set(task_ids)):
        raise ValueError('invalid comparison task coverage')
    missing_parent = sorted(set(task_ids) - set(parent))
    missing_candidate = sorted(set(task_ids) - set(candidate))
    return {'status': 'incomplete' if missing_parent or missing_candidate else 'coverage_complete_unverified',
            'missing_parent': missing_parent, 'missing_candidate': missing_candidate,
            'improvement_claim_supported': False, 'independently_verified': False}


def freeze_comparison(*, root, registration_sha256, task_ids, task_hashes, seeds, parent_adapter,
                      candidate_adapter, parent, candidate, budget_before, budget_after):
    record = {'registration_sha256': registration_sha256, 'settings': dict(_SETTINGS),
              'task_ids': task_ids, 'task_hashes': task_hashes, 'seeds': seeds, 'parent_adapter': parent_adapter,
              'candidate_adapter': candidate_adapter, 'parent': parent, 'candidate': candidate,
              'budget_before': budget_before, 'budget_after': budget_after,
              'evidence_type': 'caller_supplied_outcomes'}
    _validate(record)
    raw = (json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    with _directory(root) as directory:
        _create(directory, _NAME, record)
    return {'status': 'frozen_unverified', 'sha256': digest, 'independently_verified': False}


def summarize_frozen_comparison(root, *, expected_sha256):
    if not _hash(expected_sha256):
        raise ValueError('invalid frozen digest')
    with _directory(root) as directory:
        try:
            fd = os.open(_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or
                        stat.S_IMODE(info.st_mode) != 0o600):
                    raise ValueError('comparison file must be owner-only')
                raw = stream.read(1024 * 1024 + 1)
        except OSError as exc:
            raise ValueError('frozen comparison unavailable') from exc
    if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('frozen comparison digest differs')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate comparison key')
            result[key] = value
        return result
    try:
        record = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('invalid frozen comparison') from exc
    _validate(record)
    summary = paired_summary({task: record['parent'][task]['solved'] for task in record['task_ids']},
                             {task: record['candidate'][task]['solved'] for task in record['task_ids']})
    criterion = summary.pop('improvement_criterion_met')
    return {**summary, 'budget_usage': _budget_delta(record['budget_before'], record['budget_after']),
            'descriptive_bootstrap_criterion_met': criterion,
            'improvement_claim_supported': False, 'independently_verified': False,
            'frozen_sha256': expected_sha256}

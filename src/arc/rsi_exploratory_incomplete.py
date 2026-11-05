"""Read completed rows of a stopped exploratory run without replay or authorization."""
import hashlib
import json
import os
import re

from src.arc.rsi_comparison_audit import _read
from src.arc.rsi_comparison_freeze import _SETTINGS
from src.arc.rsi_exploratory_report import summarize_exploratory
from src.arc.rsi_slot_audit import manifest_sha256
from src.arc.rsi_slot_lifecycle import _directory, _exists


def read_incomplete_exploratory(*, root, registration, registration_sha256, task_ids,
                                confirmatory_task_ids, candidate_adapter):
    try:
        raw = json.dumps(registration, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode()
        if (hashlib.sha256(raw).hexdigest() != registration_sha256 or
                registration.get('status') != 'exploratory_registration_draft_unverified' or
                registration.get('execution_ready') is not False or
                registration.get('exploratory_manifest_sha256') != manifest_sha256(task_ids) or
                registration.get('development_manifest_sha256') != manifest_sha256(confirmatory_task_ids) or
                not isinstance(candidate_adapter, str) or not re.fullmatch(r'[0-9a-f]{64}', candidate_adapter)):
            raise ValueError('partial exploratory registration differs')
        settings = dict(_SETTINGS)
        data = {'kind':'exploratory_50', 'model':_SETTINGS['model_id'], 'revision':_SETTINGS['model_revision'],
                'task_ids':task_ids, 'confirmatory_task_ids':confirmatory_task_ids,
                'arms':{name:{'adapter':adapter, 'settings':settings, 'results':[]}
                        for name,adapter in (('base','none'),('trained',candidate_adapter))}}
        with _directory(root) as directory:
            if _exists(directory, 'exploratory.completed.json') or _exists(directory, 'exploratory-results.json'):
                raise ValueError('partial view requires an incomplete run')
            _, started = _read(directory, 'exploratory.started.json')
            incomplete_raw, incomplete = _read(directory, 'exploratory.incomplete.json')
            counts = incomplete.get('completed')
            if (started.get('registration_sha256') != registration_sha256 or
                    incomplete.get('status') != 'exploratory_incomplete' or
                    incomplete.get('claim_eligible') is not False or incomplete.get('independently_verified') is not False or
                    not isinstance(counts, dict) or set(counts) != {'base','trained'} or
                    any(type(counts[k]) is not int or not 0 <= counts[k] <= 50 for k in counts) or
                    counts['base'] - counts['trained'] not in (0,1)):
                raise ValueError('partial exploratory marker differs')
            expected_names = {f'exploratory-{i:02d}-{arm}.json' for arm in counts for i in range(counts[arm])}
            present_names = {name for name in os.listdir(directory)
                             if re.fullmatch(r'exploratory-\d{2}-(base|trained)\.json', name)}
            if present_names != expected_names:
                raise ValueError('partial exploratory row coverage differs')
            recorded = {}
            for arm in ('base','trained'):
                for index in range(counts[arm]):
                    name = f'exploratory-{index:02d}-{arm}.json'
                    record_raw, record = _read(directory, name)
                    key = task_ids[index]
                    if (set(record) != {'task_id','result','score','request_ids','reported_usage','row'} or
                            record['task_id'] != key or not isinstance(record['row'], dict) or
                            record['row'].get('task_id') != key or not isinstance(record['score'], dict) or
                            record['row'].get('solved') is not record['score'].get('solved')):
                        raise ValueError('partial exploratory row differs')
                    recorded[name] = record_raw
                    data['arms'][arm]['results'].append(record['row'])
            summary = summarize_exploratory(data)
            row_coverage_complete = summary['complete']
            # All rows can exist even when finalization fails; stage completion is still false.
            summary['complete'] = False
            if _read(directory, 'exploratory.incomplete.json')[0] != incomplete_raw or any(
                    _read(directory, name)[0] != value for name, value in recorded.items()):
                raise ValueError('partial exploratory evidence changed')
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError('invalid partial exploratory inputs') from exc
    return {'status':'exploratory_partial_view_unverified', 'claim_eligible':False, 'independently_verified':False,
            'failure_type':incomplete.get('failure_type'), 'row_coverage_complete':row_coverage_complete,
            'data':data, 'summary':summary,
            'limitations':['completed rows and caller-supplied identities are not independently audited',
                           'the interrupted request may lack a completed row', 'missing measurements are unknown']}

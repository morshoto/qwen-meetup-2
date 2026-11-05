"""Replay generation capture consistency without authenticating provenance."""
import hashlib
from collections import Counter

from src.arc.rsi_collected_slot_audit import _index, audit_collected_slot
from src.arc.rsi_comparison_audit import _read
from src.arc.rsi_generation_stage import _pool_sha
from src.arc.rsi_slot_lifecycle import _directory, _exists, observe_slot
from src.arc.rsi_task_generation import generation_slots


def audit_generation(*, root, registration_sha256, generation, parent_adapter, task_ids, tasks,
                     expected_manifest_sha256, expected_task_hashes, public_probes, private_probes,
                     expected_public_probes_sha256, expected_private_probes_sha256, reference_pool,
                     expected_reference_pool_sha256, request_records, response_records, ledger_records, sandbox):
    slots = generation_slots(task_ids, generation)
    if not isinstance(reference_pool, list) or _pool_sha(reference_pool) != expected_reference_pool_sha256:
        raise ValueError('generation reference pool differs')
    requests, responses, ledger = map(_index, (request_records, response_records, ledger_records))
    pool = [dict(item) for item in reference_pool]
    expected_start = {'registration_sha256': registration_sha256, 'generation': generation,
        'parent_adapter': parent_adapter, 'manifest_sha256': expected_manifest_sha256,
        'task_hashes': expected_task_hashes, 'public_probe_sha256': expected_public_probes_sha256,
        'private_probe_sha256': expected_private_probes_sha256,
        'initial_reference_pool_sha256': expected_reference_pool_sha256}
    counts, checked, seen = Counter(), [], set()
    with _directory(root) as directory:
        if _exists(directory, f'g{generation}.stage.incomplete.json'):
            raise ValueError('incomplete generation cannot be replayed as complete')
        _, start = _read(directory, f'g{generation}.stage.started.json')
        _, report = _read(directory, f'g{generation}.stage.completed.json')
        if start != expected_start or not isinstance(report.get('slots'), list) or len(report['slots']) != 10:
            raise ValueError('generation stage identity or coverage differs')
        for slot in slots:
            index, key = slot['slot'], slot['task_id']
            raw, value = _read(directory, f'g{generation}-slot-{index:02d}.collection.json')
            digest = hashlib.sha256(raw).hexdigest()
            observed = observe_slot(root, registration_sha256=registration_sha256, generation=generation, slot=index)
            if observed.get('status') != 'completion_recorded_unverified' or observed.get('evidence_sha256') != digest:
                raise ValueError('slot completion binding differs')
            expected_entry = {'slot': index, 'task_id': key, 'status': value.get('status'), 'evidence_sha256': digest}
            if report['slots'][index] != expected_entry:
                raise ValueError('stage slot entry differs')
            audit_collected_slot(evidence_bytes=raw, evidence_sha256=digest, request_records=request_records,
                response_records=response_records, ledger_records=ledger_records, task_ids=task_ids, task=tasks[key],
                generation=generation, slot=index, expected_manifest_sha256=expected_manifest_sha256,
                expected_task_sha256=expected_task_hashes[key], public_probes=public_probes, private_probes=private_probes,
                expected_public_probes_sha256=expected_public_probes_sha256,
                expected_private_probes_sha256=expected_private_probes_sha256, reference_pool=pool, sandbox=sandbox)
            for attempt in value['attempts']:
                request_id = attempt['request_id']
                if request_id in seen or request_id not in requests or request_id not in responses or request_id not in ledger:
                    raise ValueError('generation request coverage differs')
                seen.add(request_id)
            if value['status'] == 'novel_unverified':
                novelty = value['novelty']
                pool.append({name: novelty[name] for name in
                             ('source_sha256', 'ast_sha256', 'behavior_sha256', 'probe_set_sha256')})
            counts[value['status']] += 1
            checked.append(expected_entry)
    expected_report = {'status': 'generation_complete_unverified', 'completed_slots': 10,
                       'slots': checked, 'counts': dict(counts), 'final_reference_pool_sha256': _pool_sha(pool),
                       'training_allowed': False, 'evidence_type': 'caller_supplied_components_unverified'}
    if report != expected_report or any(set(group) != seen for group in (requests, responses, ledger)):
        raise ValueError('generation stage report or capture coverage differs')
    return {'status': 'generation_replay_consistent', 'checked_slots': 10, 'counts': dict(counts),
            'training_allowed': False, 'evidence_type': 'caller_supplied_capture_and_sandbox'}

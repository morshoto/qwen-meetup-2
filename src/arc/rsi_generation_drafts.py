"""Derive a quarantined batch from replayed local generation evidence."""
import hashlib
import json

from src.arc.rsi_capture_draft import make_capture_artifacts
from src.arc.rsi_comparison_audit import _read
from src.arc.rsi_generation_audit import audit_generation
from src.arc.rsi_slot_lifecycle import _create, _directory, observe_slot
from src.arc.rsi_task_generation import generation_slots
from src.arc.rsi_synthetic_mapping import RepeatedSyntheticInput, map_synthetic_row


def prepare_generation_drafts(**inputs):
    audit_generation(**inputs)
    root, generation = inputs['root'], inputs['generation']
    pool = [dict(item) for item in inputs['reference_pool']]
    drafts, synthetic_examples, proposed_rows, mapping_rejections = [], [], [], []
    with _directory(root) as directory:
        _, stage = _read(directory, f'g{generation}.stage.completed.json')
        for item in generation_slots(inputs['task_ids'], generation):
            slot, task_id = item['slot'], item['task_id']
            raw, value = _read(directory, f'g{generation}-slot-{slot:02d}.collection.json')
            digest = hashlib.sha256(raw).hexdigest()
            observed = observe_slot(root, registration_sha256=inputs['registration_sha256'],
                                    generation=generation, slot=slot)
            if (observed.get('evidence_sha256') != digest or
                    stage['slots'][slot] != {'slot': slot, 'task_id': task_id,
                                             'status': value.get('status'), 'evidence_sha256': digest}):
                raise ValueError('generation evidence changed during draft preparation')
            if value['status'] == 'novel_unverified':
                draft, examples = make_capture_artifacts(parent_adapter=inputs['parent_adapter'], evidence_bytes=raw,
                    evidence_sha256=digest, request_records=inputs['request_records'],
                    response_records=inputs['response_records'], ledger_records=inputs['ledger_records'],
                    task_ids=inputs['task_ids'], task=inputs['tasks'][task_id], generation=generation, slot=slot,
                    expected_manifest_sha256=inputs['expected_manifest_sha256'],
                    expected_task_sha256=inputs['expected_task_hashes'][task_id],
                    public_probes=inputs['public_probes'], private_probes=inputs['private_probes'],
                    expected_public_probes_sha256=inputs['expected_public_probes_sha256'],
                    expected_private_probes_sha256=inputs['expected_private_probes_sha256'],
                    reference_pool=pool, sandbox=inputs['sandbox'])
                drafts.append(draft)
                synthetic_examples.append(examples)
                try:
                    proposed_rows.append(map_synthetic_row(task=inputs['tasks'][task_id], draft=draft,
                                                           generated=examples))
                except RepeatedSyntheticInput:
                    mapping_rejections.append({'task_id': task_id, 'request_id': draft['provenance']['request_id'],
                                               'reason': 'repeated_input'})
                novelty = value['novelty']
                pool.append({name: novelty[name] for name in
                             ('source_sha256', 'ast_sha256', 'behavior_sha256', 'probe_set_sha256')})
    audit_generation(**inputs)
    report = {'status': 'generation_drafts_quarantined', 'draft_count': len(drafts),
              'provisional_minimum_count_met': len(drafts) >= 5, 'training_allowed': False,
              'registration_sha256': inputs['registration_sha256'], 'generation': generation,
              'parent_adapter': inputs['parent_adapter'], 'drafts': drafts, 'synthetic_examples': synthetic_examples,
              'proposed_sft_rows': proposed_rows, 'mapping_rejections': mapping_rejections,
              'evidence_type': 'caller_supplied_capture_and_sandbox'}
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':')) + '\n').encode()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('draft batch exceeds size limit')
    with _directory(root) as directory:
        _create(directory, f'g{generation}.stage.drafts.json', report)
    return {'status': report['status'], 'draft_count': len(drafts),
            'provisional_minimum_count_met': report['provisional_minimum_count_met'],
            'training_allowed': False, 'sha256': hashlib.sha256(raw).hexdigest()}

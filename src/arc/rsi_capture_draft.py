"""Produce quarantined drafts from replayed, unauthenticated captures."""
from src.arc.rsi_candidate import inspect_candidate
from src.arc.rsi_collected_slot_audit import _index, audit_collected_slot
from src.arc.rsi_novelty import screen_novelty
from src.arc.rsi_sft_draft import make_sft_draft
from src.arc.rsi_synthetic_examples import retain_synthetic_examples
from src.arc.rsi_transport_audit import _json_object


def make_capture_draft(*, parent_adapter, model_key='14b', **evidence):
    draft, _ = make_capture_artifacts(parent_adapter=parent_adapter, model_key=model_key, **evidence)
    return draft


def make_capture_artifacts(*, parent_adapter, model_key='14b', **evidence):
    audit_collected_slot(**evidence, model_key=model_key)
    value = _json_object(evidence['evidence_bytes'])
    if value.get('status') != 'novel_unverified':
        raise ValueError('captured candidate is not provisionally novel')
    attempt = value['attempts'][-1]
    task = evidence['task']
    code = attempt['source']
    candidate = inspect_candidate(task, code, public_probes=evidence['public_probes'],
                                  private_probes=evidence['private_probes'], sandbox=evidence['sandbox'])
    if candidate.get('status') != 'verified_candidate':
        raise ValueError('candidate changed during draft construction')
    novelty = screen_novelty(code, candidate['private_behavior_sha256'],
                             candidate['private_probe_set_sha256'], evidence['reference_pool'])
    if novelty != value['novelty']:
        raise ValueError('novelty changed during draft construction')
    event = _index(evidence['ledger_records'])[attempt['request_id']]
    provenance = {'source_type': 'generator_distillation', 'model_id': event['model'],
                  'model_revision': event['revision'], 'request_id': event['request_id'],
                  'raw_response_sha256': event['response_sha256'], 'execution_mode': 'unverified_capture',
                  'slot_evidence_sha256': evidence['evidence_sha256'], 'slot': evidence['slot']}
    draft = make_sft_draft(task, code, candidate, novelty, provenance,
                           generation=evidence['generation'], parent_adapter=parent_adapter, model_key=model_key)
    examples = retain_synthetic_examples(task, code, candidate)
    examples.update({'request_id': event['request_id'], 'slot_evidence_sha256': evidence['evidence_sha256']})
    return draft, examples

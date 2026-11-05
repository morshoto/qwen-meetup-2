"""Freeze a local preflight proposal without starting or authorizing training."""
import hashlib
import json
from pathlib import Path

from src.arc.rsi_canary_dataset import audit_canary_loader
from src.arc.rsi_comparison_audit import _read
from src.arc.rsi_comparison_freeze import _hash
from src.arc.rsi_mlx_config import preview_mlx_configuration
from src.arc.rsi_sft_materialize import verify_quarantined_dataset
from src.arc.rsi_slot_lifecycle import _create, _directory
from src.arc.rsi_transport_audit import _json_object

_IDENTITIES = {'model_manifest', 'runtime_manifest', 'environment_manifest', 'source_inventory',
               'tokenizer', 'tokenizer_config'}


def _build_preflight_spec(*, root, preparation_bytes, expected_preparation_sha256, expected_canary_sha256,
                           expected_generation_dataset_sha256, synthetic_tasks, forbidden_tasks,
                           tokenizer, render, load_local_dataset, model_dir,
                           generation_data_dir, adapter_dir, previous_adapter_file, identity_hashes):
    if (not isinstance(preparation_bytes, bytes) or not 1 <= len(preparation_bytes) <= 8 * 1024 * 1024 or
            not _hash(expected_preparation_sha256) or
            hashlib.sha256(preparation_bytes).hexdigest() != expected_preparation_sha256 or
            not _hash(expected_canary_sha256) or not _hash(expected_generation_dataset_sha256) or
            not isinstance(identity_hashes, dict) or
            set(identity_hashes) != _IDENTITIES or not all(_hash(value) for value in identity_hashes.values())):
        raise ValueError('invalid preflight specification identity')
    plan = _json_object(preparation_bytes)
    stage_hash = plan.get('stage_sha256')
    if (not _hash(stage_hash) or plan.get('mapping') != 'generated_demos_holdout_first_visible_v1' or
            Path(generation_data_dir).expanduser().resolve() != (Path(root) / 'generation-sft.quarantine').resolve()):
        raise ValueError('synthetic generation preparation or path differs')

    def verify_generation():
        with _directory(root) as directory:
            saved, _ = _read(directory, 'synthetic.training.preparation.json')
            _, training_marker = _read(directory, 'synthetic.training.replay.json')
            _, dataset_marker = _read(directory, 'synthetic.dataset.replay.json')
        expected_training = {'status': 'synthetic_preparation_replay_consistent', 'training_allowed': False,
            'stage_sha256': stage_hash, 'preparation_sha256': expected_preparation_sha256,
            'evidence_type': 'caller_supplied_capture_and_sandbox'}
        expected_dataset = {'status': 'synthetic_dataset_replay_consistent', 'training_allowed': False,
            'stage_sha256': stage_hash, 'preparation_sha256': expected_preparation_sha256,
            'dataset_jsonl_sha256': expected_generation_dataset_sha256,
            'evidence_type': 'caller_supplied_capture_sandbox_and_tokenizer'}
        if saved != preparation_bytes or training_marker != expected_training or dataset_marker != expected_dataset:
            raise ValueError('synthetic generation replay markers differ')
        verify_quarantined_dataset(root=root, preparation_bytes=preparation_bytes,
            expected_preparation_sha256=expected_preparation_sha256,
            expected_dataset_sha256=expected_generation_dataset_sha256,
            tasks=synthetic_tasks, tokenizer=tokenizer, render=render)

    verify_generation()
    preview = preview_mlx_configuration(plan=plan, phase='preflight', model_dir=model_dir,
        generation_data_dir=generation_data_dir, canary_data_dir=Path(root) / 'preflight-canary.quarantine',
        adapter_dir=adapter_dir, previous_adapter_file=previous_adapter_file)
    if Path(adapter_dir).expanduser().exists():
        raise ValueError('preflight adapter output already exists')
    args = dict(root=root, forbidden_tasks=forbidden_tasks, tokenizer=tokenizer, render=render,
                expected_dataset_sha256=expected_canary_sha256, load_local_dataset=load_local_dataset)
    audit_canary_loader(**args)
    with _directory(Path(root) / 'preflight-canary.quarantine') as directory:
        audit_raw, audit = _read(directory, 'audit.json')
    if audit.get('dataset_jsonl_sha256') != expected_canary_sha256:
        raise ValueError('canary audit digest differs')
    record = {'status': 'preflight_proposed_unverified', 'execution_ready': False, 'training_allowed': False,
              'preparation_sha256': expected_preparation_sha256, 'canary_dataset_sha256': expected_canary_sha256,
              'generation_dataset_sha256': expected_generation_dataset_sha256, 'stage_sha256': stage_hash,
              'canary_audit_sha256': hashlib.sha256(audit_raw).hexdigest(), 'identity_hashes': dict(identity_hashes),
              'generation': plan['generation'], 'parent_adapter_claim': plan['parent_adapter'],
              'config': preview['config'], 'resource_limits': {'timeout_seconds': 300,
                  'min_disk_free_bytes': 20 * 1024 ** 3, 'min_memory_free_percent': 20,
                  'max_swap_growth_mib': 512, 'max_process_tree_rss_bytes': 32 * 1024 ** 3,
                  'max_sample_interval_seconds': 5},
              'evidence_type': 'caller_supplied_identities_inventory_loader_and_tokenizer'}
    raw = (json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode()
    audit_canary_loader(**args)
    verify_generation()
    return record, raw


def prepare_preflight_spec(**inputs):
    record, raw = _build_preflight_spec(**inputs)
    root = inputs['root']
    with _directory(root) as directory:
        _create(directory, 'preflight.proposal.json', record)
    return {'status': record['status'], 'execution_ready': False, 'training_allowed': False,
            'sha256': hashlib.sha256(raw).hexdigest()}


def verify_preflight_spec(*, root, expected_spec_sha256, synthetic_tasks, forbidden_tasks,
                          tokenizer, render, load_local_dataset):
    if not _hash(expected_spec_sha256):
        raise ValueError('invalid preflight specification digest')
    with _directory(root) as directory:
        spec_raw, spec = _read(directory, 'preflight.proposal.json')
        preparation, _ = _read(directory, 'synthetic.training.preparation.json')
    if hashlib.sha256(spec_raw).hexdigest() != expected_spec_sha256:
        raise ValueError('preflight specification digest differs')
    try:
        config = spec['config']
        expected, raw = _build_preflight_spec(root=root, preparation_bytes=preparation,
            expected_preparation_sha256=spec['preparation_sha256'], expected_canary_sha256=spec['canary_dataset_sha256'],
            expected_generation_dataset_sha256=spec['generation_dataset_sha256'], synthetic_tasks=synthetic_tasks,
            forbidden_tasks=forbidden_tasks, tokenizer=tokenizer, render=render, load_local_dataset=load_local_dataset,
            model_dir=config['model'], generation_data_dir=Path(root) / 'generation-sft.quarantine',
            adapter_dir=config['adapter_path'], previous_adapter_file=config['resume_adapter_file'],
            identity_hashes=spec['identity_hashes'])
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid preflight specification') from exc
    with _directory(root) as directory:
        after, _ = _read(directory, 'preflight.proposal.json')
    if spec_raw != after or spec_raw != raw or spec != expected:
        raise ValueError('preflight specification differs from replay')
    return {'status': 'preflight_spec_consistent_unverified', 'execution_ready': False,
            'training_allowed': False, 'proposal_sha256': expected_spec_sha256}

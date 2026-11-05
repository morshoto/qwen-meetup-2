"""Re-audit the two historical Qwen Oracles into an immutable E6-G seed pool."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.arc.e6_curriculum import diagnose_e5_artifact, write_private_jsonl
from src.arc.e6f_config import load_e6f_config
from src.arc.http_backend import BackendError
from src.arc.sandbox import DockerSandbox
from src.pipelines.e5 import _docker_image_id
from src.pipelines.e6 import _verify_seal
from src.pipelines.e6f import (PROJECT, _audit_record, _compat, _hash, _prior_records,
                              _probe_inputs, _split_evidence, _verify_final)

OUTPUT = PROJECT / 'artifacts/e6g/seed-pool'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    settings, base = load_e6f_config(PROJECT / 'configs/e6f.live.smoke.yml')
    source = PROJECT / settings.source_artifact
    _verify_seal(source)
    f_collection = PROJECT / 'artifacts/e6f/live-collection'
    _verify_final(f_collection)
    e5 = PROJECT / base.e6.e5_artifact
    diagnostic = diagnose_e5_artifact(e5)
    source_identity = json.loads((source / 'identity.json').read_text())
    image = _docker_image_id(_compat(base, settings))
    if image != source_identity['sandbox_image_id']:
        raise ValueError('historical and current Docker image IDs differ')
    splits, forbidden = _split_evidence(base)
    probes = _probe_inputs(settings)
    agent = base.agent
    docker = DockerSandbox(image=image, timeout=agent.timeout_seconds, memory_mb=agent.memory_mb,
        cpus=agent.cpus, pids=agent.pids, max_output_bytes=agent.max_output_bytes,
        max_code_bytes=agent.max_code_bytes)

    class Metered:
        starts = 0
        def run(self, code, grid):
            if self.starts >= 60:
                raise TimeoutError('offline seed audit sandbox budget exhausted')
            self.starts += 1
            return docker.run(code, grid)

    meter = Metered()
    candidates = _prior_records(settings)
    diagnosis = PROJECT / 'artifacts/e6g/diagnosis'
    expected = json.loads((diagnosis / 'seal.json').read_text())
    actual = {str(p.relative_to(diagnosis)): _sha(p) for p in sorted(diagnosis.rglob('*'))
              if p.is_file() and p.name != 'seal.json'}
    if expected != actual:
        raise ValueError('G1 diagnosis seal mismatch')
    with (diagnosis / 'duplicates.csv').open(newline='') as stream:
        historical_duplicates = {r['canonical_ast_hash'] for r in csv.DictReader(stream)
            if r['run'] == 'e6f_collection' and r['generation_stage'] == 'initial' and
               r['historical_final_status'] == 'duplicate_oracle' and r['matches_initial_pool_ast'] == 'True'}
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row, labels, origin in candidates:
        before = meter.starts
        try:
            behavior = _audit_record(row, labels, sandbox=meter, probes=probes,
                                     revision=base.model.revision, model_id=base.model.id,
                                     forbidden_ids=forbidden)
            if any(item['record']['oracle_ast_sha256'] == row['oracle_ast_sha256'] or
                   item['record']['oracle_shape_sha256'] == row['oracle_shape_sha256'] or
                   item['behavior_sha256'] == behavior for item in accepted):
                raise ValueError('duplicate_or_sampled_behavior_match')
            entry: dict[str, Any] = {'record': dict(row, _verifier_only={'inputs': labels['inputs'], 'outputs': labels['outputs']}),
                     'behavior_sha256': behavior, 'origin': origin,
                     'historical_duplicate_observed': row['oracle_ast_sha256'] in historical_duplicates,
                     'audit_status': 'historical_verified',
                     'legacy_identity_status': 'legacy_identity_incomplete'}
            accepted.append(entry)
            audit.append({'task_id': row['task_id'], 'status': 'historical_verified',
                          'legacy_identity_status': 'legacy_identity_incomplete',
                          'historical_duplicate_observed': entry['historical_duplicate_observed'],
                          'oracle_sha256': row['oracle_sha256'], 'oracle_ast_sha256': row['oracle_ast_sha256'],
                          'model_revision': row['base_revision'], 'sandbox_starts': meter.starts - before})
        except (BackendError, TimeoutError):
            raise
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            audit.append({'task_id': row.get('task_id'), 'status': 'legacy_unverifiable',
                          'error_type': type(exc).__name__, 'sandbox_starts': meter.starts - before})
    if len(candidates) != 2:
        raise ValueError('unexpected historical seed-record count')
    _verify_seal(source)
    _verify_final(f_collection)
    if diagnose_e5_artifact(e5)['source_root_seal_sha256'] != diagnostic['source_root_seal_sha256']:
        raise ValueError('historical E5 seal changed')
    OUTPUT.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(OUTPUT, 0o700)
    (OUTPUT / 'private').mkdir(mode=0o700)
    write_private_jsonl(OUTPUT / 'private/pool.jsonl', accepted)
    source_hash = _sha(source / 'artifact_hashes.json')
    implementation = ('src/arc/e6f_repair.py', 'src/arc/e6_curriculum.py', 'src/arc/training.py',
                      'src/arc/sandbox.py', 'src/arc/data.py', 'src/arc/e5_selfplay.py',
                      'src/arc/mlx_lora.py', 'src/arc/ast_fingerprint.py', 'src/pipelines/e6f.py',
                      'src/cli/e6g_seed.py')
    manifest = {'protocol': 'e6g_reaudited_seed_pool_v1', 'pool_sha256': _sha(OUTPUT / 'private/pool.jsonl'),
        'model_id': base.model.id, 'model_revision': base.model.revision, 'quantization': base.model.quantization,
        'sandbox_image_id': image, 'probe_inputs_sha256': _hash(probes), 'split_manifest_sha256': _hash(splits),
        'source_e5_seal_sha256': diagnostic['source_root_seal_sha256'], 'source_e6_seal_sha256': source_hash,
        'source_e6f_collection_seal_sha256': _sha(f_collection / 'seal.json'),
        'source_diagnosis_seal_sha256': _sha(diagnosis / 'seal.json'),
        'implementation_sha256': {name: _sha(PROJECT / name) for name in implementation},
        'source_status': audit, 'historical_verified': len(accepted), 'eligible_seed_oracles': len(accepted),
        'legacy_identity_incomplete': True, 'offline_model_calls': 0, 'offline_sandbox_starts': meter.starts,
        'safety_note': 'Current validation is pinned and reproducible; old static-filter identity remains unproven.'}
    (OUTPUT / 'pool_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    (OUTPUT / 'report.md').write_text('# E6-G re-audited pool\n\n' +
        f"Historical records independently re-audited: {len(accepted)}/{len(candidates)}. "
        f"The original source identity omitted the static-filter hash. "
        f"No model calls; Docker starts: {meter.starts}.\n")
    seal = {str(path.relative_to(OUTPUT)): _sha(path) for path in sorted(OUTPUT.rglob('*'))
            if path.is_file() and path.name != 'seal.json'}
    (OUTPUT / 'seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True) + '\n')
    for path in OUTPUT.rglob('*'):
        if path.is_file(): os.chmod(path, 0o600)
    print(json.dumps({'output': str(OUTPUT), 'audited': len(accepted), 'sandbox_starts': meter.starts,
                      'confirmed_historical_duplicate_seeds': sum(x['historical_duplicate_observed'] for x in accepted)}))


if __name__ == '__main__':
    main()

"""Construct a matched exploratory evaluation descriptor after training replay."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.arc.qwen38_local_runner import MODEL_ID, MODEL_REVISION, SEED, parse_prediction
from src.arc.qwen38_rsi_comparison import digest
from src.cli.qwen38_weight_prepare_sft import SFT_SOURCES

COMPARISON_SHA256 = 'cff354ce06e7e1604725f87f1567b34f0d1408e57e2b73c1d3889e2d9fe2f626'
EVALUATION_SOURCES = tuple(dict.fromkeys(SFT_SOURCES + (
    'src/arc/qwen38_weight_paired.py', 'src/arc/qwen38_weight_training_receipt.py',
    'src/arc/qwen38_weight_eval_schedule.py', 'src/arc/qwen38_weight_eval_rows.py',
    'src/arc/qwen38_weight_eval_registration.py', 'src/cli/qwen38_weight_eval_worker.py',
    'src/cli/qwen38_weight_evaluate.py',
)))


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _valid_digest(value):
    return isinstance(value, dict) and value.get('sha256') == digest({k: v for k, v in value.items() if k != 'sha256'})


def build_evaluation_descriptor(comparison, execution, training, evidence, challenges, baseline_files, repo, *, expected_count=50):
    """Build a descriptor; caller must independently replay training sources and evidence."""
    if (not _valid_digest(comparison) or comparison['sha256'] != COMPARISON_SHA256 or
        not _valid_digest(execution) or execution.get('comparison_sha256') != COMPARISON_SHA256 or
        not _valid_digest(training) or training.get('status') != 'trained' or training.get('trained') is not True or
        training.get('weights_updated') is not True or training.get('execution_sha256') != execution['sha256'] or
        evidence.get('status') != 'source_evidence_matches' or evidence.get('training_result_sha256') != training['sha256'] or
        evidence.get('adapter_sha256') != training.get('adapter_sha256')):
        raise ValueError('comparison or training identity mismatch')
    for key in ('adapter_sha256', 'adapter_config_sha256'):
        if not isinstance(training.get(key), str) or re.fullmatch('[0-9a-f]{64}', training[key]) is None:
            raise ValueError('invalid adapter identity')
    if (comparison.get('model_id') != MODEL_ID or comparison.get('model_revision') != MODEL_REVISION or
        execution.get('model', {}).get('id') != MODEL_ID or execution['model'].get('revision') != MODEL_REVISION or
        execution['model'].get('quantization') != '4-bit MLX' or comparison.get('seed') != SEED or
        comparison.get('candidate_count') != 2 or comparison.get('temperature') != .6 or
        comparison.get('max_tokens') != 4096 or comparison.get('max_context') != 18432 or
        comparison.get('format_recovery') is not True or _hash(challenges) != comparison.get('challenge_sha256') or
        execution['input_hashes'].get('challenges') != comparison['challenge_sha256']):
        raise ValueError('baseline generation settings differ')
    if comparison.get('model_files_sha256') != execution['model'].get('files_sha256'):
        raise ValueError('baseline model snapshot differs')
    repo = Path(repo).resolve()
    baseline_sources = comparison.get('source_files_sha256')
    if not isinstance(baseline_sources, dict) or not baseline_sources:
        raise ValueError('baseline source inventory missing')
    for name, expected in baseline_sources.items():
        path = (repo / name).resolve()
        if not path.is_relative_to(repo) or _hash(path.read_bytes()) != expected:
            raise ValueError('baseline source inventory changed')
    tasks = json.loads(challenges); ids = comparison['allocation']['evaluation']
    if len(ids) != expected_count or len(set(ids)) != expected_count or set(baseline_files) != set(ids) or not set(ids) <= tasks.keys():
        raise ValueError('baseline comparison task inventory mismatch')
    checkpoint_hashes = {}
    for task_id in ids:
        raw = baseline_files[task_id]; row = json.loads(raw)
        identity = digest({'manifest': COMPARISON_SHA256, 'label': 'evaluation/baseline', 'task_id': task_id,
                           'prompt_sha256': comparison['prompt_sha256'], 'variant_sha256': None})
        if (row.get('identity') != identity or row.get('task_id') != task_id or row.get('status') != 'complete' or
            row.get('manifest_sha256') != COMPARISON_SHA256 or row.get('prompt_sha256') != comparison['prompt_sha256'] or
            row.get('variant_sha256') is not None or len(row.get('candidates', [])) != 2):
            raise ValueError('baseline checkpoint identity mismatch')
        for index, candidate in enumerate(row['candidates']):
            if candidate.get('candidate_index') != index or not isinstance(candidate.get('raw'), str):
                raise ValueError('invalid baseline candidate')
            parsed, mode = parse_prediction(candidate['raw'], len(tasks[task_id]['test']))
            if parsed != candidate.get('outputs') or (parsed is not None) != candidate.get('valid') or mode != candidate.get('parse_mode'):
                raise ValueError('baseline candidate parse mismatch')
        checkpoint_hashes[task_id] = _hash(raw)
    repo = Path(repo)
    result = {'protocol': 'qwen38-weight-paired-evaluation-v1', 'status': 'evaluation_ready',
              'comparison_sha256': COMPARISON_SHA256, 'execution_sha256': execution['sha256'],
              'training_result_sha256': training['sha256'], 'adapter_sha256': training['adapter_sha256'],
              'adapter_config_sha256': training['adapter_config_sha256'], 'task_ids': ids,
              'baseline_checkpoint_sha256': checkpoint_hashes, 'prompt_sha256': comparison['prompt_sha256'],
              'challenge_sha256': comparison['challenge_sha256'], 'solution_sha256': execution['input_hashes']['solutions'],
              'seed_rule': f'{SEED} + candidate_index', 'candidate_count': 2, 'temperature': .6,
              'max_tokens': 4096, 'max_context': 18432, 'format_recovery': True,
              'reserved_completion_tokens': expected_count * 2 * 4096, 'wall_seconds_limit': 12 * 3600,
              'per_attempt_seconds': 600,
              'supervision': {'minimum_free_percent': 10, 'maximum_swap_bytes': 16 * 1024**3, 'max_output_bytes': 2 * 1024**2},
              'run_dir': str((Path(execution['run_dir']) / 'evaluation/adapter-v1').resolve()),
              'source_files_sha256': {name: _hash((repo / name).read_bytes()) for name in EVALUATION_SOURCES},
              'limitations': ['Previously evaluated 50 tasks; exploratory only.',
                              'Recorded candidates are not assumed independent.']}
    result['sha256'] = digest(result)
    return result

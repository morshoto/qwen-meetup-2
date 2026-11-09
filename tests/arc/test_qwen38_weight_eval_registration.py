import copy
import hashlib
import json

import pytest

from src.arc.qwen38_local_runner import MODEL_ID, MODEL_REVISION
from src.arc.qwen38_rsi_comparison import digest
import src.arc.qwen38_weight_eval_registration as evaluation


def fixture(tmp_path, monkeypatch):
    tasks = {k: {'train': [{'input': [[i]], 'output': [[1]]}], 'test': [{'input': [[2, i]]}]}
             for i, k in enumerate(('a', 'b'))}
    challenges = json.dumps(tasks, sort_keys=True).encode(); prompt = 'b' * 64
    comparison = {'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'prompt_sha256': prompt,
                  'challenge_sha256': hashlib.sha256(challenges).hexdigest(), 'seed': 20261111,
                  'candidate_count': 2, 'temperature': .6, 'max_tokens': 4096, 'max_context': 18432,
                  'format_recovery': True, 'allocation': {'evaluation': ['a', 'b']},
                  'model_files_sha256': {'model.safetensors': 'f' * 64},
                  'source_files_sha256': {'logic.py': hashlib.sha256(b'source').hexdigest()}}
    comparison['sha256'] = digest(comparison)
    execution = {'comparison_sha256': comparison['sha256'], 'run_dir': str((tmp_path / 'run').resolve()),
                 'input_hashes': {'challenges': comparison['challenge_sha256'], 'solutions': 'c' * 64},
                 'model': {'id': MODEL_ID, 'revision': MODEL_REVISION, 'quantization': '4-bit MLX',
                           'files_sha256': {'model.safetensors': 'f' * 64}}}
    execution['sha256'] = digest(execution)
    training = {'status': 'trained', 'trained': True, 'weights_updated': True, 'execution_sha256': execution['sha256'],
                'adapter_sha256': 'd' * 64, 'adapter_config_sha256': 'e' * 64}
    training['sha256'] = digest(training)
    evidence = {'status': 'source_evidence_matches', 'training_result_sha256': training['sha256'], 'adapter_sha256': 'd' * 64}
    rows = {}
    for task in ('a', 'b'):
        row = {'task_id': task, 'status': 'complete', 'manifest_sha256': comparison['sha256'], 'prompt_sha256': prompt,
               'variant_sha256': None, 'candidates': [{'candidate_index': i, 'raw': '[[[1]]]', 'outputs': [[[1]]],
                                                       'valid': True, 'parse_mode': 'strict'} for i in range(2)]}
        row['identity'] = digest({'manifest': comparison['sha256'], 'label': 'evaluation/baseline', 'task_id': task,
                                  'prompt_sha256': prompt, 'variant_sha256': None})
        rows[task] = json.dumps(row, sort_keys=True).encode()
    repo = tmp_path / 'repo'; repo.mkdir(); (repo / 'logic.py').write_text('source')
    monkeypatch.setattr(evaluation, 'COMPARISON_SHA256', comparison['sha256'])
    monkeypatch.setattr(evaluation, 'EVALUATION_SOURCES', ('logic.py',))
    return comparison, execution, training, evidence, challenges, rows, repo


def test_evaluation_descriptor_binds_baseline_adapter_and_matched_settings(tmp_path, monkeypatch):
    comparison, execution, training, evidence, challenges, rows, repo = fixture(tmp_path, monkeypatch)
    result = evaluation.build_evaluation_descriptor(comparison, execution, training, evidence, challenges, rows, repo, expected_count=2)
    assert result['status'] == 'evaluation_ready'
    assert result['task_ids'] == ['a', 'b'] and result['adapter_sha256'] == training['adapter_sha256']
    assert result['seed_rule'] == '20261111 + candidate_index'
    assert result['baseline_checkpoint_sha256']['a'] == hashlib.sha256(rows['a']).hexdigest()
    assert result['run_dir'] == str((tmp_path / 'run/evaluation/adapter-v1').resolve())
    assert '[[[1]]]' not in json.dumps(result)


@pytest.mark.parametrize('change', ['baseline', 'adapter', 'challenge', 'comparison', 'training'])
def test_evaluation_descriptor_rejects_changed_inputs(tmp_path, monkeypatch, change):
    comparison, execution, training, evidence, challenges, rows, repo = fixture(tmp_path, monkeypatch)
    if change == 'baseline':
        item = json.loads(rows['a']); item['candidates'][0]['outputs'] = [[[9]]]; rows['a'] = json.dumps(item).encode()
    elif change == 'adapter': evidence['adapter_sha256'] = 'f' * 64
    elif change == 'challenge': challenges += b' '
    elif change == 'comparison': comparison['allocation']['evaluation'][0] = 'b'
    else: training['weights_updated'] = False
    with pytest.raises(ValueError):
        evaluation.build_evaluation_descriptor(comparison, execution, training, evidence, challenges, rows, repo, expected_count=2)


def test_evaluation_source_inventory_includes_scheduler_and_worker():
    assert 'src/arc/qwen38_weight_eval_schedule.py' in evaluation.EVALUATION_SOURCES
    assert 'src/arc/qwen38_weight_eval_rows.py' in evaluation.EVALUATION_SOURCES
    assert 'src/cli/qwen38_weight_eval_worker.py' in evaluation.EVALUATION_SOURCES


def test_evaluation_rejects_baseline_source_drift(tmp_path, monkeypatch):
    comparison, execution, training, evidence, challenges, rows, repo = fixture(tmp_path, monkeypatch)
    (repo / 'logic.py').write_text('changed')
    with pytest.raises(ValueError):
        evaluation.build_evaluation_descriptor(comparison, execution, training, evidence, challenges, rows, repo, expected_count=2)


def test_evaluation_inventory_covers_transitive_project_imports():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]; inventory = set(evaluation.EVALUATION_SOURCES)
    for name in inventory:
        path = root / name
        assert path.is_file(), name
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('src.'):
                target = node.module.replace('.', '/') + '.py'
                if (root / target).is_file(): assert target in inventory, (name, target)
                for alias in node.names:
                    child = node.module.replace('.', '/') + '/' + alias.name + '.py'
                    if (root / child).is_file(): assert child in inventory, (name, child)

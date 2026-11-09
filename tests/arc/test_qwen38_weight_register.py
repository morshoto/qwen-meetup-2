import copy
import hashlib
import json

import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_register import audited_registration
from src.arc.qwen38_local_runner import MODEL_ID, MODEL_REVISION


def fixture():
    tasks = {f'{i:04}': {'train': [{'input': [[i]], 'output': [[i + 1]]}],
                         'test': [{'input': [[i + 10000]]}]} for i in range(350)}
    solutions = {k: [[[90000 + int(k)]]] for k in tasks}
    challenges_bytes = json.dumps(tasks, sort_keys=True).encode()
    solutions_bytes = json.dumps(solutions, sort_keys=True).encode()
    official = {f'e{i:03}': {'train': [{'input': [[i + 20000]], 'output': [[1]]}],
                             'test': [{'input': [[i + 30000]]}]} for i in range(120)}
    proposal = {'dataset_sha256': hashlib.sha256(challenges_bytes).hexdigest(),
                'task_ids': list(tasks)[:200], 'training_exclusions': list(tasks)[200:205],
                'protected_exclusions': list(tasks)[205:210], 'known_exposure_exclusions': list(tasks)[210:215],
                'development_exclusions': list(tasks)[215:218]}
    prompt = 'direct prompt'
    allocation = {'evaluation': list(tasks)[:50], 'development': list(tasks)[218:228],
                  'selection': list(tasks)[228:238]}
    allocation['sha256'] = digest(allocation)
    baseline = {'protocol': 'qwen38-local-exploratory-v1', 'allocation': allocation,
                'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
                'challenge_sha256': proposal['dataset_sha256'],
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'candidate_count': 2, 'temperature': .6, 'max_tokens': 4096, 'max_context': 18432,
                'format_recovery': True, 'model_files_sha256': {'model.safetensors': 'a' * 64}}
    baseline['sha256'] = digest(baseline)
    return dict(challenges_bytes=challenges_bytes, solutions_bytes=solutions_bytes,
                official=official, baseline=baseline, proposal=proposal,
                phase_b=proposal['training_exclusions'][:],
                prior={'confirmatory_task_ids': proposal['protected_exclusions'][:], 'exploratory_task_ids': []},
                exposure={'matched_task_ids': proposal['known_exposure_exclusions'][:]},
                lineage={}, prompt=prompt, source_hashes={'official_challenges': digest(official)},
                expected_comparison_sha256=baseline['sha256'])


def test_registration_binds_inputs_and_preserves_protected_tasks():
    args = fixture()
    result = audited_registration(**args)
    allocation = result['allocation']
    assert len(allocation['validation']) == 25
    assert not set(allocation['fresh'] + allocation['validation']) & set(args['baseline']['allocation']['evaluation'])
    assert result['comparison_sha256'] == args['baseline']['sha256']
    assert result['model']['files_sha256'] == args['baseline']['model_files_sha256']
    assert result['input_hashes']['solutions'] == hashlib.sha256(args['solutions_bytes']).hexdigest()
    assert set(result['solution_fingerprints']) == set(allocation['fresh'] + allocation['validation'])
    assert '90000' not in json.dumps(result)


@pytest.mark.parametrize('change', ['baseline', 'challenge', 'solutions', 'phase_b', 'prompt', 'official'])
def test_registration_rejects_untrusted_inputs(change):
    args = fixture()
    if change == 'baseline': args['baseline']['allocation']['evaluation'][0] = '0349'
    elif change == 'challenge': args['challenges_bytes'] += b' '
    elif change == 'solutions': args['solutions_bytes'] = b'{}'
    elif change == 'phase_b': args['phase_b'] = []
    elif change == 'prompt': args['prompt'] += '!'
    else: args['official'] = {}
    with pytest.raises(ValueError): audited_registration(**args)


def test_registration_checks_official_content_hash():
    args = fixture()
    args['source_hashes']['official_challenges'] = 'f' * 64
    with pytest.raises(ValueError): audited_registration(**args)


def test_expected_comparison_identity_cannot_be_replaced():
    args = fixture()
    args['expected_comparison_sha256'] = 'f' * 64
    with pytest.raises(ValueError): audited_registration(**args)


def test_load_sources_hashes_actual_files_and_public_evaluation(tmp_path):
    from zipfile import ZipFile
    from src.cli.qwen38_weight_register import load_sources
    args = fixture()
    paths = {
        'data/raw/arc-agi_training_challenges.json': args['challenges_bytes'],
        'data/raw/arc-agi_training_solutions.json': args['solutions_bytes'],
        'configs/lt.phase-a.comparison-proposal.json': json.dumps(args['proposal']).encode(),
        'configs/lt.phase-b.training-pool.json': json.dumps(args['phase_b']).encode(),
        'research/rsi/exploratory-split-proposal.json': json.dumps(args['prior']).encode(),
        'research/rsi/extended-exposure-review.json': json.dumps(args['exposure']).encode(),
        'lineage.json': b'{}', 'baseline.json': json.dumps(args['baseline']).encode(),
    }
    for name, content in paths.items():
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    with ZipFile(tmp_path / 'arc-prize-2025.zip', 'w') as archive:
        archive.writestr('arc-agi_evaluation_challenges.json', json.dumps(args['official']))
    result = load_sources(tmp_path, tmp_path, tmp_path / 'baseline.json', tmp_path / 'lineage.json')
    assert result['source_hashes']['official_challenges'] == digest(args['official'])
    assert result['source_hashes']['phase_a_proposal'] == hashlib.sha256(paths['configs/lt.phase-a.comparison-proposal.json']).hexdigest()
    assert result['baseline'] == args['baseline']
    assert result['lineage'] == {}

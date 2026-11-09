import copy
import hashlib
import json

import pytest

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_local_runner import MODEL_ID, MODEL_REVISION
import src.arc.qwen38_weight_preflight as preflight


def fixture(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'; repo.mkdir(); (repo / 'logic.py').write_text('source')
    model = tmp_path / MODEL_REVISION; model.mkdir(); (model / 'model.safetensors').write_bytes(b'base')
    base = {'model.safetensors': hashlib.sha256(b'base').hexdigest()}
    pins = {'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'comparison_sha256': 'c' * 64,
            'base_files_sha256': base, 'artifact_sha256': {'x': 'a' * 64}}
    pins_path = repo / 'pins.json'; pins_path.write_text(json.dumps(pins))
    challenge, solutions = b'challenges', b'solutions'
    proposal = {'status': 'proposal_not_authorization', 'comparison_sha256': 'c' * 64,
                'model': {'id': MODEL_ID, 'revision': MODEL_REVISION, 'quantization': '4-bit MLX', 'files_sha256': base},
                'allocation': {'fresh': ['a'], 'validation': ['v']}, 'generation': {'order': 'breadth_first'},
                'known_lineage': {}, 'input_hashes': {'challenges': hashlib.sha256(challenge).hexdigest(),
                                                      'solutions': hashlib.sha256(solutions).hexdigest()}}
    proposal['sha256'] = digest(proposal)
    review = {'status': 'reviewed_bounded', 'proposal_sha256': proposal['sha256'],
              'known_lineage_sha256': digest(proposal['known_lineage']), 'unresolved_exposure': True}
    review['sha256'] = digest(review)
    manifest = {'protocol': 'qwen38-weight-execution-v1', 'status': 'execution_ready',
                'proposal_sha256': proposal['sha256'], 'comparison_sha256': proposal['comparison_sha256'],
                'allocation': proposal['allocation'], 'generation': proposal['generation'], 'model': proposal['model'],
                'input_hashes': proposal['input_hashes'], 'lineage_review_sha256': review['sha256'],
                'source_files_sha256': {'logic.py': hashlib.sha256(b'source').hexdigest()},
                'canary_pins_sha256': hashlib.sha256(pins_path.read_bytes()).hexdigest(),
                'packages': preflight.PACKAGES, 'supervision': preflight.SUPERVISION}
    manifest['sha256'] = digest(manifest)
    monkeypatch.setattr(preflight, 'EXECUTION_SOURCES', ('logic.py',))
    monkeypatch.setattr(preflight, 'PINS_PATH', 'pins.json')
    monkeypatch.setattr(preflight, 'PROPOSAL_SHA256', proposal['sha256'])
    monkeypatch.setattr(preflight.metadata, 'version', lambda name: preflight.PACKAGES[name])
    monkeypatch.setattr(preflight, 'verify_canary_bundle', lambda *args: {'status': 'passed'})
    return manifest, proposal, review, repo, model, challenge, solutions


def test_preflight_binds_source_model_data_canary_and_review(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    assert preflight.verify_preflight(*args, canary_dir=tmp_path / 'canary')['status'] == 'passed'


@pytest.mark.parametrize('change', ['proposal', 'source', 'model', 'dataset', 'review', 'canary'])
def test_preflight_rejects_identity_drift(tmp_path, monkeypatch, change):
    manifest, proposal, review, repo, model, challenge, solutions = fixture(tmp_path, monkeypatch)
    if change == 'proposal': proposal['allocation']['fresh'] = ['other']
    elif change == 'source': (repo / 'logic.py').write_text('changed')
    elif change == 'model': (model / 'model.safetensors').write_bytes(b'changed')
    elif change == 'dataset': solutions += b' '
    elif change == 'review': review['unresolved_exposure'] = False
    else: monkeypatch.setattr(preflight, 'verify_canary_bundle', lambda *args: {'status': 'failed'})
    with pytest.raises(ValueError):
        preflight.verify_preflight(manifest, proposal, review, repo, model, challenge, solutions, canary_dir=tmp_path / 'canary')

import hashlib
import json

from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_preflight import PACKAGES
import src.cli.qwen38_weight_prepare_sft as prepare


def fixture(tmp_path, monkeypatch):
    tasks = {}; answers = {}; fresh = []; history = []
    for i in range(10):
        task = str(i); tasks[task] = {'train': [{'input': [[0, i]], 'output': [[1]]}], 'test': [{'input': [[2, i]]}]}
        answers[task] = [[[3, i, 4]]]; raw = json.dumps(answers[task], separators=(',', ':'))
        record = {'task_id': task, 'raw': raw, 'raw_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                  'source': 'current_gen0_self_generated' if i < 6 else 'historical_fp8', 'candidate_index': 0,
                  'checkpoint_sha256': 'a' * 64, 'manifest_sha256': 'b' * 64}
        (fresh if i < 6 else history).append(record)
    challenges = json.dumps(tasks, sort_keys=True).encode(); solutions = json.dumps(answers, sort_keys=True).encode()
    official = {'official': {'train': [{'input': [[9, 9]], 'output': [[1]]}], 'test': [{'input': [[9, 8]]}]}}
    official_bytes = json.dumps(official, sort_keys=True).encode()
    proposal = {'allocation': {'fresh': list(tasks)[:6], 'validation': ['9']},
                'exclusions': {'prior_phase_a': ['6', '7', '8'], 'protected': ['9']}, 'known_lineage': {},
                'input_hashes': {'challenges': hashlib.sha256(challenges).hexdigest(),
                                 'solutions': hashlib.sha256(solutions).hexdigest(),
                                 'official_challenges': digest(official), 'official_challenges_file': hashlib.sha256(official_bytes).hexdigest()}}
    settings = {'rank': 8, 'batch_size': 1, 'learning_rate': 2e-5, 'max_steps': 20, 'seed': 20261111,
                'max_sequence_length': 2048, 'gradient_checkpointing': True, 'completion_only_loss': True}
    manifest = {'sha256': 'b' * 64, 'input_hashes': proposal['input_hashes'], 'model': {'files_sha256': {'model': 'c' * 64}},
                'packages': PACKAGES.copy(), 'training_settings': settings}
    review = {'history_inspection': {'checkpoint_inventory_sha256': 'd' * 64}}
    repo = tmp_path / 'repo'; repo.mkdir(); (repo / 'logic.py').write_text('source')
    monkeypatch.setattr(prepare, 'SFT_SOURCES', ('logic.py',))
    monkeypatch.setattr(prepare, 'replay_fresh', lambda *a, **k: {'records': fresh, 'journal_sha256': 'e' * 64,
                                                                'verification_sha256': 'f' * 64})
    return repo, manifest, proposal, review, challenges, solutions, official_bytes, fresh, history


def test_sft_preparation_prefers_fresh_without_opening_history(tmp_path, monkeypatch):
    repo, manifest, proposal, review, challenges, solutions, official, fresh, history = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(prepare, 'load_history', lambda *a: (_ for _ in ()).throw(AssertionError('history opened')))
    selection, registration = prepare.prepare_selection(manifest, proposal, review, repo, tmp_path, challenges, solutions,
                                                         official, tmp_path, tmp_path, tmp_path / 'missing', token_counter=lambda _: 1)
    assert selection['mode'] == 'fresh_only' and selection['counts']['unique_parents'] == 6
    assert registration['historical_aggregate_sha256'] is None
    assert registration['training_code_sha256'] == digest(registration['source_files_sha256'])


def test_sft_preparation_uses_history_only_after_fresh_gate_fails(tmp_path, monkeypatch):
    repo, manifest, proposal, review, challenges, solutions, official, fresh, history = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(prepare, 'replay_fresh', lambda *a, **k: {'records': fresh[:3], 'journal_sha256': 'e' * 64,
                                                                'verification_sha256': 'f' * 64})
    monkeypatch.setattr(prepare, 'load_history', lambda *a: {'history_manifest': {}, 'aggregate_bytes': b'', 'checkpoint_bytes': {}})
    monkeypatch.setattr(prepare, 'replay_history', lambda **kwargs: {'records': history[:3]})
    selection, registration = prepare.prepare_selection(manifest, proposal, review, repo, tmp_path, challenges, solutions,
                                                         official, tmp_path, tmp_path, tmp_path / 'history', token_counter=lambda _: 1)
    assert selection['mode'] == 'mixed_source' and selection['counts']['unique_parents'] == 6
    assert registration['historical_aggregate_sha256'] == prepare.HISTORY_AGGREGATE_SHA256


def test_sft_source_inventory_covers_project_imports():
    import ast
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    sources = set(prepare.SFT_SOURCES)
    for name in sources:
        path = repo / name
        if not path.exists():
            continue  # The training entrypoint is under construction.
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('src.'):
                target = node.module.replace('.', '/') + '.py'
                if (repo / target).is_file(): assert target in sources, (name, target)
                for alias in node.names:
                    child = node.module.replace('.', '/') + '/' + alias.name + '.py'
                    if (repo / child).is_file(): assert child in sources, (name, child)


def test_sft_inventory_includes_training_worker_and_its_instrumentation():
    assert 'src/cli/qwen38_weight_train_worker.py' in prepare.SFT_SOURCES
    assert 'src/cli/qwen38_canary_worker.py' in prepare.SFT_SOURCES
    assert 'src/arc/qwen38_weight_train.py' in prepare.SFT_SOURCES
    assert 'src/cli/qwen38_weight_probe.py' in prepare.SFT_SOURCES


def test_prepare_and_replay_bundle_use_fixed_private_path(tmp_path, monkeypatch):
    from src.arc.qwen38_weight_sft import select_training_examples
    repo, manifest, proposal, review, challenges, solutions, official, fresh, history = fixture(tmp_path, monkeypatch)
    tasks, answers = json.loads(challenges), json.loads(solutions)
    selection = select_training_examples(tasks, answers, fresh, [], blocked_ids={'9'},
                                          forbidden_answer_fingerprints=set(), token_counter=lambda _: 1)
    files = {'logic.py': hashlib.sha256(b'source').hexdigest()}
    registration = {'execution_sha256': manifest['sha256'], 'fresh_journal_sha256': 'e' * 64,
                    'fresh_verification_sha256': 'f' * 64, 'challenge_sha256': proposal['input_hashes']['challenges'],
                    'solution_sha256': proposal['input_hashes']['solutions'], 'source_files_sha256': files,
                    'training_code_sha256': digest(files), 'historical_aggregate_sha256': None,
                    'historical_checkpoint_inventory_sha256': None, 'packages': PACKAGES,
                    'model_files_sha256': manifest['model']['files_sha256'], 'training_settings': manifest['training_settings']}
    monkeypatch.setattr(prepare, 'prepare_selection', lambda *a, **k: (selection, registration))
    root = tmp_path / 'run'; root.mkdir(mode=0o700)
    manifest['run_dir'] = str(root.resolve())
    kwargs = dict(manifest=manifest, proposal=proposal, review=review, repo=repo, model_path=tmp_path,
                  challenges=challenges, solutions=solutions, official_bytes=official, canary_dir=tmp_path,
                  run_dir=root, history_dir=tmp_path, token_counter=lambda _: 1)
    result = prepare.prepare_bundle(**kwargs)
    assert result['status'] == 'prepared_pending_source_replay'
    assert (root / 'training/sft-v1/train.jsonl').is_file()
    assert prepare.replay_bundle(**kwargs)[2]['status'] == 'matches_supplied_replay'
    (root / 'training/sft-v1/train.jsonl').write_text('{}\n')
    import pytest
    with pytest.raises(ValueError): prepare.replay_bundle(**kwargs)


def test_prepare_cli_reads_only_public_evaluation_challenges(tmp_path, monkeypatch, capsys):
    from zipfile import ZipFile
    repo, manifest, proposal, review, challenges, solutions, official, fresh, history = fixture(tmp_path, monkeypatch)
    for name, payload in [('manifest.json', json.dumps(manifest).encode()), ('proposal.json', json.dumps(proposal).encode()),
                          ('review.json', json.dumps(review).encode()), ('challenges.json', challenges), ('solutions.json', solutions)]:
        (tmp_path / name).write_bytes(payload)
    with ZipFile(tmp_path / 'archive.zip', 'w') as archive:
        archive.writestr('arc-agi_evaluation_challenges.json', official)
    calls = []
    def fake_bundle(**kwargs):
        calls.append(kwargs)
        return {'status': 'insufficient_verified_data', 'mode': 'none', 'counts': {'unique_parents': 0}, 'sha256': 'a' * 64}
    monkeypatch.setattr(prepare, 'prepare_bundle', fake_bundle)
    args = ['--manifest', str(tmp_path / 'manifest.json'), '--proposal', str(tmp_path / 'proposal.json'),
            '--review', str(tmp_path / 'review.json'), '--repo', str(repo), '--model-path', str(tmp_path),
            '--challenges', str(tmp_path / 'challenges.json'), '--solutions', str(tmp_path / 'solutions.json'),
            '--archive', str(tmp_path / 'archive.zip'), '--canary-dir', str(tmp_path), '--run-dir', str(tmp_path / 'run'),
            '--history-dir', str(tmp_path / 'history')]
    assert prepare.main(args) == 0
    assert calls[0]['official_bytes'] == official
    output = capsys.readouterr().out
    assert 'insufficient_verified_data' in output and '[[[3' not in output

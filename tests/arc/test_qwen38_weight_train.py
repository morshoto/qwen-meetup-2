import pytest

from src.arc.qwen38_weight_train import parse_training_losses


def test_loss_log_requires_every_finite_training_step():
    log = '\n'.join(f'Iter {i}: Train loss \x1b[92m{i / 10:.2f}\x1b[0m, Learning Rate' for i in range(1, 21))
    losses = parse_training_losses(log, steps=20)
    assert len(losses) == 20 and losses[0] == .1 and losses[-1] == 2.0


@pytest.mark.parametrize('log', ['Iter 1: Train loss nan, Learning Rate',
                                 'Iter 1: Train loss 1,\nIter 1: Train loss 2,',
                                 'Iter 2: Train loss 1,', ''])
def test_loss_log_rejects_missing_duplicate_nonfinite_and_wrong_steps(log):
    with pytest.raises(ValueError): parse_training_losses(log, steps=1)


def test_private_trainer_dataset_contains_only_replayed_data(tmp_path):
    import hashlib
    from src.cli.qwen38_weight_train import materialize_trainer_dataset
    training = tmp_path / 'training'; training.mkdir(mode=0o700)
    bundle = training / 'sft-v1'; bundle.mkdir(mode=0o700)
    payload = b'{"messages":[]}\n'; (bundle / 'train.jsonl').write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    dataset = materialize_trainer_dataset(bundle, training, sha)
    assert {p.name for p in dataset.iterdir()} == {'train.jsonl'}
    assert (dataset / 'train.jsonl').read_bytes() == payload
    assert (dataset / 'train.jsonl').stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError): materialize_trainer_dataset(bundle, training, sha)


def test_trainer_dataset_refuses_changed_source(tmp_path):
    from src.cli.qwen38_weight_train import materialize_trainer_dataset
    training = tmp_path / 'training'; training.mkdir(mode=0o700)
    bundle = training / 'sft-v1'; bundle.mkdir(mode=0o700); (bundle / 'train.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError): materialize_trainer_dataset(bundle, training, 'a' * 64)
    assert not (training / 'trainer-data-v1').exists()


def test_insufficient_gate_writes_skip_without_starting_training(tmp_path, monkeypatch):
    import json
    import src.cli.qwen38_weight_train as train
    root = tmp_path / 'run'; root.mkdir(mode=0o700); (root / 'training').mkdir(mode=0o700)
    manifest = {'sha256': 'a' * 64, 'run_dir': str(root.resolve())}
    selection = {'status': 'insufficient_verified_data', 'mode': 'none', 'counts': {'unique_parents': 0}}
    monkeypatch.setattr(train, 'replay_bundle', lambda **kwargs: (selection, {}, {'status': 'matches_supplied_replay', 'manifest_sha256': 'b' * 64}))
    monkeypatch.setattr(train, 'run_supervised', lambda *a, **k: (_ for _ in ()).throw(AssertionError('trainer launched')))
    result = train.execute_training(manifest=manifest, run_dir=root)
    assert result['status'] == 'insufficient_verified_data' and 'score' not in result
    assert not (root / 'training/trainer-data-v1').exists()
    assert json.loads((root / 'training/training-decision-v1.json').read_text()) == result


def _training_fixture(tmp_path, monkeypatch, *, bad_loss=False):
    import hashlib
    import json
    from pathlib import Path
    import src.cli.qwen38_weight_train as train
    from src.arc.qwen38_rsi_comparison import digest
    root = tmp_path / 'run'; root.mkdir(mode=0o700); training = root / 'training'; training.mkdir(mode=0o700)
    bundle = training / 'sft-v1'; bundle.mkdir(mode=0o700); data = b'{"messages":[]}\n'; (bundle / 'train.jsonl').write_bytes(data)
    bundle_manifest = {'status': 'prepared_pending_source_replay', 'train_sha256': hashlib.sha256(data).hexdigest()}
    bundle_manifest['sha256'] = digest(bundle_manifest); (bundle / 'manifest.json').write_text(json.dumps(bundle_manifest))
    manifest = {'sha256': 'a' * 64, 'run_dir': str(root.resolve())}
    selection = {'status': 'eligible_pending_source_replay', 'mode': 'fresh_only',
                 'counts': {'unique_parents': 5, 'current_gen0_self_generated': 5, 'historical_fp8': 0}}
    registration = {'execution_sha256': manifest['sha256'], 'model_files_sha256': {'model.safetensors': 'c' * 64},
                    'training_settings': {'rank': 8, 'batch_size': 1, 'learning_rate': 2e-5, 'max_steps': 20,
                                          'seed': 20261111, 'max_sequence_length': 2048,
                                          'gradient_checkpointing': True, 'completion_only_loss': True}}
    monkeypatch.setattr(train, 'replay_bundle', lambda **kwargs: (selection, registration,
                        {'status': 'matches_supplied_replay', 'manifest_sha256': bundle_manifest['sha256']}))
    model = tmp_path / 'model'; model.mkdir(); repo = tmp_path / 'repo'; repo.mkdir()
    calls = []
    def supervisor(command, request, directory, **kwargs):
        directory = Path(directory); directory.mkdir(mode=0o700); calls.append(command)
        for name in ('request.json', 'stdout.log', 'stderr.log'): (directory / name).write_text('')
        execution = {'status': 'complete', 'reason': None, 'exit_code': 0, 'process_group_gone': True,
                     'elapsed_seconds': 1, 'command': command, 'pid': 123,
                     'samples': [{'free_percent': 80, 'swap_bytes': 0}]}
        (directory / 'execution.json').write_text(json.dumps(execution))
        if 'src.cli.qwen38_weight_train_worker' in command:
            losses = '\n'.join(f'Iter {i}: Train loss {"nan" if bad_loss and i == 20 else "0.5"}, Learning Rate' for i in range(1, 21))
            (directory / 'stdout.log').write_text(losses)
            (directory / 'initial.safetensors').write_bytes(b'initial')
            (directory / 'adapter').mkdir(mode=0o700)
            (directory / 'adapter/adapters.safetensors').write_bytes(b'adapter')
            (directory / 'adapter/adapter_config.json').write_text('{}')
        else:
            parent = directory.parent
            probe = {'adapter_sha256': hashlib.sha256(b'adapter').hexdigest(), 'nonzero_lora_tensors': 1,
                     'max_abs_logit_delta': .25, 'changed_lora_tensors': 1}
            (parent / 'tensor-probe.json').write_text(json.dumps(probe))
            (parent / 'adapter/probe.json').write_text(json.dumps({k: v for k, v in probe.items() if k != 'changed_lora_tensors'}))
        return execution
    monkeypatch.setattr(train, 'run_supervised', supervisor)
    return train, dict(manifest=manifest, run_dir=root, model_path=model, repo=repo), calls


def test_training_requires_losses_changed_tensors_and_reload_probe(tmp_path, monkeypatch):
    import json
    train, kwargs, calls = _training_fixture(tmp_path, monkeypatch)
    result = train.execute_training(**kwargs)
    assert result['status'] == 'trained' and len(result['losses']) == 20
    assert result['changed_lora_tensors'] == 1 and result['max_abs_logit_delta'] == .25
    assert len(calls) == 2
    assert json.loads((kwargs['run_dir'] / 'training/training-result-v1.json').read_text()) == result


def test_nonfinite_training_loss_never_writes_success_receipt(tmp_path, monkeypatch):
    train, kwargs, calls = _training_fixture(tmp_path, monkeypatch, bad_loss=True)
    with pytest.raises(ValueError): train.execute_training(**kwargs)
    assert not (kwargs['run_dir'] / 'training/training-result-v1.json').exists()
    assert len(calls) == 1


def test_training_failure_is_recorded_without_a_score_or_retry(tmp_path, monkeypatch):
    import json
    train, kwargs, calls = _training_fixture(tmp_path, monkeypatch, bad_loss=True)
    with pytest.raises(ValueError): train.execute_training(**kwargs)
    failure = json.loads((kwargs['run_dir'] / 'training/training-failed-v1.json').read_text())
    assert failure['status'] == 'training_failed' and 'score' not in failure
    with pytest.raises(FileExistsError): train.execute_training(**kwargs)
    assert len(calls) == 1

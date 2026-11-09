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

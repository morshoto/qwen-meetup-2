import hashlib
from pathlib import Path

import pytest

from src.arc.qwen38_local_runner import MODEL_REVISION
from src.cli.qwen38_weight_train_worker import run_training


def fixture(tmp_path):
    model = tmp_path / MODEL_REVISION; model.mkdir()
    dataset = tmp_path / 'dataset'; dataset.mkdir(); (dataset / 'train.jsonl').write_text('{}\n')
    root = tmp_path / 'training'; root.mkdir(mode=0o700)
    return model, dataset, root, hashlib.sha256((dataset / 'train.jsonl').read_bytes()).hexdigest()


def test_training_worker_pins_arguments_and_initial_tensor_capture(tmp_path):
    model, dataset, root, sha = fixture(tmp_path); seen = []
    run_training(model, dataset, root, sha, runner=lambda argv, initial: seen.append((argv, initial)))
    argv, initial = seen[0]
    assert argv[argv.index('--iters') + 1] == '20'
    assert argv[argv.index('--lora-rank') + 1] == '8'
    assert argv[argv.index('--max-seq-length') + 1] == '2048'
    assert '--grad-checkpoint' in argv and '--train-on-completions' in argv
    assert initial == root / 'initial.safetensors'
    assert argv[argv.index('--output-path') + 1] == str(root / 'adapter')


def test_training_worker_refuses_changed_data_or_existing_outputs(tmp_path):
    model, dataset, root, sha = fixture(tmp_path); seen = []
    (dataset / 'train.jsonl').write_text('changed')
    with pytest.raises(ValueError): run_training(model, dataset, root, sha, runner=lambda *args: seen.append(args))
    assert not seen
    (dataset / 'train.jsonl').write_text('{}\n'); (root / 'adapter').mkdir()
    with pytest.raises(ValueError): run_training(model, dataset, root, sha, runner=lambda *args: seen.append(args))
    assert not seen

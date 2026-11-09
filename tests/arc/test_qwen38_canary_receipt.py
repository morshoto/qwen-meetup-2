import hashlib
import json

import pytest

from src.arc.qwen38_canary_receipt import verify_canary_bundle
from src.cli.qwen38_canary import MODEL_REVISION, PACKAGES


def fixture(tmp_path):
    root = tmp_path / 'canary'; root.mkdir(); (root / 'adapter').mkdir(); (root / 'data').mkdir()
    model = tmp_path / MODEL_REVISION; model.mkdir(); (model / 'model.safetensors').write_bytes(b'base')
    base = {'model.safetensors': hashlib.sha256(b'base').hexdigest()}
    adapter = b'adapter'; (root / 'adapter/adapters.safetensors').write_bytes(adapter)
    (root / 'adapter/adapter_config.json').write_bytes(b'config')
    (root / 'initial.safetensors').write_bytes(b'initial')
    command = ['python', '-m', 'src.cli.qwen38_canary_worker', '--model-path', str(model),
               '--iters', '1', '--batch-size', '1', '--lora-rank', '8', '--grad-checkpoint',
               '--train-on-completions']
    reg = {'model_revision': MODEL_REVISION, 'packages': PACKAGES, 'base_before': base, 'command': command,
           'timeout_seconds': 900, 'minimum_free_percent': 10, 'maximum_swap_bytes': 16 * 1024**3}
    run = {'status': 'execution_only_unverified', 'exit_code': 0, 'reason': None, 'process_group_gone': True,
           'base_after': base, 'samples': [{'free_percent': 50, 'swap_bytes': 0}]}
    probe_run = {'exit_code': 0, 'reason': None, 'process_group_gone': True,
                 'samples': [{'free_percent': 50, 'swap_bytes': 0}]}
    probe = {'adapter_sha256': hashlib.sha256(adapter).hexdigest(), 'changed_lora_tensors': 1, 'max_abs_logit_delta': .1}
    for name, value in [('registration.json', reg), ('execution.json', run), ('probe-execution.json', probe_run),
                        ('tensor-probe.json', probe)]: (root / name).write_text(json.dumps(value))
    (root / 'train.log').write_text('Iter 1: Train loss 0.25, Learning Rate')
    (root / 'probe.log').write_text(json.dumps(probe))
    (root / 'data/train.jsonl').write_text(json.dumps({'messages': [{'role': 'user', 'content': 'Return one.'}, {'role': 'assistant', 'content': '1'}]}) + '\n')
    names = ['registration.json', 'execution.json', 'probe-execution.json', 'tensor-probe.json', 'train.log',
             'probe.log', 'initial.safetensors', 'adapter/adapters.safetensors', 'adapter/adapter_config.json', 'data/train.jsonl']
    pins = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    return root, model, pins, base


def test_verified_canary_bundle_binds_loss_tensors_logits_and_base(tmp_path):
    root, model, pins, base = fixture(tmp_path)
    receipt = verify_canary_bundle(root, model, pins, base)
    assert receipt['status'] == 'passed'
    assert receipt['loss'] == .25 and receipt['changed_lora_tensors'] == 1
    assert receipt['max_abs_logit_delta'] == .1


@pytest.mark.parametrize('change', ['artifact', 'base', 'missing_pin', 'probe_cleanup'])
def test_canary_bundle_rejects_changed_or_incomplete_evidence(tmp_path, change):
    root, model, pins, base = fixture(tmp_path)
    if change == 'artifact': (root / 'train.log').write_text('changed')
    elif change == 'base': (model / 'model.safetensors').write_bytes(b'changed')
    elif change == 'missing_pin': del pins['initial.safetensors']
    else:
        path = root / 'probe-execution.json'; value = json.loads(path.read_text()); value['process_group_gone'] = False
        path.write_text(json.dumps(value)); pins[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError): verify_canary_bundle(root, model, pins, base)

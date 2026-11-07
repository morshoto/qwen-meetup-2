"""Propose a fixed synthetic one-step 14B canary; never start training."""
import hashlib
import json
import os
import stat
from pathlib import Path

_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
_EXAMPLE = {'messages': [{'role': 'user', 'content': 'What is two plus two? Answer only the numeral.'},
                         {'role': 'assistant', 'content': '4'}]}


def build_canary_config(*, model_dir, data_dir, adapter_dir):
    try:
        raw_model, raw_data, raw_adapter = (Path(x).expanduser() for x in (model_dir, data_dir, adapter_dir))
        if any(path.is_symlink() for path in (raw_model, raw_data, raw_adapter)):
            raise ValueError('canary paths may not be symlinks')
        model, data = raw_model.resolve(strict=True), raw_data.resolve(strict=True)
        adapter = raw_adapter.resolve(strict=False)
        info = data.stat()
        if (not model.is_dir() or model.name != _REVISION or not stat.S_ISDIR(info.st_mode) or
                info.st_uid != os.getuid() or info.st_mode & 0o077 or raw_adapter.exists() or
                not adapter.parent.is_dir()):
            raise ValueError('invalid canary model, data or output')
        paths = (model, data, adapter)
        if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i + 1:]):
            raise ValueError('canary paths overlap')
    except (OSError, TypeError, RuntimeError) as exc:
        raise ValueError('invalid canary paths') from exc
    dataset = json.dumps(_EXAMPLE, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n'
    config = {'model': str(model), 'train': True, 'fine_tune_type': 'lora', 'optimizer': 'adam',
              'optimizer_config': {'adam': {}, 'adamw': {}, 'muon': {}, 'sgd': {}, 'adafactor': {}},
              'data': str(data), 'seed': 20261111, 'num_layers': 8, 'batch_size': 1, 'iters': 1,
              'val_batches': 0, 'learning_rate': 0.0001, 'steps_per_report': 1, 'steps_per_eval': 2,
              'resume_adapter_file': None, 'adapter_path': str(adapter), 'save_every': 1,
              'test': False, 'test_batches': 500, 'max_seq_length': 512, 'grad_checkpoint': False,
              'grad_accumulation_steps': 1, 'lr_schedule': None,
              'lora_parameters': {'rank': 8, 'scale': 2.0, 'dropout': 0.0},
              'mask_prompt': True, 'report_to': None, 'project_name': None}
    return {'status': 'canary_configuration_proposed_unverified', 'execution_ready': False,
            'promotion_allowed': False, 'thinking_mode': 'off', 'dataset_jsonl': dataset,
            'dataset_sha256': hashlib.sha256(dataset.encode()).hexdigest(), 'trainer_config': config}

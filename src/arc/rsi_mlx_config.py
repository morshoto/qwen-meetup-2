"""Preview proposed MLX settings without loading a model or starting a process."""
import re
from pathlib import Path

from src.arc.rsi_training_preparation import _MODEL, _REVISION


def preview_mlx_configuration(*, plan, phase, model_dir, generation_data_dir, canary_data_dir,
                              adapter_dir, previous_adapter_file):
    if (not isinstance(plan, dict) or plan.get('status') != 'quarantined_not_authorized' or
            plan.get('training_allowed') is not False or plan.get('model_id') != _MODEL or
            plan.get('model_revision') != _REVISION or phase not in ('preflight', 'generation') or
            type(plan.get('seed')) is not int or plan['seed'] < 0 or
            type(plan.get('generation')) is not int or plan['generation'] not in (0, 1)):
        raise ValueError('invalid training preview identity')
    parent = plan.get('parent_adapter')
    if ((plan['generation'] == 0 and (parent != 'none' or previous_adapter_file is not None)) or
            (plan['generation'] == 1 and (not isinstance(parent, str) or
             not re.fullmatch(r'[0-9a-f]{64}', parent) or previous_adapter_file is None))):
        raise ValueError('invalid generation parent adapter')
    fixed = {'batch_size': 1, 'learning_rate': 0.0001, 'lora_rank': 8, 'lora_alpha': 16,
             'max_seq_length': 2048, 'num_layers': 2, 'mask_prompt': True}
    preflight, training = plan.get('preflight'), plan.get('generation_training')
    for settings, steps, timeout in ((preflight, 1, 300), (training, 20, 1800)):
        if (not isinstance(settings, dict) or any(settings.get(key) != value or
                type(settings.get(key)) is not type(value) for key, value in fixed.items()) or
                type(settings.get('max_steps')) is not int or settings['max_steps'] != steps or
                type(settings.get('timeout_seconds')) is not int or settings['timeout_seconds'] != timeout):
            raise ValueError('proposed training settings differ')
    if preflight.get('data_source') != 'separate_synthetic_canary_not_registered' or 'records' in preflight:
        raise ValueError('preflight data is not separate')
    try:
        model, generation_data, canary_data, adapter = (Path(value).expanduser().resolve() for value in
            (model_dir, generation_data_dir, canary_data_dir, adapter_dir))
        previous = Path(previous_adapter_file).expanduser().resolve() if previous_adapter_file is not None else None
    except (TypeError, OSError, RuntimeError) as exc:
        raise ValueError('invalid configuration paths') from exc
    paths = (model, generation_data, canary_data, adapter)
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError('model, datasets and output must be separate')
    if previous is not None and (previous == adapter or adapter in previous.parents):
        raise ValueError('parent adapter must be outside output')
    settings = preflight if phase == 'preflight' else training
    steps = settings['max_steps']
    config = {'model': str(model), 'train': True, 'fine_tune_type': 'lora', 'optimizer': 'adam',
              'data': str(canary_data if phase == 'preflight' else generation_data), 'seed': plan['seed'],
              'num_layers': fixed['num_layers'], 'batch_size': fixed['batch_size'], 'iters': steps,
              'val_batches': 0, 'learning_rate': fixed['learning_rate'], 'steps_per_report': 1,
              'steps_per_eval': steps + 1, 'resume_adapter_file': str(previous) if previous else None,
              'adapter_path': str(adapter), 'save_every': steps, 'max_seq_length': fixed['max_seq_length'],
              'grad_checkpoint': False, 'lora_parameters': {'rank': fixed['lora_rank'],
                  'scale': fixed['lora_alpha'] / fixed['lora_rank'], 'dropout': 0.0},
              'mask_prompt': True, 'report_to': None}
    return {'status': 'configuration_preview_unverified', 'training_allowed': False,
            'phase': phase, 'timeout_seconds': settings['timeout_seconds'], 'config': config,
            'parent_adapter_claim': parent, 'parent_adapter_verified': False}

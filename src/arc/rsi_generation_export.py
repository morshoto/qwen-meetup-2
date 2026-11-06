"""Prepare generation-only bytes; this does not enforce process isolation."""
import hashlib
import json

from src.arc.data import Task, validate_grid
from src.arc.rsi_allocation_review import review_task_allocation
from src.arc.rsi_slot_audit import manifest_sha256
from src.arc.rsi_transport_audit import _json_object


def prepare_generation_export(dataset_bytes, bounded_bytes, selection, *, expected_bounded_sha256):
    review = review_task_allocation(dataset_bytes, bounded_bytes, selection,
                                    expected_bounded_sha256=expected_bounded_sha256)
    data = _json_object(dataset_bytes)
    result = {}
    try:
        for task_id in selection['generation_task_ids']:
            task = data[task_id]
            if set(task) != {'train', 'test'} or not task['train'] or not task['test']:
                raise ValueError('invalid generation task')
            train = []
            for pair in task['train']:
                if set(pair) != {'input', 'output'}:
                    raise ValueError('invalid training example')
                train.append({'input': validate_grid(pair['input']), 'output': validate_grid(pair['output'])})
            test = []
            for pair in task['test']:
                if set(pair) != {'input'}:
                    raise ValueError('test labels or extra fields present')
                test.append({'input': validate_grid(pair['input'])})
            result[task_id] = {'train': train, 'test': test}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid generation task') from exc
    raw = json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return raw, {'status': 'generation_export_unverified', 'execution_ready': False,
                 'isolation_verified': False, 'count': len(result),
                 'source_sha256': review['dataset_sha256'], 'selection_sha256': review['selection_sha256'],
                 'generation_manifest_sha256': review['id_manifest_sha256']['generation'],
                 'export_sha256': hashlib.sha256(raw).hexdigest()}


def load_generation_export(raw, *, expected_export_sha256, task_ids, expected_manifest_sha256):
    """Load a bound generation-only input without reading any source dataset."""
    if (not isinstance(raw, bytes) or not isinstance(task_ids, list) or not task_ids or
            any(not isinstance(key, str) or not key for key in task_ids) or
            len(set(task_ids)) != len(task_ids) or
            hashlib.sha256(raw).hexdigest() != expected_export_sha256 or
            manifest_sha256(task_ids) != expected_manifest_sha256):
        raise ValueError('generation export identity differs')
    data = _json_object(raw)
    if set(data) != set(task_ids):
        raise ValueError('generation export task IDs differ')
    tasks = {}
    try:
        for key, item in data.items():
            if (not isinstance(item, dict) or set(item) != {'train', 'test'} or
                    not isinstance(item['train'], list) or not item['train'] or
                    not isinstance(item['test'], list) or not item['test']):
                raise ValueError('invalid generation task')
            if any(not isinstance(pair, dict) or set(pair) != {'input', 'output'} for pair in item['train']):
                raise ValueError('invalid training example')
            if any(not isinstance(pair, dict) or set(pair) != {'input'} for pair in item['test']):
                raise ValueError('invalid test example')
            tasks[key] = Task(key, tuple((validate_grid(p['input']), validate_grid(p['output']))
                                         for p in item['train']),
                              tuple(validate_grid(p['input']) for p in item['test']))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid generation task') from exc
    return tasks

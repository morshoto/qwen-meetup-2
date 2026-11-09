"""Assemble offline evaluation checkpoints without reading held-out solutions."""
from __future__ import annotations

from collections import Counter

from src.arc.qwen38_weight_eval_schedule import next_evaluation_action
from src.arc.qwen38_weight_generation import capture_response, create_request
from src.arc.qwen38_rsi_comparison import digest


def assemble_rows(descriptor, tasks, journal):
    """Rebuild paired-score rows; caller must replay the worker artifact journal."""
    action = next_evaluation_action(descriptor, journal)
    groups = {task_id: [] for task_id in descriptor['task_ids']}; errors = Counter()
    for row in journal:
        task_id, index = row['task_id'], row['candidate_index']
        if (row.get('manifest_sha256') != descriptor['sha256'] or row.get('seed') != 20261111 + index or
            task_id not in tasks):
            raise ValueError('evaluation attempt identity mismatch')
        if row['status'] == 'complete':
            request = create_request(task_id, tasks[task_id], candidate_index=index, seed=row['seed'])
            if request['prompt_sha256'] != descriptor['prompt_sha256']:
                raise ValueError('evaluation prompt mismatch')
            capture = capture_response(request, row, descriptor['max_tokens'])
            if any(row.get(k) != value for k, value in capture.items()):
                raise ValueError('evaluation response differs from saved evidence')
            groups[task_id].append(capture)
        elif row['status'] == 'error':
            errors[row['error']] += 1
        elif row['status'] != 'started':
            raise ValueError('invalid evaluation outcome')
    rows = []
    for task_id in descriptor['task_ids']:
        candidates = groups[task_id]
        observed = [row for row in journal if row['task_id'] == task_id]
        if not observed:
            continue
        identity = digest({'manifest': descriptor['comparison_sha256'], 'label': 'evaluation/weights',
                           'task_id': task_id, 'prompt_sha256': descriptor['prompt_sha256'],
                           'variant_sha256': descriptor['adapter_sha256']})
        common = {'task_id': task_id, 'identity': identity, 'manifest_sha256': descriptor['comparison_sha256'],
                  'prompt_sha256': descriptor['prompt_sha256'], 'variant_sha256': descriptor['adapter_sha256']}
        if len(candidates) == 2 and [c['candidate_index'] for c in candidates] == [0, 1] and len(observed) == 2:
            rows.append({**common, 'status': 'complete', 'candidates': candidates,
                         'runtime_seconds': sum(c['seconds'] for c in candidates)})
        else:
            rows.append({**common, 'status': 'incomplete'})
    return {'status': 'complete' if action['status'] == 'complete' else 'incomplete',
            'schedule_status': action['status'], 'errors': dict(errors), 'rows': rows}

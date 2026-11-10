"""Reconstruct both comparison arms from authenticated, ungraded attempts."""
from collections import Counter

from src.arc.qwen38_replication_schedule import next_replication_action
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_generation import capture_response, create_request


def assemble_replication_rows(descriptor, tasks, journal):
    """Caller must independently replay worker artifacts before invoking this."""
    action = next_replication_action(descriptor, journal)
    groups = {(arm, task): [] for arm in ('base', 'adapter') for task in descriptor['task_ids']}
    observed = Counter(); errors = Counter()
    for row in journal:
        arm, task, index = row['arm'], row['task_id'], row['candidate_index']
        if (row.get('manifest_sha256') != descriptor['sha256'] or row.get('seed') != 20261111 + index or task not in tasks):
            raise ValueError('replication attempt identity mismatch')
        observed[arm, task] += 1
        if row['status'] == 'complete':
            request = create_request(task, tasks[task], candidate_index=index, seed=row['seed'])
            if request['prompt_sha256'] != descriptor['prompt_sha256']:
                raise ValueError('replication prompt changed')
            capture = capture_response(request, row, descriptor['max_tokens'])
            if any(row.get(k) != value for k, value in capture.items()):
                raise ValueError('replication response differs from journal')
            groups[arm, task].append(capture)
        elif row['status'] == 'error':
            errors[row['error']] += 1
        elif row['status'] != 'started':
            raise ValueError('invalid replication attempt outcome')
    result = {'schedule_status': action['status'], 'errors': dict(errors), 'base': [], 'adapter': []}
    for arm in ('base', 'adapter'):
        variant = None if arm == 'base' else descriptor['adapter_sha256']
        label = 'evaluation/baseline' if arm == 'base' else 'evaluation/weights'
        for task in descriptor['task_ids']:
            if not observed[arm, task]: continue
            candidates = groups[arm, task]
            identity = digest({'manifest': descriptor['sha256'], 'label': label, 'task_id': task,
                               'prompt_sha256': descriptor['prompt_sha256'], 'variant_sha256': variant})
            row = {'task_id': task, 'identity': identity, 'manifest_sha256': descriptor['sha256'],
                   'prompt_sha256': descriptor['prompt_sha256'], 'variant_sha256': variant}
            if len(candidates) == 2 and [c['candidate_index'] for c in candidates] == [0, 1] and observed[arm, task] == 2:
                row.update({'status': 'complete', 'candidates': candidates,
                            'runtime_seconds': sum(c['seconds'] for c in candidates)})
            else:
                row['status'] = 'incomplete'
            result[arm].append(row)
    return result

"""Offline reporting for a separately registered exploratory replication."""
from src.arc.qwen38_replication_statistics import paired_sign_test
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_paired import paired_report


def score_replication(descriptor, baseline_rows, adapter_rows, verifier):
    """Score authenticated rows; callers must replay source and journal evidence."""
    if (descriptor.get('status') != 'evaluation_ready' or
        descriptor.get('sha256') != digest({k: v for k, v in descriptor.items() if k != 'sha256'})):
        raise ValueError('replication registration is not ready or changed')
    pairs = paired_report(descriptor['task_ids'], baseline_rows, adapter_rows, verifier,
                          manifest_sha256=descriptor['sha256'], prompt_sha256=descriptor['prompt_sha256'],
                          adapter_sha256=descriptor['adapter_sha256'])
    pairs['comparison'] = 'new exploratory replication tasks'
    counts = pairs['paired']
    complete = counts['complete'] == len(descriptor['task_ids'])
    result = {'status': 'complete' if complete else 'incomplete', 'paired': pairs,
              'uncertainty': paired_sign_test(counts['gained'], counts['lost']) if counts['complete'] else None,
              'uncertainty_scope': 'complete paired tasks only',
              'limitations': ['Exploratory tasks; unknown semantic and pretraining exposure.',
                              'Recorded candidates are not assumed independent.',
                              'Mixed-source distillation is not demonstrated recursive improvement.']}
    result['sha256'] = digest(result)
    return result

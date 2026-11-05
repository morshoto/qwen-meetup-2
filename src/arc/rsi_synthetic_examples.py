"""Retain provisional model-generated examples without asserting provenance."""
import hashlib

from src.arc.ast_fingerprint import stable_ast_sha256
from src.arc.data import Task, validate_grid
from src.arc.rsi_slot_audit import task_sha256


def retain_synthetic_examples(task, code, candidate):
    if not isinstance(task, Task) or not isinstance(code, str) or not isinstance(candidate, dict):
        raise ValueError('invalid source task or candidate')
    try:
        source_hash = hashlib.sha256(code.strip().encode()).hexdigest()
        ast_hash = stable_ast_sha256(code)
        synthetic = candidate['synthetic_task']
        expected_train = [{'input': grid, 'output': target} for grid, target in task.train]
        if (candidate.get('status') != 'verified_candidate' or candidate.get('eligible_for_training') is not False or
                candidate.get('source_type') != 'generator_distillation' or
                candidate.get('source_sha256') != source_hash or candidate.get('ast_sha256') != ast_hash or
                not isinstance(synthetic, dict) or synthetic.get('parent_task_id') != task.id or
                synthetic.get('train') != expected_train or synthetic.get('query_inputs') != list(task.test_inputs)):
            raise ValueError('candidate differs from source task')
        outputs = synthetic['query_outputs']
        if not isinstance(outputs, list) or not task.test_inputs or len(outputs) != len(task.test_inputs):
            raise ValueError('generated output count differs')
        examples = [{'input': validate_grid(grid), 'output': validate_grid(output)}
                    for grid, output in zip(task.test_inputs, outputs)]
    except (KeyError, TypeError, SyntaxError, RecursionError) as exc:
        raise ValueError('invalid generated examples') from exc
    return {'status': 'synthetic_examples_unverified', 'training_allowed': False,
            'label_origin': 'model_generated_program', 'parent_task_id': task.id,
            'parent_task_sha256': task_sha256(task), 'source_sha256': source_hash,
            'examples': examples}

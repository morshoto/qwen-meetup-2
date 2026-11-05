"""Construct quarantined SFT drafts; never assert provenance or train eligibility."""
import hashlib
import re

from src.arc.ast_fingerprint import stable_ast_sha256
from src.arc.data import Task
from src.arc.e5_selfplay import solve_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT

_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
_HASH = re.compile(r'^[0-9a-f]{64}$')


def make_sft_draft(task: Task, code: str, candidate: dict, novelty: dict,
                   provenance: dict, *, generation: int, parent_adapter: str) -> dict:
    if (type(generation) is not int or generation not in (0, 1) or
            (generation == 0 and parent_adapter != 'none') or
            (generation == 1 and (not isinstance(parent_adapter, str) or not _HASH.fullmatch(parent_adapter)))):
        raise ValueError('invalid generation parent adapter')
    if (not isinstance(provenance, dict) or provenance.get('source_type') != 'generator_distillation' or
            provenance.get('model_id') != _MODEL or provenance.get('model_revision') != _REVISION or
            provenance.get('execution_mode') not in {'fake_fixture', 'unverified_capture', 'live_model'} or
            not isinstance(provenance.get('request_id'), str) or not provenance['request_id'] or
            not isinstance(provenance.get('raw_response_sha256'), str) or
            not _HASH.fullmatch(provenance['raw_response_sha256'])):
        raise ValueError('incomplete or mismatched generator provenance')
    if not isinstance(task, Task) or not isinstance(code, str) or not isinstance(candidate, dict):
        raise ValueError('invalid candidate source or task')
    source_hash = hashlib.sha256(code.strip().encode()).hexdigest()
    try:
        ast_hash = stable_ast_sha256(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError('invalid candidate source') from exc
    if (candidate.get('status') != 'verified_candidate' or candidate.get('eligible_for_training') is not False or
            candidate.get('source_type') != 'generator_distillation' or
            candidate.get('source_sha256') != source_hash or candidate.get('ast_sha256') != ast_hash):
        raise ValueError('candidate source does not match provisional verification')
    if (not isinstance(novelty, dict) or novelty.get('status') != 'novel_unverified' or
            novelty.get('eligible_for_training') is not False or novelty.get('source_sha256') != source_hash or
            novelty.get('ast_sha256') != ast_hash or
            novelty.get('behavior_sha256') != candidate.get('private_behavior_sha256') or
            novelty.get('probe_set_sha256') != candidate.get('private_probe_set_sha256') or
            not isinstance(novelty.get('reference_pool_sha256'), str) or
            not _HASH.fullmatch(novelty['reference_pool_sha256'])):
        raise ValueError('candidate has no matching provisional novelty result')
    synthetic = candidate.get('synthetic_task')
    expected_train = [{'input': grid, 'output': output} for grid, output in task.train]
    if (not isinstance(synthetic, dict) or synthetic.get('parent_task_id') != task.id or
            synthetic.get('train') != expected_train or synthetic.get('query_inputs') != list(task.test_inputs)):
        raise ValueError('candidate synthetic task does not match its parent')
    synthetic_task = Task(task.id, tuple((pair['input'], pair['output']) for pair in synthetic['train']),
                          tuple(synthetic['query_inputs']))
    safe_provenance = {key: provenance[key] for key in
                       ('source_type', 'model_id', 'model_revision', 'request_id',
                        'raw_response_sha256', 'execution_mode')}
    return {'status': 'pending_independent_audit', 'eligible_for_training': False,
            'source_type': 'generator_distillation', 'parent_task_id': task.id,
            'generation': generation, 'parent_adapter': parent_adapter,
            'source_sha256': source_hash, 'ast_sha256': ast_hash,
            'behavior_sha256': novelty['behavior_sha256'], 'probe_set_sha256': novelty['probe_set_sha256'],
            'reference_pool_sha256': novelty['reference_pool_sha256'], 'provenance': safe_provenance,
            'mask_prompt': True, 'loss_target': 'assistant_completion_only',
            'messages': [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                         {'role': 'user', 'content': solve_prompt(synthetic_task)},
                         {'role': 'assistant', 'content': code}]}

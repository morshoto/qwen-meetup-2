"""Build current reconstructions of historical prompt digests."""
import hashlib
import json

from src.arc.agent import _code_analysis_prompt, _prompt_json
from src.arc.data import Task, validate_grid
from src.arc.prompting import system_prompt_for_kind


def historical_prompt_hashes(tasks):
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('tasks must be a nonempty list')
    result = {}
    seen = set()
    for task in tasks:
        if not isinstance(task, Task) or not isinstance(task.id, str) or not task.id or task.id in seen:
            raise ValueError('invalid or repeated task')
        if not task.train or not task.test_inputs:
            raise ValueError('task examples missing')
        for input_grid, output_grid in task.train:
            validate_grid(input_grid)
            validate_grid(output_grid)
        for grid in task.test_inputs:
            validate_grid(grid)
        seen.add(task.id)
        prompts = {'direct': _prompt_json({'instruction': 'Return a JSON array of output grids, one per test input.',
                                           'task': task.prompt_data()}),
                   'analysis': _code_analysis_prompt(task)}
        for kind, prompt in prompts.items():
            result[f'{task.id}/{kind}'] = hashlib.sha256(prompt.encode()).hexdigest()
            for mode in (False, True):
                messages = [{'role': 'system', 'content': system_prompt_for_kind(kind, analysis_mode=mode)},
                            {'role': 'user', 'content': prompt}]
                result[f'{task.id}/{kind}/wire_{mode}'] = hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return result

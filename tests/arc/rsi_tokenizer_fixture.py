"""Read-only local tokenizer format fixture; never calls a model or trainer.

Run with MLX-LM 0.30.7 and a local revision directory as the sole argument.
The fabricated rows are protocol fixtures, not live-model provenance.
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from src.arc.data import Task
from src.arc.e5_selfplay import solve_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.rsi_local_tokenizer import audit_local_tokenizer
from src.arc.rsi_training_preparation import prepare_training_inputs
from tests.arc.test_rsi_batch_precheck import row


def main(model_dir):
    model = Path(model_dir)
    tasks = {f't{i}': Task(f't{i}', (([[1, 2]], [[2, 1]]),), ([[3, 4]],)) for i in range(5)}
    drafts = [row(i) for i in range(5)]
    for draft in drafts:
        draft['messages'][0]['content'] = FINAL_ANSWER_SYSTEM_PROMPT
        draft['messages'][1]['content'] = solve_prompt(tasks[draft['parent_task_id']])
    with tempfile.TemporaryDirectory() as tmp:
        prepare_training_inputs(root=tmp, drafts=drafts, tasks=tasks, generation=0,
                                parent_adapter='none', registration_sha256='a' * 64, seed=42)
        plan = json.loads((Path(tmp) / 'training.preparation.json').read_text())
        def digest(name):
            return hashlib.sha256((model / name).read_bytes()).hexdigest()
        result = audit_local_tokenizer(plan, tasks=tasks, model_dir=model,
            expected_tokenizer_sha256=digest('tokenizer.json'),
            expected_config_sha256=digest('tokenizer_config.json'))
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m tests.arc.rsi_tokenizer_fixture MODEL_DIRECTORY')
    main(sys.argv[1])

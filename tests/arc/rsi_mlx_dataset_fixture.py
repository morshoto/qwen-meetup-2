"""Exercise the local MLX dataset loader on fabricated rows; never train."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from src.arc.data import Task
from src.arc.e5_selfplay import solve_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.rsi_local_tokenizer import _REVISION, _load_components, _sha
from src.arc.rsi_mlx_dataset import audit_mlx_dataset
from src.arc.rsi_sft_materialize import materialize_quarantined_dataset
from src.arc.rsi_training_preparation import prepare_training_inputs
from tests.arc.test_rsi_batch_precheck import row


def main(model_dir):
    model = Path(model_dir).expanduser().resolve()
    if not model.is_dir() or model.name != _REVISION:
        raise ValueError('model path must be the proposed revision directory')
    paths = [model / 'tokenizer.json', model / 'tokenizer_config.json']
    before = [_sha(path) for path in paths]
    tokenizer, render, version = _load_components(model)
    from mlx_lm.tuner.datasets import load_local_dataset
    tasks = {f't{i}': Task(f't{i}', (([[1, 2]], [[2, 1]]),), ([[3, 4]],)) for i in range(5)}
    drafts = [row(i) for i in range(5)]
    for draft in drafts:
        draft['messages'][0]['content'] = FINAL_ANSWER_SYSTEM_PROMPT
        draft['messages'][1]['content'] = solve_prompt(tasks[draft['parent_task_id']])
    with tempfile.TemporaryDirectory() as root:
        prepare_training_inputs(root=root, drafts=drafts, tasks=tasks, generation=0,
                                parent_adapter='none', registration_sha256='a' * 64, seed=42)
        raw = (Path(root) / 'training.preparation.json').read_bytes()
        args = dict(root=root, preparation_bytes=raw, expected_preparation_sha256=hashlib.sha256(raw).hexdigest(),
                    tasks=tasks, tokenizer=tokenizer, render=render)
        result = materialize_quarantined_dataset(**args)
        report = audit_mlx_dataset(**args, expected_dataset_sha256=result['dataset_jsonl_sha256'],
                                   load_local_dataset=load_local_dataset)
    if [_sha(path) for path in paths] != before:
        raise ValueError('local tokenizer changed during audit')
    report.update({'mlx_lm_version': version, 'model_revision': _REVISION,
                   'tokenizer_sha256': before[0], 'tokenizer_config_sha256': before[1],
                   'fixture_only': True})
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m tests.arc.rsi_mlx_dataset_fixture MODEL_DIRECTORY')
    main(sys.argv[1])

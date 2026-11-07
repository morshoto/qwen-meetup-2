"""Optional, tokenizer-only integration with the pinned local MLX-LM trainer."""
import json
import os
import tempfile
import types
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get('LT_PHASE_B_TRAINER_PREFLIGHT') == '1', 'local MLX-LM preflight not requested')
class TrainerPreflightTests(unittest.TestCase):
    def test_chat_jsonl_masks_prompt_and_keeps_thinking_off_prefix(self):
        from mlx_lm.tuner.datasets import load_local_dataset
        from transformers import AutoTokenizer

        path = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        from src.arc.data import Task, Verifier
        from src.arc.lt_phase_b_dataset import build_dataset
        ids = ['a', 'b', 'c']
        tasks = {x: Task(x, (([[ord(x) % 10]], [[1]]),), ([[ord(x) % 10]],)) for x in ids}
        calls = [{'candidate': i, 'raw': '[[[1]]]', 'usage': None, 'error': None} for i in (0, 1)]
        rows = [{'task_id': x, 'model': '14b', 'status': 'complete',
                 'candidates': [[[[1]]], [[[1]]]], 'calls': calls,
                 'cleanup': {'leader_reaped': True, 'group_absent': True}} for x in ids]
        output = build_dataset(tasks, rows, Verifier({x: [[[1]]] for x in ids}),
            comparison_tasks={'x': Task('x', (([[0]], [[1]]),), ([[0]],))}, source={'model_id': 'mlx-community/Qwen3-14B-4bit',
            'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
            'source_code_sha256': 'a' * 64, 'run_id': 'synthetic'})
        messages = output['train'][0]['messages']
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'train.jsonl').write_text(json.dumps({'messages': messages}) + '\n')
            train, valid, test = load_local_dataset(Path(tmp), tokenizer, types.SimpleNamespace(mask_prompt=True))
            self.assertEqual((len(train), len(valid), len(test)), (1, 0, 0))
            tokens, offset = train.process(train[0])
        prefix = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True,
            enable_thinking=False, return_dict=False)
        self.assertGreater(offset, 0)
        self.assertLess(offset, len(tokens))
        self.assertLess(len(tokens), 512)
        self.assertEqual(tokens[:len(prefix)], prefix)
        self.assertLess(offset, len(prefix))
        self.assertIn('[[[1]]]', tokenizer.decode(tokens[offset:]))


if __name__ == '__main__': unittest.main()

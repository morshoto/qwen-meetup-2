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
        from src.arc.lt_live_backend import PROMPT
        import hashlib
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for row in rows:
            task = tasks[row['task_id']]
            row.update(run_sha256='c' * 64, source_code_sha256='a' * 64, adapter_state='off', adapter_sha256=None,
                       public_task_sha256=digest(task.prompt_data()),
                       prompt_sha256=digest({'instruction': PROMPT, 'task': task.prompt_data()}))
        output = build_dataset(tasks, rows, Verifier({x: [[[1]]] for x in ids}),
            comparison_tasks={'x': Task('x', (([[0]], [[1]]),), ([[0]],))}, source={'model_id': 'mlx-community/Qwen3-14B-4bit',
            'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
            'source_code_sha256': 'a' * 64, 'run_sha256': 'c' * 64, 'run_id': 'synthetic'})
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

    def test_thinking_off_policy_masks_the_inference_prefix(self):
        from mlx_lm.utils import load_tokenizer
        from mlx_lm.tuner.datasets import ChatDataset
        from src.arc.mlx_chat_template import install_training_template_policy
        path = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
        install_training_template_policy('off')
        tokenizer = load_tokenizer(path)
        messages = [{'role': 'user', 'content': 'Synthetic arithmetic: 2 + 2?'},
                    {'role': 'assistant', 'content': '4'}]
        dataset = ChatDataset([{'messages': messages}], tokenizer, mask_prompt=True)
        tokens, offset = dataset.process(dataset[0])
        prefix = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True,
            enable_thinking=False, return_dict=False)
        self.assertEqual(tokens[:len(prefix)], prefix)
        self.assertEqual(offset, len(prefix))
        self.assertIn('4', tokenizer.decode(tokens[offset:]))
        self.assertLess(len(tokens), 512)

    def test_synthetic_canary_config_and_dataset_load_with_native_trainer(self):
        from mlx_lm.lora import CONFIG_DEFAULTS
        from mlx_lm.tuner.datasets import load_local_dataset
        from mlx_lm.utils import load_tokenizer
        from src.arc.lt_phase_b_canary_config import build_canary_config
        from src.arc.mlx_chat_template import install_training_template_policy
        install_training_template_policy('off')
        model = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
        tokenizer = load_tokenizer(model)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'data'; data.mkdir(mode=0o700)
            config = build_canary_config(model_dir=model, data_dir=data, adapter_dir=Path(tmp) / 'adapter')
            trainer = config['trainer_config']
            self.assertTrue(set(trainer) <= set(CONFIG_DEFAULTS))
            (data / 'train.jsonl').write_text(config['dataset_jsonl'])
            train, valid, test = load_local_dataset(data, tokenizer, types.SimpleNamespace(**trainer))
            self.assertEqual((len(train), len(valid), len(test)), (1, 0, 0))
            tokens, offset = train.process(train[0])
            messages = json.loads(config['dataset_jsonl'])['messages']
            prefix = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True,
                                                   enable_thinking=False, return_dict=False)
            self.assertEqual(tokens[:len(prefix)], prefix)
            self.assertEqual(offset, len(prefix))
            self.assertGreater(len(tokens), offset)
            self.assertLessEqual(len(tokens), trainer['max_seq_length'])

    def test_verified_fixture_dataset_passes_native_mask_audit(self):
        from mlx_lm.tuner.datasets import ChatDataset
        from mlx_lm.utils import load_tokenizer
        from src.arc.lt_phase_b_sft_preflight import audit_sft_dataset
        from src.arc.mlx_chat_template import install_training_template_policy
        from tests.arc import test_lt_phase_b_sft_preflight as fixture_module
        fixture = fixture_module.SFTPreflightTests('test_audits_exact_verified_examples_and_prefix')
        fixture.setUp()
        path = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
        install_training_template_policy('off')
        tokenizer = load_tokenizer(path)
        report = audit_sft_dataset(dataset=fixture.dataset, tasks=fixture.fixture.tasks, tokenizer=tokenizer,
            dataset_factory=lambda example, tok: ChatDataset([example], tok, mask_prompt=True))
        self.assertEqual((report['train_count'], report['valid_count']), (2, 1))
        self.assertGreater(report['maximum_tokens'], 0)
        self.assertLessEqual(report['maximum_tokens'], 512)
        self.assertFalse(report['training_allowed'])


if __name__ == '__main__': unittest.main()

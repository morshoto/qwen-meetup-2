"""Exercise the real subprocess and journal using an isolated fake vLLM package."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.data import Verifier
from src.cli.lt_single import prepare, run, score_directory


class VLLMWorkerIntegrationTests(unittest.TestCase):
    def test_subprocess_generates_two_candidates_and_scores_then_resumes(self):
        task = {'train': [{'input': [[1]], 'output': [[1]]}], 'test': [{'input': [[2]]}]}
        manifest = {'task_ids': ['a'], 'sha256': 'a' * 64, 'batch_size': 10,
                    'task_timeout_seconds': 15, 'model': {'model_id': 'example/model', 'revision': 'b' * 40},
                    'backend': {'name': 'vllm', 'version': '0.33.0'}, 'solutions_sha256': 'c' * 64,
                    'protocol': 'lt-single-model-exploratory', 'candidate_count': 2,
                    'temperature': 0.6, 'seed_base': 20261111, 'max_tokens': 4096, 'thinking': 'off'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data/raw').mkdir(parents=True)
            (root / 'data/raw/arc-agi_training_challenges.json').write_text(json.dumps({'a': task}))
            fake = root / 'fake'
            fake.mkdir()
            (fake / 'vllm-0.33.0.dist-info').mkdir()
            (fake / 'vllm-0.33.0.dist-info/METADATA').write_text('Metadata-Version: 2.1\nName: vllm\nVersion: 0.33.0\n')
            (fake / 'vllm.py').write_text('''\nfrom types import SimpleNamespace\nclass SamplingParams:\n    def __init__(self, **kwargs): self.seed = kwargs['seed']\nclass Tokenizer:\n    def apply_chat_template(self, messages, **kwargs):\n        assert kwargs['enable_thinking'] is False\n        return messages[0]['content']\nclass LLM:\n    def __init__(self, **kwargs): assert kwargs['dtype'] == 'bfloat16'\n    def get_tokenizer(self): return Tokenizer()\n    def generate(self, prompts, params, **kwargs):\n        assert params.seed in (20261111, 20261112)\n        output = SimpleNamespace(text='[[[2]]]', token_ids=[1, 2], finish_reason='stop')\n        return [SimpleNamespace(outputs=[output], prompt_token_ids=[1, 2, 3])]\n''')
            config = root / 'config.json'
            config.write_text('{}')
            dest = root / 'run'
            repo = str(Path(__file__).resolve().parents[2])
            with patch.dict(os.environ, {'PYTHONPATH': os.pathsep.join((str(fake), repo))}), \
                 patch('src.cli.lt_single.ROOT', root), patch('src.cli.lt_single.build', return_value=manifest), \
                 patch('src.cli.lt_single.verify'), patch('src.cli.lt_single.verify_model', return_value=root), \
                 patch('src.cli.lt_single.importlib.metadata.version', return_value='0.33.0'), \
                 patch('src.cli.lt_single.load_arc', return_value=({'a': object()}, Verifier({'a': [[[2]]]}))):
                prepare(dest, config)
                self.assertEqual(run(dest), 1)
                row = json.loads((dest / 'single/a.result.json').read_text())
                self.assertEqual(row['status'], 'complete', row.get('worker_diagnostics'))
                self.assertEqual(row['candidates'], [[[[2]]], [[[2]]]])
                self.assertTrue(row['cleanup']['leader_reaped'])
                self.assertTrue(row['cleanup']['group_absent'])
                report = score_directory(dest)
                self.assertEqual(report['models']['single']['status'], 'complete')
                self.assertEqual(report['models']['single']['pass_at_2'], 1)
                self.assertEqual(run(dest), 0)

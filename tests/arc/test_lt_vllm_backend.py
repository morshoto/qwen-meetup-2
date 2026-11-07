import json
import sys
import types
import unittest
from unittest.mock import patch


class VLLMBackendTests(unittest.TestCase):
    def test_direct_generation_uses_registered_settings_and_reports_usage(self):
        calls = {}
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls['template'] = (messages, kwargs)
                return 'formatted'
        class LLM:
            def __init__(self, **kwargs): calls['init'] = kwargs
            def get_tokenizer(self): return Tokenizer()
            def generate(self, prompts, sampling_params, **kwargs):
                calls['generate'] = (prompts, sampling_params, kwargs)
                out = types.SimpleNamespace(text='[[[2]]]', token_ids=[1, 2], finish_reason='stop')
                return [types.SimpleNamespace(outputs=[out], prompt_token_ids=[1, 2, 3])]
        def sampling(**kwargs): return kwargs
        with patch.dict(sys.modules, {'vllm': types.SimpleNamespace(LLM=LLM, SamplingParams=sampling)}):
            from src.arc.lt_vllm_backend import VLLMBackend
            backend = VLLMBackend('/snapshot')
            result = backend.generate(json.dumps({'task': {'test': [{}]}}), kind='direct', candidate=1)
        self.assertEqual(json.loads(result), [[[2]]])
        self.assertEqual(calls['init']['dtype'], 'bfloat16')
        self.assertEqual(calls['init']['model'], '/snapshot')
        self.assertFalse(calls['template'][1]['enable_thinking'])
        self.assertEqual(calls['generate'][1]['seed'], 20261112)
        self.assertEqual(calls['generate'][1]['max_tokens'], 4096)
        self.assertEqual(calls['generate'][1]['temperature'], 0.6)
        self.assertEqual(backend.stats['prompt_tokens'], 3)
        self.assertEqual(backend.stats['completion_tokens'], 2)
        self.assertEqual(backend.raw, '[[[2]]]')
        with self.assertRaises(ValueError):
            backend.generate('{}', kind='other', candidate=0)

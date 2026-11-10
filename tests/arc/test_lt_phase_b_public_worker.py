import hashlib
import json
import unittest
from types import SimpleNamespace

from src.arc.lt_live_backend import MLXBackend
from src.arc.lt_public import build_public_export
from src.arc.lt_phase_b_public_worker import execute_public_worker


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class PublicWorkerTests(unittest.TestCase):
    def setUp(self):
        self.task = {'train': [{'input': [[0]], 'output': [[1]]}], 'test': [{'input': [[2]]}]}
        self.receipt = {'comparison_sha256': 'a' * 64, 'selected_task_sha256': {'t': sha(self.task)}}
        self.export = build_public_export({'t': self.task}, self.receipt)
        self.prompts = []
        self.created = []

    def factory(self):
        self.created.append(True)
        backend = MLXBackend.__new__(MLXBackend)
        def template(messages, **kwargs):
            self.assertFalse(kwargs['enable_thinking'])
            self.prompts.append(messages[0]['content'])
            return 'synthetic tokenized prompt'
        backend.tokenizer = SimpleNamespace(apply_chat_template=template)
        backend.model = object()
        backend.adapter_state, backend.adapter_sha256 = 'off', None
        backend.mx = SimpleNamespace(random=SimpleNamespace(seed=lambda _: None))
        backend.seed, backend.max_tokens, backend.temperature = 20261111, 4096, .6
        backend.make_sampler = lambda **kwargs: kwargs
        backend.stream_generate = lambda *a, **k: iter([SimpleNamespace(
            text='[[[1]]]', prompt_tokens=10, generation_tokens=4, peak_memory=1., finish_reason='stop')])
        return backend

    def execute(self, **kwargs):
        data = {'export': self.export, 'receipt': self.receipt, 'task_id': 't',
                'expected_run_sha256': 'a' * 64, 'source_code_sha256': 'b' * 64,
                'adapter_state': 'off', 'backend_factory': self.factory}
        data.update(kwargs)
        return execute_public_worker(**data)

    def test_public_prompt_is_bound_to_actual_backend_payload(self):
        row = self.execute()
        self.assertEqual(len(self.prompts), 2)
        self.assertEqual(self.prompts[0], self.prompts[1])
        self.assertEqual(row['prompt_sha256'], hashlib.sha256(self.prompts[0].encode()).hexdigest())
        self.assertEqual(row['public_task_sha256'], sha(self.task))
        self.assertEqual(row['run_sha256'], 'a' * 64)
        self.assertEqual(row['source_code_sha256'], 'b' * 64)
        self.assertEqual(row['adapter_state'], 'off')
        self.assertEqual(row['candidates'], [[[[1]]], [[[1]]]])
        for prompt in self.prompts:
            self.assertEqual(set(json.loads(prompt)['task']['test'][0]), {'input'})

    def test_mismatched_backend_state_fails_before_generation(self):
        def mismatched():
            backend = self.factory()
            backend.adapter_state, backend.adapter_sha256 = 'on', 'd' * 64
            return backend
        with self.assertRaises(ValueError): self.execute(backend_factory=mismatched)
        self.assertEqual(self.prompts, [])
        with self.assertRaises(ValueError):
            self.execute(adapter_state='on', expected_adapter_sha256='e' * 64, backend_factory=mismatched)
        self.assertEqual(self.prompts, [])

    def test_invalid_handoff_fails_before_model_load(self):
        leaked = {'comparison_sha256': 'a' * 64, 'tasks': {'t': {**self.task, 'test': [{'input': [[2]], 'output': [[1]]}]}}}
        for kwargs in ({'export': leaked}, {'expected_run_sha256': 'c' * 64},
                       {'task_id': 'absent'}, {'source_code_sha256': 'bad'}, {'adapter_state': 'other'},
                       {'adapter_state': 'on'}, {'adapter_state': 'off', 'expected_adapter_sha256': 'd' * 64}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.execute(**kwargs)
        self.assertEqual(self.created, [])


if __name__ == '__main__': unittest.main()

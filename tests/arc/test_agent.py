import unittest
import json
import tempfile
from pathlib import Path
from src.arc.agent import FakeLLM, code_agent, direct
from src.arc.data import load_arc
from src.arc.sandbox import FakeSandbox, IDENTITY_CODE, ZERO_CODE

class AgentTests(unittest.TestCase):
    def test_baselines_and_shared_agent(self):
        tasks, verifier = load_arc('tests/fixtures/arc_challenges.json', 'tests/fixtures/arc_solutions.json')
        task = tasks['toy_000']
        a = direct(task, FakeLLM(), 0)
        b = code_agent(task, FakeLLM(), FakeSandbox(), 2, 0)
        c = code_agent(task, FakeLLM(), FakeSandbox(), 2, 0)
        self.assertEqual(b, c)
        self.assertTrue(verifier.matches(task.id, a.outputs))
        self.assertTrue(verifier.matches(task.id, b.outputs))

    def test_repairs_with_training_feedback(self):
        class RepairLLM:
            def generate(self, prompt, *, kind, candidate):
                if kind == 'analysis': return 'copy the input grid unchanged'
                return ZERO_CODE if 'Previous attempts and training-pair feedback:' not in prompt else IDENTITY_CODE
        tasks, _ = load_arc('tests/fixtures/arc_challenges.json', 'tests/fixtures/arc_solutions.json')
        result = code_agent(tasks['toy_001'], RepairLLM(), FakeSandbox(), 1, 0)
        self.assertEqual(len(result.history), 2)
        self.assertIsNone(result.error)

    def test_model_prompts_and_repair_feedback_ignore_held_out_outputs(self):
        class CapturingLLM:
            def __init__(self): self.calls = []
            def generate(self, prompt, *, kind, candidate):
                self.calls.append((kind, prompt))
                if kind == 'direct': return '[[[2]]]'
                if kind == 'analysis': return 'copy input'
                return ZERO_CODE if sum(k == 'code' for k, _ in self.calls) == 1 else IDENTITY_CODE
        with tempfile.TemporaryDirectory() as directory:
            challenges = Path(directory) / 'challenges.json'
            solutions = Path(directory) / 'solutions.json'
            challenges.write_text(json.dumps({'task': {'train': [{'input': [[1]], 'output': [[1]]}],
                                                        'test': [{'input': [[2]]}]}}))
            prompts = []
            for answer in ([[9, 9, 9]], [[8, 8, 8]]):
                solutions.write_text(json.dumps({'task': [answer]}))
                tasks, verifier = load_arc(challenges, solutions)
                self.assertTrue(verifier.matches('task', [answer]))
                backend = CapturingLLM()
                direct(tasks['task'], backend, 0)
                code_agent(tasks['task'], backend, FakeSandbox(), 1, 0, prompt_variant='structured')
                self.assertEqual([k for k, _ in backend.calls], ['direct', 'analysis', 'code', 'code'])
                prompts.append(backend.calls)
            self.assertEqual(prompts[0], prompts[1])
            self.assertIn('Previous attempts and training-pair feedback', prompts[0][-1][1])
            for _, prompt in prompts[0]:
                self.assertNotIn('9 9 9', prompt)
                self.assertNotIn('8 8 8', prompt)

    def test_timeout_is_preserved_for_scoring(self):
        from src.arc.scoring import score
        class TimeoutLLM:
            def generate(self, prompt, *, kind, candidate):
                raise TimeoutError('deadline')
        tasks, verifier = load_arc('tests/fixtures/arc_challenges.json', 'tests/fixtures/arc_solutions.json')
        result = direct(tasks['toy_000'], TimeoutLLM(), 0)
        self.assertEqual(result.error_kind, 'timeout')
        self.assertEqual(score(verifier, 'toy_000', ['timeout']).timeouts, 1)

    def test_invalid_direct_response_retained(self):
        class BadLLM:
            def generate(self, prompt, *, kind, candidate): return 'not json'
        tasks, _ = load_arc('tests/fixtures/arc_challenges.json', 'tests/fixtures/arc_solutions.json')
        result=direct(tasks['toy_000'], BadLLM(), 0)
        self.assertEqual(result.raw_response, 'not json')
        self.assertEqual(result.error_kind, 'invalid')

    def test_python_fence_is_extracted_before_sandbox(self):
        class FencedLLM:
            def generate(self, prompt, *, kind, candidate):
                return 'Solution:\n```python\n' + IDENTITY_CODE + '\n```'
        tasks, verifier = load_arc('tests/fixtures/arc_challenges.json', 'tests/fixtures/arc_solutions.json')
        result = code_agent(tasks['toy_000'], FencedLLM(), FakeSandbox(), 0, 0)
        self.assertEqual(result.code, IDENTITY_CODE)
        self.assertTrue(verifier.matches('toy_000', result.outputs))

    def test_code_prompt_requests_general_concise_complete_function(self):
        class CapturingLLM:
            calls = None
            def __init__(self): self.calls = []
            def generate(self, prompt, *, kind, candidate):
                self.calls.append((kind, prompt))
                if kind == 'analysis': return 'copy the input grid unchanged'
                return IDENTITY_CODE
        tasks, _ = load_arc('tests/fixtures/arc_challenges.json', 'tests/fixtures/arc_solutions.json')
        backend = CapturingLLM()
        result = code_agent(tasks['toy_000'], backend, FakeSandbox(), 0, 0)
        self.assertEqual([kind for kind, _ in backend.calls], ['analysis', 'code'])
        self.assertEqual(result.analysis, 'copy the input grid unchanged')
        analysis_prompt, code_prompt = backend.calls[0][1], backend.calls[1][1]
        instruction = code_prompt.split('\n\n', 1)[0]
        self.assertIn('every training pair', instruction)
        self.assertIn('complete, concise Python function', instruction)
        self.assertIn('Do not copy one training output', instruction)
        self.assertIn('Training examples:', analysis_prompt)
        self.assertIn('Test inputs:', analysis_prompt)
        self.assertIn('Test 1 input:', analysis_prompt)
        self.assertIn('Transformation hypothesis:\ncopy the input grid unchanged', code_prompt)
        self.assertNotIn('test outputs', analysis_prompt.lower())

    def test_direct_prompt_uses_compact_json_without_changing_payload(self):
        class CapturingLLM:
            prompt = None
            def generate(self, prompt, *, kind, candidate):
                self.prompt = prompt
                data = json.loads(prompt)
                return json.dumps([entry['input'] for entry in data['task']['test']])
        tasks, _ = load_arc('tests/fixtures/arc_challenges.json', 'tests/fixtures/arc_solutions.json')
        backend = CapturingLLM()
        result = direct(tasks['toy_000'], backend, 0)
        self.assertIsNone(result.error)
        self.assertEqual(backend.prompt, json.dumps(json.loads(backend.prompt), sort_keys=True, separators=(',', ':')))

    def test_code_grid_prompt_uses_rows_and_spaces(self):
        class CapturingLLM:
            prompt = None
            def generate(self, prompt, *, kind, candidate):
                self.prompt = prompt
                return IDENTITY_CODE
        tasks, _ = load_arc('tests/fixtures/arc_challenges.json', 'tests/fixtures/arc_solutions.json')
        task = tasks['toy_000']
        backend = CapturingLLM()
        code_agent(task, backend, FakeSandbox(), 0, 0)
        input_grid, output_grid = task.train[0]
        render = lambda grid: '\n'.join(' '.join(str(cell) for cell in row) for row in grid)
        self.assertIn(render(input_grid), backend.prompt)
        self.assertIn(render(output_grid), backend.prompt)

import unittest
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6f_repair import run_candidate
from src.arc.e6g_fake import FakeSandbox
from src.arc.e6f_fake import PROGRAMS


class PrivacyTests(unittest.TestCase):
    def test_hidden_failure_never_returns_hidden_data_to_generator(self):
        prompts = []
        class Sandbox:
            def __init__(self):
                self.delegate = FakeSandbox()
                self.calls = 0
            def run(self, code, grid):
                self.calls += 1
                if self.calls > 10:
                    raise RuntimeError('sandbox execution failed: SECRET_HIDDEN_GRID SECRET_OUTPUT')
                return self.delegate.run(code, grid)
        def generate(prompt, index, attempt):
            prompts.append(prompt)
            return PROGRAMS[0], {'request_id': 'fixture'}
        result = run_candidate(index=0, max_repairs=2, generate=generate, sandbox=Sandbox(),
                               emit=lambda event: None, counters=lambda: {},
                               probe_inputs=_quality_grids(88000, 5))
        self.assertEqual(result['status'], 'private_acceptance_rejected')
        self.assertEqual(len(prompts), 1)
        self.assertNotIn('SECRET', ''.join(prompts))


if __name__ == '__main__':
    unittest.main()

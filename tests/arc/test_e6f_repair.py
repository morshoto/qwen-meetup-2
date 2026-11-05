from __future__ import annotations

import json
import unittest

from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.e6f_repair import run_candidate
from src.arc.http_backend import BackendError


BAD = 'def transform(grid):\n    return grid[999]'
BAD2 = 'def transform(grid):\n    return grid[998]'
PROVENANCE = {'source_type': 'generator_distillation', 'request_id': 'fake',
              'raw_response_sha256': 'a' * 64, 'model_revision': 'fake',
              'model_id': 'fake', 'execution_mode': 'fake_fixture', 'thinking_mode': 'off'}


class Sandbox:
    def __init__(self):
        self.calls = 0
        self.delegate = FakeE6Sandbox()

    def run(self, code, grid):
        self.calls += 1
        if code in {BAD, BAD2}:
            raise RuntimeError('sandbox execution failed: Traceback\nIndexError: private detail')
        return self.delegate.run(code, grid)


class RepairTests(unittest.TestCase):
    def run_case(self, replies, *, sandbox=None, repairs=2):
        sandbox = sandbox or Sandbox()
        events, prompts = [], []
        pending = iter(replies)

        def generate(prompt, index, attempt):
            prompts.append(prompt)
            reply = next(pending)
            if isinstance(reply, Exception):
                raise reply
            return reply, dict(PROVENANCE, request_id=f'fake-{attempt}')

        result = run_candidate(index=0, max_repairs=repairs, generate=generate,
            sandbox=sandbox, emit=events.append,
            counters=lambda: {'model_calls': len(prompts), 'sandbox_runs': sandbox.calls},
            probe_inputs=_quality_grids(88000, 5))
        return result, events, prompts

    def test_failure_then_model_repair_uses_only_allowlisted_feedback(self):
        result, events, prompts = self.run_case([BAD, FAKE_PROGRAMS[1]])
        self.assertEqual(result['status'], 'passed')
        self.assertFalse(result['baseline_pass'])
        self.assertTrue(result['repair_pass'])
        self.assertIn('IndexError', prompts[1])
        self.assertNotIn('private detail', prompts[1])
        self.assertNotIn(json.dumps(_quality_grids(88000, 5)[0]), prompts[1])
        self.assertEqual(events[-1]['parent_trial_id'], 'candidate-0-attempt-0')
        self.assertEqual(result['record']['data_source'], 'generator_distillation')

    def test_identical_code_and_repair_limit_stop(self):
        result, _, prompts = self.run_case([BAD, BAD])
        self.assertEqual(result['status'], 'repeated_code')
        self.assertEqual(len(prompts), 2)
        result, _, prompts = self.run_case([BAD, BAD2], repairs=1)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(result['status'], 'repair_limit')

    def test_infrastructure_budget_and_prohibited_operations_do_not_repair(self):
        class Broken(Sandbox):
            def run(self, code, grid):
                raise BackendError('sandbox_unavailable', 'unavailable')

        result, _, prompts = self.run_case([FAKE_PROGRAMS[1]], sandbox=Broken())
        self.assertEqual(result['status'], 'sandbox_infrastructure_error')
        self.assertEqual(len(prompts), 1)
        result, _, prompts = self.run_case([TimeoutError('budget')])
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(len(prompts), 1)
        result, _, prompts = self.run_case(['import os\ndef transform(grid):\n    return grid'])
        self.assertEqual(result['status'], 'prohibited_operation')
        self.assertEqual(len(prompts), 1)

    def test_private_acceptance_failure_is_never_fed_back(self):
        class HiddenFailure(Sandbox):
            def run(self, code, grid):
                if grid in _quality_grids(88000, 5):
                    raise RuntimeError('sandbox execution failed: Traceback\nIndexError: hidden')
                return super().run(code, grid)

        result, events, prompts = self.run_case([FAKE_PROGRAMS[1]], sandbox=HiddenFailure())
        self.assertEqual(result['status'], 'private_acceptance_rejected')
        self.assertEqual(events[-1]['validation_stage'], 'private_acceptance')
        self.assertEqual(len(prompts), 1)


if __name__ == '__main__':
    unittest.main()

import unittest
from src.arc.e7_sandbox import TracedSandbox, _digest


class SandboxTests(unittest.TestCase):
    def test_trace_contains_only_digests_and_counts(self):
        class Budget:
            def reserve_sandbox(self): pass
        class Sandbox:
            def run(self, code, grid): return grid
        events = []
        trace = TracedSandbox(Sandbox(), Budget(), 'sha256:image', events.append)
        trace.public_inputs = {_digest([[9]])}
        trace.run('code', [[9]])
        trace.run('code', [[9]])
        dispatch = [e for e in events if e['event'] == 'sandbox_dispatch']
        self.assertEqual([x['prior_executions_same_code_input'] for x in dispatch], [0, 1])
        self.assertEqual(dispatch[0]['phase'], 'public_validation')
        self.assertNotIn('[[9]]', str(events))


if __name__ == '__main__':
    unittest.main()

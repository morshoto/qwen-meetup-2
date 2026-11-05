import tempfile
import unittest
from pathlib import Path

from src.arc.e7_budget import LayeredBudget
from src.arc.e6g_config import load_e6g_config
from src.pipelines.e5 import E5Budget
from src.pipelines.e6g import _compat


class BudgetTests(unittest.TestCase):
    def test_all_layers_stop_before_dispatch(self):
        _, base = load_e6g_config('configs/e6g.fake.yml')
        with tempfile.TemporaryDirectory() as tmp:
            layers = []
            for i, cap in enumerate((1, 2, 3)):
                config = _compat(base, calls=cap, tokens=50000, sandboxes=cap, seconds=60, seed=42)
                layers.append(E5Budget(Path(tmp) / f'{i}.json', config, resume=False))
            budget = LayeredBudget(*layers)
            budget.reserve_sandbox()
            with self.assertRaisesRegex(TimeoutError, 'candidate_budget_exhausted'):
                budget.reserve_sandbox()
            self.assertEqual([x.state['sandbox_runs'] for x in layers], [1, 1, 1])
            budget.reserve_model('prompt', 'test')
            with self.assertRaisesRegex(TimeoutError, 'candidate_budget_exhausted'):
                budget.reserve_model('prompt', 'test')
            self.assertEqual([x.state['model_calls'] for x in layers], [1, 1, 1])
            for layer in layers:
                layer.finish()


    def test_arm_and_total_limits_also_fail_before_mutation(self):
        _, base = load_e6g_config('configs/e6g.fake.yml')
        for caps, expected in (((3, 1, 3), 'arm'), ((3, 3, 1), 'total')):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                layers = [E5Budget(Path(tmp) / f'{i}.json',
                    _compat(base, calls=cap, tokens=50000, sandboxes=cap, seconds=60, seed=42),
                    resume=False) for i, cap in enumerate(caps)]
                budget = LayeredBudget(*layers)
                budget.reserve_sandbox()
                with self.assertRaisesRegex(TimeoutError, expected + '_budget_exhausted'):
                    budget.reserve_sandbox()
                self.assertEqual([x.state['sandbox_runs'] for x in layers], [1, 1, 1])
                budget.reserve_model('prompt', 'test')
                with self.assertRaisesRegex(TimeoutError, expected + '_budget_exhausted'):
                    budget.reserve_model('prompt', 'test')
                self.assertEqual([x.state['model_calls'] for x in layers], [1, 1, 1])
                for layer in layers:
                    layer.finish()


if __name__ == '__main__':
    unittest.main()

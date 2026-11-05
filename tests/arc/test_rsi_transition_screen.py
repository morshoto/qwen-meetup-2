import tempfile
import unittest
from pathlib import Path

from src.arc.data import Task, Verifier
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_comparison_budget import ComparisonBudget
from src.arc.rsi_comparison_runner import run_comparison
from src.arc.rsi_transition_screen import screen_rsi_chain


class TransitionScreenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = [f'task-{i:03}' for i in range(200)]
        self.tasks = {key: Task(key, (([[1, 2]], [[2, 1]]),), ([[3, 4]],)) for key in self.ids}
        self.verifier = Verifier({key: [[[4, 3]]] for key in self.ids})
        self.seeds = {key: [10000 + 2*i, 10001 + 2*i] for i, key in enumerate(self.ids)}

    def comparison(self, folder, generation, parent, candidate, positive):
        root = self.root / folder
        root.mkdir(mode=0o700)
        budget = ComparisonBudget.create(root, registration_sha256='a' * 64, generation=generation)
        def generate(prompt, *, arm, **kwargs):
            return FAKE_PROGRAMS[0] if positive and arm == 'parent' else FAKE_PROGRAMS[1]
        result = run_comparison(root=root, registration_sha256='a' * 64, generation=generation,
            task_ids=self.ids, tasks=self.tasks, seeds=self.seeds, parent_adapter=parent,
            candidate_adapter=candidate, generate=generate, sandbox=FakeE6Sandbox(),
            verifier=self.verifier, budget=budget)
        return {'root': root, 'sha256': result['sha256']}

    def screen(self, first, second=None):
        return screen_rsi_chain(first=first, second=second, tasks=self.tasks,
                                sandbox=FakeE6Sandbox(), verifier=self.verifier)

    def test_first_screen_and_second_parent_are_bound(self):
        first = self.comparison('first', 0, 'none', 'b' * 64, True)
        second = self.comparison('second', 1, 'b' * 64, 'c' * 64, False)
        result = self.screen(first)
        self.assertEqual(result['status'], 'first_screen_positive_unverified')
        self.assertEqual(result['provisional_selected_adapter'], 'b' * 64)
        self.assertFalse(result['execution_ready'])
        result = self.screen(first, second)
        self.assertEqual(result['status'], 'second_screen_nonpositive_unverified')
        self.assertEqual(result['provisional_selected_adapter'], 'b' * 64)
        self.assertFalse(result['improvement_claim_supported'])
        wrong_parent = self.comparison('wrong-parent', 1, 'd' * 64, 'e' * 64, True)
        with self.assertRaises(ValueError):
            self.screen(first, wrong_parent)
        record = second['root'] / 'comparison-000-candidate.json'
        record.write_text('{}')
        with self.assertRaises(ValueError):
            self.screen(first, second)

    def test_nonpositive_first_stops_and_wrong_parent_fails(self):
        first = self.comparison('first', 0, 'none', 'b' * 64, False)
        second = self.comparison('second', 1, 'c' * 64, 'd' * 64, True)
        self.assertEqual(self.screen(first)['status'], 'first_screen_nonpositive_unverified')
        with self.assertRaises(ValueError):
            self.screen(first, second)

    def test_two_positive_screens_remain_unverified(self):
        first = self.comparison('first', 0, 'none', 'b' * 64, True)
        second = self.comparison('second', 1, 'b' * 64, 'c' * 64, True)
        result = self.screen(first, second)
        self.assertEqual(result['status'], 'second_screen_positive_unverified')
        self.assertEqual(result['provisional_selected_adapter'], 'c' * 64)
        self.assertFalse(result['independently_verified'])
        self.assertFalse(result['improvement_claim_supported'])


if __name__ == '__main__':
    unittest.main()

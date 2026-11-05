import unittest
from pathlib import Path
from src.arc.data import exploration_window, load_arc, split_tasks, validate_grid
from src.arc.scoring import score
from src.arc.sandbox import DockerSandbox
from src.core.config import load_experiment_config


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.tasks, self.verifier = load_arc('tests/fixtures/arc_challenges.json', 'tests/fixtures/arc_solutions.json')

    def test_config(self):
        cfg = load_experiment_config('configs/pilot.cpu.yml')
        self.assertEqual(cfg.agent.candidates, 2)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            text = Path('configs/pilot.cpu.yml').read_text().replace('max_seconds: 300', 'max_seconds: -1')
            path = Path(tmp) / 'bad.yml'; path.write_text(text)
            with self.assertRaises(ValueError): load_experiment_config(path)

    def test_splits(self):
        counts = {'exploration': 4, 'promotion_gate': 2, 'monitoring': 2}
        a = split_tasks(list(self.tasks), 42, counts)
        self.assertEqual(a, split_tasks(list(reversed(list(self.tasks))), 42, counts))
        self.assertEqual(len(set(sum(a.values(), []))), 8)
        with self.assertRaises(ValueError): split_tasks(['x', 'x'], 1, counts)

    def test_exploration_windows_are_bounded_and_disjoint(self):
        splits = split_tasks(list(self.tasks), 42,
                             {'exploration': 4, 'promotion_gate': 2, 'monitoring': 2})
        first = exploration_window(splits, 0, 2)
        second = exploration_window(splits, 2, 2)
        self.assertEqual(first, splits['exploration'][:2])
        self.assertEqual(second, splits['exploration'][2:])
        self.assertFalse(set(first) & set(second))
        with self.assertRaises(ValueError): exploration_window(splits, 3, 2)

    def test_hidden_labels_and_scoring(self):
        task = self.tasks['toy_000']
        self.assertTrue(all('output' not in example for example in task.prompt_data()['test']))
        correct = [[[0]], [[2]]]
        wrong = [[[1]], [[2]]]
        result = score(self.verifier, task.id, [wrong, correct])
        self.assertFalse(result.pass_at_1); self.assertTrue(result.pass_at_2)
        result = score(self.verifier, task.id, ['timeout', [[[99]]]])
        self.assertEqual((result.timeouts, result.invalid), (1, 1))
        self.assertFalse(self.verifier.matches(task.id, [[[0]]]))
        self.assertEqual(score(self.verifier, task.id, [[[[0]]]]).invalid, 1)

    def test_grid_and_sandbox(self):
        for grid in ([], [[True]], [[1], [1, 2]], [[10]]):
            with self.assertRaises(ValueError): validate_grid(grid)
        with self.assertRaises(RuntimeError): DockerSandbox().run('print(1)', [[1]])

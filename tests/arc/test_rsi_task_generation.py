"""Tests for task-conditioned planning without model or sandbox calls."""
import unittest

from src.arc.data import Task
from src.arc.rsi_task_generation import generation_slots, task_prompt, task_repair_prompt


class TaskGenerationTests(unittest.TestCase):
    def test_task_prompt_contains_only_visible_examples_and_test_inputs(self):
        task = Task('task-1', (([[1, 2]], [[3, 4]]),), ([[5, 6]],))
        prompt = task_prompt(task)
        self.assertIn('[[1,2]]', prompt)
        self.assertIn('[[3,4]]', prompt)
        self.assertIn('[[5,6]]', prompt)
        self.assertNotIn('test output', prompt.lower())
        self.assertNotIn('task-1', prompt)

    def test_generation_uses_disjoint_sorted_tasks_and_registered_seeds(self):
        tasks = [f'task-{i:02}' for i in reversed(range(20))]
        first, second = generation_slots(tasks, 0), generation_slots(tasks, 1)
        self.assertEqual([item['task_id'] for item in first], sorted(tasks)[:10])
        self.assertEqual([item['task_id'] for item in second], sorted(tasks)[10:20])
        self.assertEqual(first[0]['attempt_seeds'], [12000, 12001, 12002])
        self.assertEqual(second[9]['attempt_seeds'], [12127, 12128, 12129])
        self.assertTrue(set(x['task_id'] for x in first).isdisjoint(x['task_id'] for x in second))

    def test_schedule_rejects_duplicate_short_or_invalid_manifests(self):
        with self.assertRaises(ValueError):
            generation_slots(['same'] * 20, 0)
        with self.assertRaises(ValueError):
            generation_slots(['a'], 0)
        with self.assertRaises(ValueError):
            generation_slots([f't{i}' for i in range(20)], 2)
        with self.assertRaises(ValueError):
            generation_slots([f't{i}' for i in range(20)], True)

    def test_repair_feedback_is_allowlisted_and_has_no_private_data(self):
        task = Task('task-1', (([[1]], [[2]]),), ([[3]],))
        prompt = task_repair_prompt(task, 'def transform(grid):\n    return grid', 'runtime_exception')
        self.assertIn('runtime_exception', prompt)
        self.assertIn('[[3]]', prompt)
        self.assertIn('<previous_candidate>', prompt)
        with self.assertRaises(ValueError):
            task_repair_prompt(task, 'code', 'hidden expected output [[9]]')
        with self.assertRaises(ValueError):
            task_repair_prompt(task, 'code', 'private_acceptance_rejected')
        for category in ('duplicate_source', 'duplicate_ast'):
            self.assertIn(category, task_repair_prompt(task, 'def transform(grid): return grid', category))
        with self.assertRaises(ValueError):
            task_repair_prompt(task, 'code', 'sampled_behavior_match')


if __name__ == '__main__':
    unittest.main()

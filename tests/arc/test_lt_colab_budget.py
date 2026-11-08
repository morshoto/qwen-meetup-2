import unittest
from src.arc.lt_colab_budget import Budget

class ColabBudgetTests(unittest.TestCase):
    def test_uses_smaller_of_time_and_estimated_unit_cap(self):
        b=Budget(minutes=10,units=1,rate_per_hour=12)
        self.assertEqual(b.limit_seconds,300)
        self.assertEqual(b.task_timeout(120,1800),180)
        self.assertIsNone(b.task_timeout(275,1800))

    def test_rejects_unbounded_or_invalid_values(self):
        for value in (0,-1,float('inf'),float('nan'),True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Budget(minutes=10,units=value,rate_per_hour=6)

    def test_caps_worker_timeout(self):
        b=Budget(minutes=60,units=100,rate_per_hour=1)
        self.assertEqual(b.task_timeout(0,1800),1800)

class LaunchGateTests(unittest.TestCase):
    def test_invalid_task_limit_does_not_create_journal(self):
        import json, tempfile
        from pathlib import Path
        from src.cli.lt_single import run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'manifest.json').write_text(json.dumps({'batch_size':10,'task_timeout_seconds':1800}))
            with self.assertRaisesRegex(ValueError,'batch limit'):
                run(root,max_tasks=11)
            self.assertFalse((root/'single').exists())
            self.assertFalse((root/'run.lock').exists())

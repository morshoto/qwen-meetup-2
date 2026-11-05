import tempfile
import unittest
from pathlib import Path

from src.arc.rsi_comparison_freeze import comparison_coverage, freeze_comparison, summarize_frozen_comparison

REG = 'a' * 64
CANDIDATE = 'b' * 64


class FrozenComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = [f'task-{i:03}' for i in range(200)]
        self.seeds = {task: [i * 2 + 10000, i * 2 + 10001] for i, task in enumerate(self.ids)}
        self.base = {task: {'solved': i < 20, 'seeds': self.seeds[task]} for i, task in enumerate(self.ids)}
        self.candidate = {task: {'solved': i < 50, 'seeds': self.seeds[task]} for i, task in enumerate(self.ids)}

    def freeze(self, **overrides):
        kwargs = dict(root=self.root, registration_sha256=REG, task_ids=self.ids, seeds=self.seeds,
                      parent_adapter='none', candidate_adapter=CANDIDATE, parent=self.base, candidate=self.candidate)
        kwargs.update(overrides)
        return freeze_comparison(**kwargs)

    def test_freeze_before_descriptive_summary(self):
        frozen = self.freeze()
        report = summarize_frozen_comparison(self.root, expected_sha256=frozen['sha256'])
        self.assertEqual((report['base_solved'], report['candidate_solved']), (20, 50))
        self.assertEqual(report['task_count'], 200)
        self.assertEqual(report['difference_percentage_points'], 15.0)
        self.assertFalse(report['independently_verified'])
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_missing_task_or_mismatched_seeds_cannot_freeze(self):
        with self.assertRaises(ValueError):
            self.freeze(candidate={k: v for k, v in list(self.candidate.items())[:-1]})
        changed = dict(self.candidate)
        changed[self.ids[0]] = {'solved': True, 'seeds': [1, 2]}
        with self.assertRaises(ValueError):
            self.freeze(candidate=changed)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_incomplete_comparison_reports_missing_tasks_without_scoring(self):
        candidate = dict(self.candidate)
        del candidate[self.ids[10]]
        parent = dict(self.base)
        del parent[self.ids[20]]
        report = comparison_coverage(self.ids, parent=parent, candidate=candidate)
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['missing_parent'], [self.ids[20]])
        self.assertEqual(report['missing_candidate'], [self.ids[10]])
        self.assertFalse(report['improvement_claim_supported'])
        self.assertNotIn('difference_percentage_points', report)

    def test_nonindependent_samples_and_invalid_identity_fail(self):
        seeds = dict(self.seeds)
        seeds[self.ids[0]] = [1, 1]
        with self.assertRaises(ValueError):
            self.freeze(seeds=seeds)
        with self.assertRaises(ValueError):
            self.freeze(candidate_adapter='none')
        with self.assertRaises(ValueError):
            self.freeze(parent_adapter=CANDIDATE)

    def test_tampering_or_wrong_hash_prevents_summary(self):
        frozen = self.freeze()
        with self.assertRaises(ValueError):
            summarize_frozen_comparison(self.root, expected_sha256='f' * 64)
        (self.root / 'comparison.frozen.json').write_text('{}')
        with self.assertRaises(ValueError):
            summarize_frozen_comparison(self.root, expected_sha256=frozen['sha256'])

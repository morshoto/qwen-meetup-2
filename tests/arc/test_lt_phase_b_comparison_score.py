import json
import unittest

from src.arc.data import Verifier
from src.arc.lt_phase_b_comparison_score import score_completed_comparison
from tests.arc import test_lt_phase_b_comparison_journal as fixtures


class ComparisonScoreTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ComparisonJournalTests('test_off_then_on_sequential_resume')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.verifier = Verifier({task: [[[1]]] for task in self.fixture.ids})
        def worker(task, state, output, on_start):
            result = self.fixture.worker(task, state, output, on_start)
            row = json.loads(output.read_text())
            row.update(tokens=10, runtime_seconds=1., peak_memory_bytes=100, peak_swap_bytes=0)
            output.write_text(json.dumps(row)); output.chmod(0o600)
            return result
        self.worker = worker

    def score(self, **kwargs):
        f = self.fixture
        args = {'off_journal': f.off, 'on_journal': f.on, 'registration': f.registration,
                'resource_limits': f.limits, 'verifier': self.verifier}
        args.update(kwargs)
        return score_completed_comparison(**args)

    def complete(self):
        self.fixture.run_batch(worker=self.worker)
        self.fixture.run_batch(worker=self.worker)

    def test_completed_replayed_journals_are_scored_offline(self):
        self.complete()
        report = self.score()
        self.assertEqual(report['status'], 'paired_score_consistent_unverified')
        self.assertFalse(report['evaluation_verified'])
        self.assertEqual(report['paired']['pass_at_1']['denominator'], 2)
        self.assertEqual(report['paired']['pass_at_1']['unchanged_right'], 2)
        self.assertEqual(report['states']['off']['tokens'], 20)
        self.assertNotIn('answers', str(report))

    def test_incomplete_journals_are_not_scored_or_mutated(self):
        self.fixture.run_batch(worker=self.worker)
        before = set(self.fixture.on.root.iterdir())
        with self.assertRaises(ValueError): self.score()
        self.assertEqual(set(self.fixture.on.root.iterdir()), before)

    def test_changed_result_and_registration_fail(self):
        self.complete()
        path = self.fixture.on.root / (self.fixture.ids[0] + '.result.json')
        original = path.read_text(); row = json.loads(original); row['candidates'] = []; path.write_text(json.dumps(row))
        with self.assertRaises(ValueError): self.score()
        path.write_text(original)
        reg = {**self.fixture.registration, 'adapter_sha256': '0' * 64}
        with self.assertRaises(ValueError): self.score(registration=reg)


if __name__ == '__main__': unittest.main()

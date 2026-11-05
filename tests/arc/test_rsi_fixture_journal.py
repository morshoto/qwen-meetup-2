"""Fixture slot journaling must not replay uncertain work or certify training."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.data import Task
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS
from src.arc.rsi_fixture_journal import run_fixture_journaled, verify_fixture_journal
from src.arc.rsi_slot_lifecycle import observe_slot


REG = 'a' * 64


class FixtureJournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = Task('fixture-parent', (([[1, 2]], [[2, 1]]),), ([[3, 4]],))
        self.public, self.private = _quality_grids(42, 5), _quality_grids(43, 5)

    def run_slot(self):
        return run_fixture_journaled(self.root, registration_sha256=REG, slot=0, task=self.task,
            replies=[FAKE_PROGRAMS[1]], public_probes=self.public, private_probes=self.private)

    def test_journal_binds_fixture_evidence_and_never_enables_training(self):
        result = self.run_slot()
        self.assertEqual(result['status'], 'fixture_completion_recorded')
        self.assertFalse(result['eligible_for_training'])
        evidence = self.root / 'g0-slot-00.fixture.json'
        self.assertEqual(hashlib.sha256(evidence.read_bytes()).hexdigest(), result['evidence_sha256'])
        verified = verify_fixture_journal(self.root, registration_sha256=REG, slot=0)
        self.assertEqual(verified['status'], 'fixture_integrity_only')
        self.assertFalse(verified['eligible_for_training'])
        self.assertEqual(json.loads(evidence.read_text())['draft']['provenance']['execution_mode'], 'fake_fixture')
        with patch('src.arc.rsi_fixture_journal.run_fixture_slot', side_effect=AssertionError('replayed')):
            with self.assertRaises(FileExistsError):
                self.run_slot()

    def test_interruption_after_claim_never_replays(self):
        with patch('src.arc.rsi_fixture_journal.run_fixture_slot', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.run_slot()
        self.assertEqual(observe_slot(self.root, registration_sha256=REG, generation=0, slot=0)['status'],
                         'started_outcome_uncertain')
        with self.assertRaises(FileExistsError):
            self.run_slot()

    def test_evidence_tampering_is_detected(self):
        self.run_slot()
        (self.root / 'g0-slot-00.fixture.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'evidence'):
            verify_fixture_journal(self.root, registration_sha256=REG, slot=0)


if __name__ == '__main__':
    unittest.main()

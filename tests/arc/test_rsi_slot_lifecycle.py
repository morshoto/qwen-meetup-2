"""A claimed generation slot cannot be silently replayed after interruption."""
import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_slot_lifecycle import claim_slot, complete_slot, observe_slot


REG = 'a' * 64


class SlotLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_claim_is_durable_private_and_uncertain_until_completed(self):
        claim = claim_slot(self.root, registration_sha256=REG, generation=0, slot=0)
        self.assertEqual(stat.S_IMODE((self.root / 'g0-slot-00.started.json').stat().st_mode), 0o600)
        self.assertEqual(observe_slot(self.root, registration_sha256=REG, generation=0, slot=0)['status'],
                         'started_outcome_uncertain')
        with self.assertRaises(FileExistsError):
            claim_slot(self.root, registration_sha256=REG, generation=0, slot=0)
        complete_slot(self.root, registration_sha256=REG, generation=0, slot=0,
                      claim_token=claim['claim_token'], evidence_sha256='b' * 64)
        result = observe_slot(self.root, registration_sha256=REG, generation=0, slot=0)
        self.assertEqual(result['status'], 'completion_recorded_unverified')
        self.assertFalse(result['eligible_for_training'])
        with self.assertRaises(FileExistsError):
            complete_slot(self.root, registration_sha256=REG, generation=0, slot=0,
                          claim_token=claim['claim_token'], evidence_sha256='b' * 64)

    def test_wrong_registration_token_or_malformed_start_cannot_complete(self):
        claim = claim_slot(self.root, registration_sha256=REG, generation=0, slot=1)
        with self.assertRaises(ValueError):
            complete_slot(self.root, registration_sha256=REG, generation=0, slot=1,
                          claim_token='wrong', evidence_sha256='b' * 64)
        with self.assertRaises(ValueError):
            observe_slot(self.root, registration_sha256='c' * 64, generation=0, slot=1)
        (self.root / 'g0-slot-01.started.json').write_text('{')
        with self.assertRaises(ValueError):
            complete_slot(self.root, registration_sha256=REG, generation=0, slot=1,
                          claim_token=claim['claim_token'], evidence_sha256='b' * 64)
        with self.assertRaises(FileExistsError):
            claim_slot(self.root, registration_sha256=REG, generation=0, slot=1)

    def test_concurrent_claims_have_exactly_one_winner(self):
        barrier = threading.Barrier(8)
        winners, failures = [], []
        def run():
            barrier.wait()
            try:
                winners.append(claim_slot(self.root, registration_sha256=REG, generation=1, slot=2))
            except FileExistsError:
                failures.append(True)
        workers = [threading.Thread(target=run) for _ in range(8)]
        for worker in workers: worker.start()
        for worker in workers: worker.join()
        self.assertEqual((len(winners), len(failures)), (1, 7))

    def test_rejects_unregistered_slot_and_symlink_marker(self):
        with self.assertRaises(ValueError):
            claim_slot(self.root, registration_sha256=REG, generation=0, slot=10)
        target = self.root / 'target'
        target.write_text('unchanged')
        (self.root / 'g0-slot-00.started.json').symlink_to(target)
        with self.assertRaises((ValueError, FileExistsError, OSError)):
            claim_slot(self.root, registration_sha256=REG, generation=0, slot=0)
        self.assertEqual(target.read_text(), 'unchanged')

    def test_completion_marker_without_start_fails_closed(self):
        (self.root / 'g0-slot-00.completed.json').write_text(json.dumps({'fake': True}))
        with self.assertRaises(ValueError):
            observe_slot(self.root, registration_sha256=REG, generation=0, slot=0)
        with self.assertRaises(ValueError):
            claim_slot(self.root, registration_sha256=REG, generation=0, slot=0)

    def test_failed_marker_sync_still_consumes_slot(self):
        with patch('src.arc.rsi_slot_lifecycle.os.fsync', side_effect=OSError('sync failed')):
            with self.assertRaisesRegex(OSError, 'sync failed'):
                claim_slot(self.root, registration_sha256=REG, generation=0, slot=0)
        with self.assertRaises(FileExistsError):
            claim_slot(self.root, registration_sha256=REG, generation=0, slot=0)
        self.assertEqual(observe_slot(self.root, registration_sha256=REG, generation=0, slot=0)['status'],
                         'started_outcome_uncertain')


if __name__ == '__main__':
    unittest.main()

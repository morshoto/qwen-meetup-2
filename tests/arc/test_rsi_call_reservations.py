import concurrent.futures
import tempfile
import unittest
from pathlib import Path

from src.arc.rsi_call_reservations import reserve_call
from src.arc.rsi_slot_lifecycle import claim_slot


class CallReservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registration = 'a' * 64
        self.claim = claim_slot(self.root, registration_sha256=self.registration, generation=0, slot=0)

    def tearDown(self):
        self.tmp.cleanup()

    def reserve(self, attempt=0, token=None):
        return reserve_call(self.root, registration_sha256=self.registration, generation=0,
                            slot=0, attempt=attempt, claim_token=token or self.claim['claim_token'])

    def test_reservation_is_durable_and_non_reusable(self):
        self.assertEqual(self.reserve()['status'], 'call_reserved_unverified')
        with self.assertRaises(FileExistsError):
            self.reserve()

    def test_requires_valid_claim_and_order(self):
        with self.assertRaises(ValueError):
            self.reserve(token='b' * 64)
        with self.assertRaises(ValueError):
            self.reserve(attempt=1)
        self.reserve()
        self.assertEqual(self.reserve(attempt=1)['attempt'], 1)
        with self.assertRaises(ValueError):
            self.reserve(attempt=3)

    def test_competing_reservations_have_one_winner(self):
        def run(_):
            try:
                self.reserve()
                return True
            except FileExistsError:
                return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(run, range(8))), 1)

    def test_stage_has_at_most_32_calls(self):
        for slot in range(10):
            claim = self.claim if slot == 0 else claim_slot(self.root, registration_sha256=self.registration, generation=0, slot=slot)
            for attempt in range(3):
                reserve_call(self.root, registration_sha256=self.registration, generation=0, slot=slot,
                             attempt=attempt, claim_token=claim['claim_token'])
        # A generation has only 30 scheduled attempts; forged slot numbers cannot exceed it.
        with self.assertRaises(ValueError):
            reserve_call(self.root, registration_sha256=self.registration, generation=0, slot=10,
                         attempt=0, claim_token=self.claim['claim_token'])

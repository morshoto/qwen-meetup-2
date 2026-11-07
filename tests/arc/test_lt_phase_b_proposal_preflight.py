import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.arc.lt_phase_b_proposal import build_proposal

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/Users/shotomorisaki/Engineering/qwen-meetup-2')


class ProposalTests(unittest.TestCase):
    def test_rejects_unbound_challenge_data(self):
        with self.assertRaises(ValueError):
            build_proposal(ROOT, b'{}', b'{}')

    @unittest.skipUnless(os.environ.get('LT_PHASE_B_ALLOCATION_PREFLIGHT') == '1', 'local data preflight not requested')
    def test_live_proposal_replays_and_rejects_changed_official_inventory(self):
        challenges = (BASE / 'data/raw/arc-agi_training_challenges.json').read_bytes()
        with zipfile.ZipFile(BASE / 'arc-prize-2025.zip') as archive:
            official = archive.read('arc-agi_evaluation_challenges.json')
        result = build_proposal(ROOT, challenges, official)
        self.assertEqual(result, json.loads((ROOT / 'configs/lt.phase-b.comparison-proposal.json').read_bytes()))
        self.assertIn('proposal_code', result['source_hashes'])
        with self.assertRaises(ValueError):
            build_proposal(ROOT, challenges, b'{}')

    @unittest.skipUnless(os.environ.get('LT_PHASE_B_ALLOCATION_PREFLIGHT') == '1', 'local data preflight not requested')
    def test_ephemeral_pool_change_preserves_frozen_phase_a(self):
        from src.arc.lt_phase_b_proposal import _PATHS
        from tests.arc.test_lt_phase_b_allocation_receipt import AllocationReceiptTests
        fixture = AllocationReceiptTests('test_eligible_fixture_selection_produces_bound_receipt'); fixture.setUp()
        historical_path = Path('configs/lt.phase-a.comparison-proposal.json')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = set(_PATHS.values()) | {str(historical_path)}
            for name in paths:
                dest = root / name; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, dest)
            before = (root / historical_path).read_bytes()
            (root / _PATHS['training']).write_text(json.dumps(fixture.fixture_pool()))
            result = build_proposal(root, fixture.challenges, fixture.official)
            self.assertEqual(len(result['task_ids']), 50)
            self.assertEqual((root / historical_path).read_bytes(), before)
            self.assertEqual(result['scheduled_training_overlap'], [])
            historical = json.loads(before); historical['training_exclusions'] = historical['training_exclusions'][1:]
            (root / historical_path).write_text(json.dumps(historical))
            with self.assertRaises(ValueError): build_proposal(root, fixture.challenges, fixture.official)


if __name__ == '__main__': unittest.main()

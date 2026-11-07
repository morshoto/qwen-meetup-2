import json
import os
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


if __name__ == '__main__': unittest.main()

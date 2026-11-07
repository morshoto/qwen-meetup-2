"""Opt-in public inventory check; does not load a model or solution answers."""
import json
import os
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('LT_PHASE_B_ALLOCATION_PREFLIGHT') == '1', 'local allocation preflight not requested')
class RegistrationPreflightTests(unittest.TestCase):
    def test_current_pool_is_refused_before_model_or_runtime_probe(self):
        from src.arc.lt_phase_b_registration_proposal import propose_collection_registration
        root = Path(__file__).resolve().parents[2]
        main = Path('/Users/shotomorisaki/Engineering/qwen-meetup-2')
        challenge = (main / 'data/raw/arc-agi_training_challenges.json').read_bytes()
        with zipfile.ZipFile(main / 'arc-prize-2025.zip') as archive:
            official = archive.read('arc-agi_evaluation_challenges.json')
        pool = json.loads((root / 'configs/lt.phase-b.training-pool.json').read_text())
        with patch('src.arc.lt_phase_b_registration_proposal.snapshot_model_manifest') as model, \
             patch('src.arc.lt_phase_b_registration_proposal.snapshot_runtime_identity') as runtime:
            with self.assertRaises(ValueError):
                propose_collection_registration(source_root=root, challenge_bytes=challenge,
                    official_challenge_bytes=official, training_task_ids=pool, run_id='unfrozen-check',
                    python_executable='/unused', model_path='/unused')
            model.assert_not_called(); runtime.assert_not_called()


if __name__ == '__main__': unittest.main()

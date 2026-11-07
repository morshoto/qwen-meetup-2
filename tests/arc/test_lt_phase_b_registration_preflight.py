"""Opt-in public inventory check; does not load a model or solution answers."""
import json
import os
import tempfile
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
            with self.assertRaisesRegex(ValueError, 'training task intersects exclusion'):
                propose_collection_registration(source_root=root, challenge_bytes=challenge,
                    official_challenge_bytes=official, training_task_ids=pool, run_id='unfrozen-check',
                    python_executable='/unused', model_path='/unused')
            model.assert_not_called(); runtime.assert_not_called()

    def test_ephemeral_fixture_builds_draft_with_installed_identities(self):
        from src.arc.lt_phase_b_registration_proposal import propose_collection_registration
        from src.arc.lt_phase_b_registered_worker import _digest, validate_descriptor
        from tests.arc.test_lt_phase_b_allocation_receipt import AllocationReceiptTests
        root = Path(__file__).resolve().parents[2]
        fixture = AllocationReceiptTests('test_eligible_fixture_selection_produces_bound_receipt')
        fixture.setUp()
        result = propose_collection_registration(source_root=root, challenge_bytes=fixture.challenges,
            official_challenge_bytes=fixture.official, training_task_ids=fixture.fixture_pool(),
            run_id='ephemeral-fixture-not-a-run', python_executable='/tmp/qwen-p3a-mlx-venv/bin/python',
            model_path=Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4')
        self.assertEqual(result['status'], 'collection_registration_proposed_unverified')
        self.assertEqual(result['spec']['status'], 'draft')
        self.assertFalse(result['execution_ready'])
        self.assertEqual(len(result['export']['tasks']), 100)
        self.assertTrue(all(set(pair) == {'input'} for task in result['export']['tasks'].values() for pair in task['test']))
        self.assertEqual(result['spec_sha256'], _digest(result['spec']))
        with self.assertRaises(ValueError):
            validate_descriptor(spec=result['spec'], expected_run_sha256=result['spec_sha256'],
                source_inventory=result['inventory'], receipt=result['receipt'], allocation_receipt=result['allocation'])

    def test_ephemeral_canary_proposal_binds_installed_identities(self):
        from src.arc.lt_phase_b_canary_proposal import propose_canary_registration
        from src.arc.lt_phase_b_registered_worker import _digest
        from src.arc.lt_phase_b_registration_proposal import propose_collection_registration
        from tests.arc.test_lt_phase_b_allocation_receipt import AllocationReceiptTests
        root = Path(__file__).resolve().parents[2]
        fixture = AllocationReceiptTests('test_eligible_fixture_selection_produces_bound_receipt'); fixture.setUp()
        python = '/tmp/qwen-p3a-mlx-venv/bin/python'
        model = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
        proposal = propose_collection_registration(source_root=root, challenge_bytes=fixture.challenges,
            official_challenge_bytes=fixture.official, training_task_ids=fixture.fixture_pool(),
            run_id='ephemeral-fixture-not-a-run', python_executable=python, model_path=model)
        # Only an in-memory synthetic fixture uses this hypothetical frozen form.
        collection = {**proposal['spec'], 'status': 'frozen'}
        collection_sha = _digest(collection)
        receipt = {'comparison_sha256': collection_sha, 'selected_task_sha256': collection['selected_task_sha256']}
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'data'; data.mkdir(mode=0o700)
            result = propose_canary_registration(collection_spec=collection, expected_collection_sha256=collection_sha,
                source_inventory=proposal['inventory'], source_root=root, allocation_receipt=proposal['allocation'],
                receipt=receipt, model_dir=model, data_dir=data, adapter_dir=Path(tmp) / 'adapter',
                python_executable=python, canary_run_id='synthetic-fixture')
            self.assertEqual(result['spec']['status'], 'draft')
            self.assertFalse(result['execution_ready']); self.assertFalse(result['promotion_allowed'])
            self.assertEqual(result['trainer_config']['iters'], 1)
            self.assertEqual(list(data.iterdir()), [])


if __name__ == '__main__': unittest.main()

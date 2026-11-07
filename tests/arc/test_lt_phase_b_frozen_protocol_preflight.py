"""Opt-in temporary frozen-protocol fixture; never launches or chooses a real pool."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get('LT_PHASE_B_ALLOCATION_PREFLIGHT') == '1', 'local allocation preflight not requested')
class FrozenProtocolPreflightTests(unittest.TestCase):
    def test_temporary_frozen_fixture_passes_real_preflight(self):
        from src.arc.lt_phase_b_allocation_receipt import derive_allocation
        from src.arc.lt_phase_b_collection_preflight import preflight_collection
        from src.arc.lt_phase_b_proposal import build_proposal
        from src.arc.lt_phase_b_registered_worker import _SETTINGS, _digest
        from src.arc.lt_public import build_public_export
        from src.arc.lt_source import capture_source_tree
        from src.arc.rsi_environment_identity import snapshot_environment
        from src.arc.rsi_runtime_identity import snapshot_runtime_identity
        from tests.arc.test_lt_phase_b_allocation_receipt import AllocationReceiptTests
        root = Path(__file__).resolve().parents[2]
        fixture = AllocationReceiptTests('test_eligible_fixture_selection_produces_bound_receipt'); fixture.setUp()
        python = '/tmp/qwen-p3a-mlx-venv/bin/python'
        model = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp)
            for name in ('src', 'sandbox', 'configs', 'scripts', 'research/rsi'):
                shutil.copytree(root / name, copy / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            for name in ('pyproject.toml', 'uv.lock'): shutil.copy2(root / name, copy / name)
            historical = (copy / 'configs/lt.phase-a.comparison-proposal.json').read_bytes()
            pool = fixture.fixture_pool()
            (copy / 'configs/lt.phase-b.training-pool.json').write_text(json.dumps(pool))
            comparison = build_proposal(copy, fixture.challenges, fixture.official)
            (copy / 'configs/lt.phase-b.comparison-proposal.json').write_text(json.dumps(comparison, sort_keys=True))
            allocation = derive_allocation(copy, fixture.challenges, fixture.official, pool)
            protocol_path = copy / 'configs/lt.phase-b.protocol.json'
            protocol = json.loads(protocol_path.read_text())
            protocol['status'] = 'frozen'
            protocol['data']['training_pool_status'] = 'frozen'
            protocol['evaluation']['comparison_status'] = 'frozen'
            protocol['open_gates'] = ['real_14b_canary_and_reload']
            protocol['registration'].update(actual_evaluation_manifest='frozen', actual_training_settings='frozen',
                allocation=_digest(allocation), source_closure='bound_by_collection_descriptor',
                runtime_and_environment='bound_by_collection_descriptor', protocol_hash='bound_by_source_inventory')
            protocol_path.write_text(json.dumps(protocol, sort_keys=True))
            inventory = capture_source_tree(copy)
            spec = {'protocol': 'lt-phase-b-collection-v1', 'status': 'frozen', 'run_id': 'temporary-fixture-only',
                'source_code_sha256': inventory['code_sha256'], 'allocation_audit_sha256': _digest(allocation),
                'prompt_sha256': protocol['inference']['prompt_sha256'], 'selected_task_sha256': allocation['training_task_sha256'],
                'runtime_manifest': snapshot_runtime_identity(python), 'environment_manifest': snapshot_environment(python),
                'model_manifest': protocol['model'], 'settings': dict(_SETTINGS)}
            receipt = {'comparison_sha256': _digest(spec), 'selected_task_sha256': spec['selected_task_sha256']}
            export = build_public_export(json.loads(fixture.challenges), receipt)
            result = preflight_collection(source_root=copy, spec=spec, expected_run_sha256=_digest(spec),
                source_inventory=inventory, allocation_receipt=allocation, export=export, receipt=receipt,
                challenge_bytes=fixture.challenges, official_challenge_bytes=fixture.official,
                python_executable=python, model_path=model)
            self.assertEqual(result['selected_tasks'], 100)
            self.assertFalse(result['execution_ready'])
            self.assertEqual((copy / 'configs/lt.phase-a.comparison-proposal.json').read_bytes(), historical)
            self.assertEqual(json.loads((root / 'configs/lt.phase-b.protocol.json').read_text())['status'], 'draft_not_frozen')


if __name__ == '__main__': unittest.main()

import json
import os
import unittest
import zipfile
from pathlib import Path

from src.arc.lt_phase_b_allocation_receipt import derive_allocation
from src.arc.rsi_task_generation import generation_slots

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/Users/shotomorisaki/Engineering/qwen-meetup-2')


@unittest.skipUnless(os.environ.get('LT_PHASE_B_ALLOCATION_PREFLIGHT') == '1', 'local data preflight not requested')
class AllocationReceiptTests(unittest.TestCase):
    def setUp(self):
        self.challenges = (BASE / 'data/raw/arc-agi_training_challenges.json').read_bytes()
        with zipfile.ZipFile(BASE / 'arc-prize-2025.zip') as archive:
            self.official = archive.read('arc-agi_evaluation_challenges.json')
        self.original = json.loads((ROOT / 'configs/lt.phase-b.training-pool.json').read_bytes())
        self.registry = json.loads((ROOT / 'research/rsi/bounded-exposure-proposal.json').read_bytes())
        self.split = json.loads((ROOT / 'research/rsi/exploratory-split-proposal.json').read_bytes())
        self.phase_a = json.loads((ROOT / 'configs/lt.phase-a.frozen.json').read_bytes())
        self.dev = json.loads((ROOT / 'configs/lt.phase-a.development.json').read_bytes())
        self.comparison = json.loads((ROOT / 'configs/lt.phase-b.comparison-proposal.json').read_bytes())
        self.scheduled = {x['task_id'] for i in (0, 1) for x in generation_slots(self.registry['partitions']['generation'], i)}

    def fixture_pool(self):
        unavailable = (set(self.original) | set(self.phase_a['task_ids']) | set(self.dev['task_ids']) |
            set(self.split['confirmatory_task_ids']) | set(self.split['exploratory_task_ids']) |
            self.scheduled | set(self.registry['known_exposed_task_ids']) |
            set(self.comparison['task_ids']) | set(json.loads(self.official)))
        candidates = sorted(set(json.loads(self.challenges)) - unavailable)
        overlap = set(self.original) & self.scheduled
        fixture = sorted((set(self.original) - overlap) | set(candidates[:len(overlap)]))
        return fixture

    def test_preallocated_pool_refuses_scheduled_slot_overlap(self):
        with self.assertRaises(ValueError):
            derive_allocation(ROOT, self.challenges, self.official, self.original)

    def test_eligible_fixture_selection_produces_bound_receipt(self):
        # This local test does not choose or record an actual replacement pool.
        fixture = self.fixture_pool()
        receipt = derive_allocation(ROOT, self.challenges, self.official, fixture)
        self.assertEqual(receipt['status'], 'allocation_consistent_unverified')
        self.assertEqual(len(receipt['training_task_sha256']), 100)
        self.assertEqual(len(receipt['comparison_task_sha256']), 50)
        self.assertFalse(set(receipt['training_task_sha256']) & set(receipt['comparison_task_sha256']))
        self.assertFalse(receipt['execution_ready'])
        with self.assertRaises(ValueError):
            derive_allocation(ROOT, self.challenges, self.official, fixture[:-1])


    def test_local_fixture_descriptor_preflight_without_model(self):
        import hashlib
        from src.arc.lt_live_backend import PROMPT
        from src.arc.lt_phase_b_collection_preflight import preflight_collection
        from src.arc.lt_phase_b_registered_worker import _SETTINGS, _digest
        from src.arc.lt_public import build_public_export
        from src.arc.lt_source import capture_source_tree
        from src.arc.rsi_runtime_identity import snapshot_runtime_identity
        from src.arc.rsi_environment_identity import snapshot_environment
        tasks = self.fixture_pool()
        allocation = derive_allocation(ROOT, self.challenges, self.official, tasks)
        inventory = capture_source_tree(ROOT)
        manifest = json.loads((ROOT / 'research/rsi/audits/technical-model-14b.json').read_bytes())['model']
        spec = {'protocol': 'lt-phase-b-collection-v1', 'status': 'frozen', 'run_id': 'fixture-not-a-run',
                'source_code_sha256': inventory['code_sha256'], 'allocation_audit_sha256': _digest(allocation),
                'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                'selected_task_sha256': allocation['training_task_sha256'],
                'runtime_manifest': snapshot_runtime_identity('/tmp/qwen-p3a-mlx-venv/bin/python'),
                'environment_manifest': snapshot_environment('/tmp/qwen-p3a-mlx-venv/bin/python'),
                'model_manifest': manifest, 'settings': dict(_SETTINGS)}
        receipt = {'comparison_sha256': _digest(spec), 'selected_task_sha256': spec['selected_task_sha256']}
        export = build_public_export(json.loads(self.challenges), receipt)
        result = preflight_collection(source_root=ROOT, spec=spec, expected_run_sha256=_digest(spec),
            source_inventory=inventory, allocation_receipt=allocation, export=export, receipt=receipt,
            challenge_bytes=self.challenges, official_challenge_bytes=self.official)
        self.assertEqual(result['selected_tasks'], 100)
        self.assertFalse(result['execution_ready'])
        self.assertTrue(all(set(pair) == {'input'} for item in export['tasks'].values() for pair in item['test']))


if __name__ == '__main__': unittest.main()

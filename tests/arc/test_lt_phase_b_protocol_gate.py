import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_phase_b_protocol_gate import validate_frozen_protocol
from src.arc.lt_phase_b_registered_worker import _digest
from tests.arc import test_lt_phase_b_registered_worker as fixtures


class FrozenProtocolGateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); (self.root / 'configs').mkdir()
        self.fixture = fixtures.RegisteredWorkerTests('test_frozen_public_worker_binds_identity_before_and_after')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.challenge = json.dumps(f.challenges).encode()
        source = Path(__file__).resolve().parents[2] / 'configs/lt.phase-b.protocol.json'
        self.protocol = json.loads(source.read_text())
        self.protocol['status'] = 'frozen'
        self.protocol['data']['training_pool_status'] = 'frozen'
        self.protocol['data']['challenge_sha256'] = hashlib.sha256(self.challenge).hexdigest()
        self.protocol['evaluation']['comparison_status'] = 'frozen'
        self.protocol['model'] = copy.deepcopy(f.spec['model_manifest'])
        self.protocol['open_gates'] = ['real_14b_canary_and_reload']
        self.protocol['registration'].update(actual_evaluation_manifest='frozen', actual_training_settings='frozen',
            allocation=_digest(f.allocation), source_closure=f.inventory['code_sha256'],
            runtime_and_environment='bound_by_collection_descriptor', protocol_hash='bound_by_source_inventory')
        (self.root / 'configs/lt.phase-b.training-pool.json').write_text(json.dumps(sorted(f.hashes)))
        (self.root / 'configs/lt.phase-b.comparison-proposal.json').write_text(json.dumps({
            'task_ids': sorted(f.allocation['comparison_task_sha256'])}))
        self.save()
    def save(self):
        (self.root / 'configs/lt.phase-b.protocol.json').write_text(json.dumps(self.protocol))
    def validate(self):
        return validate_frozen_protocol(source_root=self.root, spec=self.fixture.spec,
            source_inventory=self.fixture.inventory, allocation_receipt=self.fixture.allocation, challenge_bytes=self.challenge)

    def test_frozen_bound_protocol_passes_without_authorizing_execution(self):
        result = self.validate()
        self.assertEqual(result['status'], 'frozen_protocol_consistent_unverified')
        self.assertFalse(result['execution_ready'])

    def test_draft_or_open_registration_gates_refuse(self):
        for field, value in (('status', 'draft_not_frozen'), ('open_gates', ['scheduled_generation_slot_overlap']),
                             ('open_gates', ['unknown_gate'])):
            old = copy.deepcopy(self.protocol); self.protocol[field] = value; self.save()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): self.validate()
            self.protocol = old
        self.protocol['registration']['actual_training_settings'] = 'not_frozen'; self.save()
        with self.assertRaises(ValueError): self.validate()

    def test_model_settings_or_task_inventory_mismatch_refuse(self):
        for mutate in (lambda p: p['inference'].__setitem__('seed_base', 1),
                       lambda p: p['training'].__setitem__('rank', 4),
                       lambda p: p['training'].__setitem__('selection', 'best validation score'),
                       lambda p: p['evaluation'].__setitem__('metrics', []),
                       lambda p: p['canary'].__setitem__('must_record', []),
                       lambda p: p['model'].__setitem__('revision', 'wrong'),
                       lambda p: p['registration'].__setitem__('allocation', '0'*64),
                       lambda p: p['data'].__setitem__('training_pool_status', 'conflict_unresolved')):
            old = copy.deepcopy(self.protocol); mutate(self.protocol); self.save()
            with self.assertRaises(ValueError): self.validate()
            self.protocol = old
        self.save()
        (self.root / 'configs/lt.phase-b.training-pool.json').write_text('[]')
        with self.assertRaises(ValueError): self.validate()


if __name__ == '__main__': unittest.main()

import json
import unittest
from unittest.mock import patch

from src.arc.rsi_model_identity import snapshot_model_manifest
from src.arc.rsi_registered_model import verify_registered_model
from src.arc.rsi_registration import prepare_draft_registration
import tests.arc.test_rsi_model_identity as model_fixtures
import tests.arc.test_rsi_registration as registration_fixtures


class RegisteredModelTests(unittest.TestCase):
    def setUp(self):
        self.reg_fixture = registration_fixtures.DraftRegistrationTests('test_draft_binds_inputs_schedule_and_remains_non_authorizing')
        self.reg_fixture.setUp()
        self.addCleanup(self.reg_fixture.doCleanups)
        self.model_fixture = model_fixtures.ModelIdentityTests('test_manifest_binds_every_file_and_template')
        self.model_fixture.setUp()
        self.addCleanup(self.model_fixture.doCleanups)
        self.root = self.reg_fixture.root
        self.model = self.model_fixture.model
        self.manifest_path = self.root / self.reg_fixture.paths['model_manifest']
        self.manifest_path.write_text(json.dumps(snapshot_model_manifest(self.model)))
        self.registration = prepare_draft_registration(self.root, input_paths=self.reg_fixture.paths, generation=0)

    def verify(self):
        return verify_registered_model(self.root, registration=self.registration, model_dir=self.model)

    def test_matching_files_are_consistent_but_draft_remains_blocked(self):
        result = self.verify()
        self.assertEqual(result['status'], 'draft_model_files_consistent')
        self.assertFalse(result['execution_ready'])
        self.assertEqual(result['model_manifest_sha256'], self.registration['input_sha256']['model_manifest'])

    def test_model_or_registered_manifest_drift_is_rejected(self):
        (self.model / 'one.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.verify()
        (self.model / 'one.safetensors').write_bytes(b'fixture one')
        self.manifest_path.write_text('{}')
        with self.assertRaises(ValueError):
            self.verify()

    def test_registration_is_rechecked_after_model_hashing(self):
        def mutate(*args, **kwargs):
            self.manifest_path.write_text('{}')
        with patch('src.arc.rsi_registered_model.verify_model_manifest', side_effect=mutate):
            with self.assertRaises(ValueError):
                self.verify()

import json
import unittest
from unittest.mock import patch

from src.arc.rsi_registered_environment import verify_registered_environment
from src.arc.rsi_registration import prepare_draft_registration
import tests.arc.test_rsi_environment_identity as environment_fixtures
import tests.arc.test_rsi_registration as registration_fixtures
import tests.arc.test_rsi_runtime_identity as runtime_fixtures


class RegisteredEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.reg = registration_fixtures.DraftRegistrationTests('test_draft_binds_inputs_schedule_and_remains_non_authorizing')
        self.reg.setUp()
        self.addCleanup(self.reg.doCleanups)
        self.env = environment_fixtures.EnvironmentIdentityTests('test_inventory_includes_unrecorded_files_and_detects_drift')
        self.env.setUp()
        self.addCleanup(self.env.doCleanups)
        from src.cli.rsi_environment_probe import inventory
        self.inventory = inventory(self.env.root, self.env.python)
        self.runtime = {'interpreter_realpath': self.inventory['interpreter_realpath'],
                        'interpreter_sha256': self.inventory['interpreter_sha256']}
        self.path = self.reg.root / self.reg.paths['runtime_manifest']
        self.path.write_text(json.dumps({'runtime_identity': self.runtime, 'environment_inventory': self.inventory}))
        self.registration = prepare_draft_registration(self.reg.root, input_paths=self.reg.paths, generation=0)

    @patch('src.arc.rsi_registered_environment.verify_environment')
    def test_consistent_inventory_is_still_not_execution_ready(self, verify):
        result = verify_registered_environment(self.reg.root, registration=self.registration,
                                               python_executable=self.env.python)
        verify.assert_called_once_with(self.env.python, self.inventory)
        self.assertEqual(result['status'], 'draft_environment_consistent')
        self.assertFalse(result['execution_ready'])

    @patch('src.arc.rsi_registered_environment.verify_environment')
    def test_changed_registration_and_interpreter_mismatch_fail(self, verify):
        self.path.write_text('{}')
        with self.assertRaises(ValueError):
            verify_registered_environment(self.reg.root, registration=self.registration, python_executable=self.env.python)
        verify.assert_not_called()

    @patch('src.arc.rsi_registered_environment.verify_environment')
    def test_registration_is_rechecked_after_inventory(self, verify):
        verify.side_effect = lambda *args: self.path.write_text('{}')
        with self.assertRaises(ValueError):
            verify_registered_environment(self.reg.root, registration=self.registration, python_executable=self.env.python)

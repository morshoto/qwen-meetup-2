import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.arc.rsi_registered_runtime import verify_registered_runtime
from src.arc.rsi_registration import prepare_draft_registration
import tests.arc.test_rsi_registration as registration_fixtures
import tests.arc.test_rsi_runtime_identity as runtime_fixtures


class RegisteredRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.reg = registration_fixtures.DraftRegistrationTests('test_draft_binds_inputs_schedule_and_remains_non_authorizing')
        self.reg.setUp()
        self.addCleanup(self.reg.doCleanups)
        self.runtime = runtime_fixtures.RuntimeIdentityTests('test_reviewed_runtime_is_an_observation_not_authorization')
        self.runtime.setUp()
        self.addCleanup(self.runtime.doCleanups)
        self.path = self.reg.root / self.reg.paths['runtime_manifest']
        self.path.write_text(json.dumps(self.runtime.report))
        self.registration = prepare_draft_registration(self.reg.root, input_paths=self.reg.paths, generation=0)

    def verify(self):
        return verify_registered_runtime(self.reg.root, registration=self.registration,
                                         python_executable=self.runtime.python)

    def test_registered_runtime_is_consistent_but_not_execution_ready(self):
        result = SimpleNamespace(returncode=0, stdout=json.dumps(self.runtime.report), stderr='')
        with patch('src.arc.rsi_runtime_identity.subprocess.run', return_value=result):
            checked = self.verify()
        self.assertEqual(checked['status'], 'draft_runtime_consistent')
        self.assertFalse(checked['execution_ready'])
        self.assertEqual(checked['runtime_manifest_sha256'], self.registration['input_sha256']['runtime_manifest'])

    def test_changed_runtime_or_registration_fails(self):
        changed = dict(self.runtime.report, generate_step_sha256='b' * 64)
        with patch('src.arc.rsi_runtime_identity.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(changed), stderr='')):
            with self.assertRaises(ValueError):
                self.verify()
        self.path.write_text('{}')
        with self.assertRaises(ValueError):
            self.verify()

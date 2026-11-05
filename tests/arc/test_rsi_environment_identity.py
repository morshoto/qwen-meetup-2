import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.arc.rsi_environment_identity import snapshot_environment, verify_environment
from src.cli.rsi_environment_probe import inventory


class EnvironmentIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'venv'
        (self.root / 'bin').mkdir(parents=True)
        self.python = self.root / 'bin' / 'python'
        self.python.write_bytes(b'fixture interpreter')
        self.python.chmod(0o700)
        (self.root / 'module.py').write_text('x = 1')
        (self.root / 'unrecorded.pth').write_text('fixture')

    def test_inventory_includes_unrecorded_files_and_detects_drift(self):
        first = inventory(self.root, self.python)
        self.assertEqual(first['file_count'], 3)
        (self.root / 'unrecorded.pth').write_text('changed')
        second = inventory(self.root, self.python)
        self.assertNotEqual(first['inventory_sha256'], second['inventory_sha256'])
        (self.root / 'new.pyc').write_bytes(b'bytecode')
        self.assertEqual(inventory(self.root, self.python)['file_count'], 4)

    def test_external_symlinks_are_rejected_except_interpreter(self):
        external = Path(self.tmp.name) / 'outside'
        external.write_bytes(b'outside')
        (self.root / 'escape').symlink_to(external)
        with self.assertRaises(ValueError):
            inventory(self.root, self.python)
        (self.root / 'escape').unlink()
        (self.root / 'alias').symlink_to(self.python)
        self.assertEqual(inventory(self.root, self.python)['file_count'], 4)

    def test_snapshot_checks_interpreter_and_manifest_without_authorizing(self):
        observed = inventory(self.root, self.python)
        response = SimpleNamespace(returncode=0, stdout=json.dumps(observed), stderr='')
        with patch('src.arc.rsi_environment_identity.subprocess.run', return_value=response) as run:
            saved = snapshot_environment(self.python)
            self.assertIn('-I', run.call_args.args[0])
            self.assertIn('-B', run.call_args.args[0])
            self.assertFalse(verify_environment(self.python, saved)['execution_ready'])
            with self.assertRaises(ValueError):
                verify_environment(self.python, {**saved, 'inventory_sha256': 'f' * 64})
        self.assertEqual(saved['interpreter_sha256'], hashlib.sha256(self.python.read_bytes()).hexdigest())

    def test_malformed_or_wrong_interpreter_probe_fails(self):
        observed = inventory(self.root, self.python)
        observed['interpreter_sha256'] = 'f' * 64
        with patch('src.arc.rsi_environment_identity.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(observed), stderr='')):
            with self.assertRaises(ValueError):
                snapshot_environment(self.python)

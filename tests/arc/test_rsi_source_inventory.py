"""Conservative source inventory detects omitted or changed local dependencies."""
import tempfile
import unittest
from pathlib import Path

from src.arc.rsi_source_inventory import build_source_inventory, verify_source_inventory


class SourceInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for directory in ('src', 'sandbox', 'configs'):
            (self.root / directory).mkdir()
        (self.root / 'src/main.py').write_text('import dynamic_module\n')
        (self.root / 'src/dynamic_module.py').write_text('VALUE = 1\n')
        (self.root / 'sandbox/runner.py').write_text('pass\n')
        (self.root / 'configs/run.yml').write_text('seed: 1\n')
        (self.root / 'pyproject.toml').write_text('[project]\nname = "example"\n')
        (self.root / 'uv.lock').write_text('version = 1\n')

    def test_includes_all_source_files_and_stays_non_authorizing(self):
        result = build_source_inventory(self.root)
        self.assertIn('src/dynamic_module.py', result['files'])
        self.assertIn('sandbox/runner.py', result['files'])
        self.assertIn('configs/run.yml', result['files'])
        self.assertFalse(result['execution_ready'])
        verify_source_inventory(self.root, result)

    def test_changed_added_or_removed_source_fails_verification(self):
        result = build_source_inventory(self.root)
        changed = self.root / 'src/dynamic_module.py'
        changed.write_text('VALUE = 2\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify_source_inventory(self.root, result)
        changed.write_text('VALUE = 1\n')
        (self.root / 'src/new.py').write_text('pass\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify_source_inventory(self.root, result)
        (self.root / 'src/new.py').unlink()
        changed.unlink()
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify_source_inventory(self.root, result)

    def test_missing_required_root_and_symlink_fail_closed(self):
        (self.root / 'uv.lock').unlink()
        with self.assertRaisesRegex(ValueError, 'required'):
            build_source_inventory(self.root)
        (self.root / 'uv.lock').write_text('version = 1\n')
        (self.root / 'src/link.py').symlink_to(self.root / 'src/main.py')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            build_source_inventory(self.root)

    def test_python_cache_files_do_not_change_inventory(self):
        result = build_source_inventory(self.root)
        (self.root / 'src/__pycache__').mkdir()
        (self.root / 'src/__pycache__/main.pyc').write_bytes(b'cache')
        verify_source_inventory(self.root, result)


if __name__ == '__main__':
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from src.arc.lt_source import capture_source_tree, verify_source_tree


class SourceTreeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('src', 'sandbox', 'configs', 'scripts'):
            (self.root / name).mkdir()
            (self.root / name / 'example.txt').write_text(name)
        for name in ('pyproject.toml', 'uv.lock'):
            (self.root / name).write_text(name)

    def test_tree_digest_covers_all_registered_files_and_changes(self):
        receipt = capture_source_tree(self.root)
        self.assertEqual(len(receipt['files']), 6)
        self.assertFalse(receipt['execution_ready'])
        self.assertEqual(receipt, capture_source_tree(self.root))
        verify_source_tree(self.root, receipt)
        (self.root / 'src' / 'example.txt').write_text('changed')
        self.assertNotEqual(receipt['code_sha256'], capture_source_tree(self.root)['code_sha256'])
        with self.assertRaises(ValueError): verify_source_tree(self.root, receipt)
        (self.root / 'scripts' / 'added.py').write_text('new')
        self.assertNotEqual(set(receipt['files']), set(capture_source_tree(self.root)['files']))

    def test_symlinked_files_and_missing_roots_are_rejected(self):
        (self.root / 'src' / 'link').symlink_to(self.root / 'uv.lock')
        with self.assertRaises(ValueError): capture_source_tree(self.root)
        (self.root / 'src' / 'link').unlink()
        (self.root / 'uv.lock').unlink()
        with self.assertRaises(ValueError): capture_source_tree(self.root)


if __name__ == '__main__': unittest.main()

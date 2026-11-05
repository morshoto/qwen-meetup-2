"""Bounded review of exact prompts in retained records."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_prompt_exposure import audit_prompt_strings


class PromptExposureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'artifacts').mkdir()

    def test_nested_strings_and_jsonl(self):
        (self.root / 'artifacts/a.json').write_text('{"messages":[{"content":"prefix exact prompt suffix"}]}')
        (self.root / 'artifacts/b.jsonl').write_text('{"body":"other"}\n{"body":"second prompt"}\n')
        result = audit_prompt_strings(self.root, {'a': 'exact prompt', 'b': 'second prompt', 'c': 'absent'})
        self.assertEqual(result['matches'], {'a': ['artifacts/a.json'], 'b': ['artifacts/b.jsonl']})
        self.assertEqual(result['file_count'], 2)
        self.assertEqual(len(result['inventory_sha256']), 64)

    def test_invalid_inputs_fail_closed(self):
        (self.root / 'artifacts/a.json').write_text('{')
        with self.assertRaises(ValueError):
            audit_prompt_strings(self.root, {'a': 'prompt'})
        with self.assertRaises(ValueError):
            audit_prompt_strings(self.root, {'a': ''})
        with self.assertRaises(ValueError):
            audit_prompt_strings(self.root / 'missing', {'a': 'prompt'})

    def test_changed_file_fails_closed(self):
        path = self.root / 'artifacts/a.json'
        path.write_text('{"content":"prompt"}')
        read = Path.read_bytes
        def changing(p):
            value = read(p)
            if p.resolve() == path.resolve():
                path.write_text('{"content":"changed"}')
            return value
        with patch.object(Path, 'read_bytes', changing):
            with self.assertRaisesRegex(ValueError, 'changed'):
                audit_prompt_strings(self.root, {'a': 'prompt'})

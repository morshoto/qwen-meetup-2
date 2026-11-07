import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.lt_single_manifest import build, verify


class SingleManifestTests(unittest.TestCase):
    def setUp(self):
        self.base = {'task_ids': ['a', 'b'], 'sha256': 'a' * 64,
                     'models': {'14b': {}, '32b': {}}, 'candidate_count': 2}
        self.config = {'model_id': 'example/model', 'revision': '1' * 40,
                       'files_sha256': {'config.json': '2' * 64},
                       'chat_template_sha256': '3' * 64}

    @patch('src.arc.lt_single_manifest.build_phase_a')
    def test_registration_binds_one_configured_model_and_phase_a_tasks(self, base):
        base.return_value = self.base
        result = build(Path('.'), self.config)
        self.assertEqual(result['task_ids'], ['a', 'b'])
        self.assertEqual(result['model'], self.config)
        self.assertEqual(result['phase_a_sha256'], 'a' * 64)
        self.assertNotIn('models', result)
        verify(Path('.'), result)
        changed = copy.deepcopy(result)
        changed['model']['revision'] = '4' * 40
        with self.assertRaises(ValueError):
            verify(Path('.'), changed)

    @patch('src.arc.lt_single_manifest.build_phase_a')
    def test_registration_rejects_changed_single_model_source(self, base):
        base.return_value = self.base
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'runner.py'
            source.write_text('original')
            with patch('src.arc.lt_single_manifest.SOURCE_FILES', ('runner.py',)):
                manifest = build(root, self.config)
                source.write_text('changed')
                with self.assertRaises(ValueError):
                    verify(root, manifest)

    @patch('src.arc.lt_single_manifest.build_phase_a')
    def test_rejects_unpinned_revision_and_unsafe_inventory(self, base):
        base.return_value = self.base
        for update in ({'revision': 'main'}, {'model_id': ''}, {'model_id': '../../escape'},
                       {'files_sha256': {'../escape': '2' * 64}},
                       {'files_sha256': {}}, {'chat_template_sha256': 'bad'}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                build(Path('.'), {**self.config, **update})

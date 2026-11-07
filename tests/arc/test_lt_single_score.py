import unittest
from src.arc.data import Verifier
from src.arc.lt_single_score import score_single


class SingleScoreTests(unittest.TestCase):
    def test_selected_model_scores_without_fabricating_second_model(self):
        row = {'model': 'single', 'task_id': 'a', 'status': 'complete',
               'candidates': [[[[2]]], [[[0]]]], 'tokens': 5, 'runtime_seconds': 1,
               'peak_memory_bytes': 10, 'peak_swap_bytes': None}
        result = score_single(['a', 'b'], [row], Verifier({'a': [[[2]]], 'b': [[[3]]]}))
        self.assertEqual(list(result['models']), ['single'])
        self.assertEqual(result['models']['single']['status'], 'incomplete')
        self.assertEqual(result['models']['single']['pass_at_1'], 1)
        self.assertEqual(result['models']['single']['selected'], 2)
        self.assertEqual(result['models']['single']['attempted'], 1)
        self.assertEqual({row['model'] for row in result['rows']}, {'single'})

    def test_rejects_foreign_model(self):
        with self.assertRaises(ValueError):
            score_single(['a'], [{'model': '14b'}], Verifier({'a': [[[1]]]}))

    def test_directory_scoring_binds_journal_and_model_identity(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from src.cli.lt_single import score_directory
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'task_ids': ['a'], 'sha256': 'a' * 64, 'solutions_sha256': 'b' * 64,
                        'protocol': 'lt-single-model-exploratory',
                        'model': {'model_id': 'example/model', 'revision': 'c' * 40}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            journal = root / 'single'
            journal.mkdir()
            identity = 'a' * 64 + ':single'
            (journal / 'identity.json').write_text(json.dumps({'identity': identity, 'task_ids': ['a']}))
            (journal / 'a.started.json').write_text(json.dumps({'task_id': 'a', 'identity': identity}))
            row = {'model': 'single', 'task_id': 'a', 'identity': identity, 'status': 'complete',
                   'candidates': [[[[2]]], [[[0]]]], 'tokens': 5, 'runtime_seconds': 1,
                   'peak_memory_bytes': 10, 'peak_swap_bytes': None}
            (journal / 'a.result.json').write_text(json.dumps(row))
            with patch('src.cli.lt_single.verify'), patch('src.cli.lt_single.load_arc',
                    return_value=({'a': object()}, Verifier({'a': [[[2]]]}))):
                report = score_directory(root)
                self.assertEqual(report['models']['single']['pass_at_1'], 1)
                self.assertEqual(report['evidence']['model'], manifest['model'])
                row['model'] = 'other'
                (journal / 'a.result.json').write_text(json.dumps(row))
                with self.assertRaises(ValueError): score_directory(root)

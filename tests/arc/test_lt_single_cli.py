import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli.lt_single import prepare, run


class SingleRunnerTests(unittest.TestCase):
    def test_configured_model_runs_and_resumes_without_other_model(self):
        task = {'train': [{'input': [[1]], 'output': [[1]]}], 'test': [{'input': [[2]]}]}
        manifest = {'task_ids': ['a'], 'sha256': 'a' * 64, 'batch_size': 10,
                    'task_timeout_seconds': 5, 'model': {'model_id': 'example/model'}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data/raw').mkdir(parents=True)
            (root / 'data/raw/arc-agi_training_challenges.json').write_text(json.dumps({'a': task}))
            config = root / 'config.json'
            config.write_text(json.dumps(manifest['model']))
            dest = root / 'run'
            def fake(journal, model, limit, worker, on_result=None):
                ids = journal.pending(limit)
                for task_id in ids:
                    journal.start(task_id)
                    journal.finish(task_id, {'task_id': task_id, 'model': model, 'status': 'complete'})
                return len(ids)
            with patch('src.cli.lt_single.ROOT', root), patch('src.cli.lt_single.build', return_value=manifest), \
                 patch('src.cli.lt_single.verify'), patch('src.cli.lt_single.verify_model', return_value=Path('/model')), \
                 patch('src.cli.lt_single.supervised_batch', side_effect=fake):
                self.assertEqual(prepare(dest, config), 'a' * 64)
                self.assertEqual(run(dest), 1)
                self.assertEqual(run(dest), 0)
                self.assertTrue((dest / 'single/a.result.json').exists())
                export = json.loads((dest / 'export.json').read_text())
                export['tasks']['a']['test'][0]['input'] = [[9]]
                (dest / 'export.json').write_text(json.dumps(export))
                with self.assertRaisesRegex(ValueError, 'export identity'):
                    run(dest)

    def test_runner_wires_supervision_to_worker_and_journal(self):
        task = {'train': [{'input': [[1]], 'output': [[1]]}], 'test': [{'input': [[2]]}]}
        manifest = {'task_ids': ['a'], 'sha256': 'a' * 64, 'batch_size': 10,
                    'task_timeout_seconds': 5, 'model': {'model_id': 'example/model'}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data/raw').mkdir(parents=True)
            (root / 'data/raw/arc-agi_training_challenges.json').write_text(json.dumps({'a': task}))
            config = root / 'config.json'
            config.write_text('{}')
            dest = root / 'run'
            def worker(argv, timeout, on_start):
                self.assertIn('src.cli.lt_single_worker', argv)
                on_start(123)
                output = Path(argv[argv.index('--output') + 1])
                output.write_text(json.dumps({'task_id': 'a', 'model': 'single', 'status': 'complete',
                                              'candidates': [None, None]}))
                return {'leader_reaped': True, 'group_absent': True, 'status': 'complete'}
            with patch('src.cli.lt_single.ROOT', root), patch('src.cli.lt_single.build', return_value=manifest), \
                 patch('src.cli.lt_single.verify'), patch('src.cli.lt_single.verify_model', return_value=Path('/model')), \
                 patch('src.cli.lt_single.run_worker', side_effect=worker):
                prepare(dest, config)
                self.assertEqual(run(dest), 1)
                self.assertEqual(run(dest), 0)
                self.assertEqual(json.loads((dest / 'single/a.result.json').read_text())['model'], 'single')

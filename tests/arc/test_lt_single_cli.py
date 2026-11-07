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

    def test_model_verification_rejects_chat_template_mismatch(self):
        import hashlib
        from src.cli.lt_single import verify_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tokenizer_config.json').write_text(json.dumps({'chat_template': 'template'}))
            manifest = {'model': {'chat_template_sha256': hashlib.sha256(b'template').hexdigest()}}
            with patch('src.cli.lt_single.verify_phase_a_model', return_value=root):
                self.assertEqual(verify_model(manifest), root)
                manifest['model']['chat_template_sha256'] = '0' * 64
                with self.assertRaises(ValueError): verify_model(manifest)

    def test_vllm_runner_routes_selected_backend_and_current_python(self):
        from src.cli.lt_single import worker_command
        command = worker_command({'backend': {'name': 'vllm', 'version': '0.16.0'}},
                                 Path('/run'), 'a', Path('/out'), Path('/model'))
        import sys
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[command.index('--backend') + 1], 'vllm')
        self.assertEqual(command[command.index('--backend-version') + 1], '0.16.0')

    def test_model_verification_accepts_separate_chat_template(self):
        import hashlib
        from src.cli.lt_single import verify_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'chat_template.jinja').write_text('template')
            manifest = {'model': {'chat_template_sha256': hashlib.sha256(b'template').hexdigest()}}
            with patch('src.cli.lt_single.verify_phase_a_model', return_value=root):
                self.assertEqual(verify_model(manifest), root)

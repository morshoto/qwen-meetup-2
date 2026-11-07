import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.lt_benchmark_manifest import canonical, sha
from src.arc.lt_public import build_public_export


class PilotTests(unittest.TestCase):
    def fixture(self, root):
        from src.cli.lt_vllm_pilot import prepare
        tasks = {name: {'train': [{'input': [[1]], 'output': [[1]]}], 'test': [{'input': [[2]]}]} for name in ('dev', 'dev2', 'dev3')}
        identity = 'a' * 64
        receipt = {'development_manifest_sha256': identity, 'selected_task_sha256': {name: sha(canonical(task)) for name, task in tasks.items()},
                   'purpose': 'development_only', 'execution_ready': False}
        export = {'development_manifest_sha256': identity, 'tasks': tasks}
        model = {'model_id': 'Qwen/Qwen3.8-27B', 'revision': 'b' * 40,
                 'files_sha256': {'file': 'c' * 64}, 'chat_template_sha256': 'd' * 64}
        with patch('src.cli.lt_vllm_pilot.development_ids', return_value=['dev', 'dev2', 'dev3']):
            prepare(root / 'run', export, receipt, model, '0.33.0')
        return model

    def test_prepares_only_registered_public_development_tasks(self):
        from src.cli.lt_vllm_pilot import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self.fixture(root)
            manifest = json.loads((root / 'run/manifest.json').read_text())
            self.assertEqual(manifest['task_ids'], ['dev', 'dev2', 'dev3'])
            self.assertEqual(manifest['purpose'], 'development_only')
            self.assertEqual(manifest['model'], model)
            self.assertEqual(manifest['backend_version'], '0.33.0')
            export = json.loads((root / 'run/export.json').read_text())
            self.assertEqual(set(export['tasks']), {'dev', 'dev2', 'dev3'})
            self.assertNotIn('output', export['tasks']['dev']['test'][0])
            receipt = json.loads((root / 'run/receipt.json').read_text())
            with patch('src.cli.lt_vllm_pilot.development_ids', return_value=['other']):
                with self.assertRaises(ValueError):
                    prepare(root / 'other', {'development_manifest_sha256': receipt['comparison_sha256'], 'tasks': export['tasks']},
                            {'development_manifest_sha256': receipt['comparison_sha256'], 'selected_task_sha256': receipt['selected_task_sha256'], 'purpose':'development_only', 'execution_ready':False}, model, '0.33.0')

    def test_supervised_failure_stops_before_next_task_and_keeps_evidence(self):
        from src.cli.lt_vllm_pilot import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            def worker(argv, timeout, on_start):
                on_start(123)
                return {'status': 'error', 'leader_reaped': True, 'group_absent': True,
                        'elapsed_seconds': 2, 'returncode': 1, 'stderr': 'failed'}
            with patch('src.cli.lt_vllm_pilot.development_ids', return_value=['dev', 'dev2', 'dev3']), \
                 patch('src.cli.lt_vllm_pilot.verify_model', return_value=Path('/model')), \
                 patch('src.cli.lt_vllm_pilot.importlib.metadata.version', return_value='0.33.0'), \
                 patch('src.cli.lt_vllm_pilot.run_worker', side_effect=worker):
                self.assertEqual(run(root / 'run'), 1)
            row = json.loads((root / 'run/single/dev.result.json').read_text())
            self.assertEqual(row['status'], 'error')
            self.assertEqual(row['worker_diagnostics']['stderr'], 'failed')
            self.assertTrue(row['cleanup']['group_absent'])

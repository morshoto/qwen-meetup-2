import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.arc.lt_batch_journal import BatchJournal
from src.arc.lt_single_supervision import supervised_batch, reconcile


class SingleSupervisionTests(unittest.TestCase):
    def test_single_worker_result_is_journaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = BatchJournal(Path(tmp), ['a'], 'run:single')
            def worker(task, path):
                path.write_text(json.dumps({'task_id': task, 'model': 'single', 'status': 'complete',
                                            'candidates': [None, None]}))
                return {'leader_reaped': True, 'group_absent': True, 'status': 'complete'}
            self.assertEqual(supervised_batch(journal, 'single', 1, worker), 1)
            self.assertEqual(json.loads((journal.root / 'a.result.json').read_text())['model'], 'single')

    def test_reconcile_recovers_single_worker_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = BatchJournal(Path(tmp), ['a'], 'run:single')
            journal.start('a')
            (journal.root / 'a.process.json').write_text(json.dumps({'task_id': 'a', 'identity': journal.identity, 'pid': 123}))
            row = {'task_id': 'a', 'model': 'single', 'status': 'complete', 'candidates': [None, None],
                   'tokens': None, 'runtime_seconds': None, 'peak_memory_bytes': None, 'peak_swap_bytes': None}
            (journal.root / 'a.worker.json').write_text(json.dumps(row))
            with patch('src.arc.lt_single_supervision.os.killpg', side_effect=ProcessLookupError):
                self.assertEqual(reconcile(journal, 'a'), 'recovered_complete')

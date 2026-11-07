import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src.arc.lt_batch_journal import BatchJournal,_create
from src.arc.lt_batch_recovery import reconcile

class RecoveryTests(unittest.TestCase):
    def test_absent_group_preserves_failed_attempt_and_allows_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);j=BatchJournal(root,['a'],'id');j.start('a')
            _create(root/'a.process.json',{'task_id':'a','identity':'id','pid':12345})
            (root/'a.worker.json').write_text('{')
            with patch('src.arc.lt_batch_recovery.os.killpg',side_effect=ProcessLookupError): reconcile(j,'a')
            self.assertEqual(j.pending(1),['a'])
            self.assertEqual(len(list(root.glob('a.interrupted-*.started.json'))),1)
            self.assertEqual(len(list(root.glob('a.interrupted-*.worker.json'))),1)
    def test_live_or_untracked_worker_remains_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);j=BatchJournal(root,['a'],'id');j.start('a')
            with self.assertRaises(ValueError): reconcile(j,'a')
            _create(root/'a.process.json',{'task_id':'a','identity':'id','pid':12345})
            with patch('src.arc.lt_batch_recovery.os.killpg',return_value=None), self.assertRaises(ValueError): reconcile(j,'a')
            with self.assertRaises(ValueError): j.pending(1)
    def test_complete_worker_output_is_committed_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);j=BatchJournal(root,['a'],'id:14b');j.start('a')
            _create(root/'a.process.json',{'task_id':'a','identity':'id:14b','pid':12345})
            _create(root/'a.worker.json',{'task_id':'a','model':'14b','status':'complete','candidates':[[[[1]]],[[[1]]]],'tokens':1,'runtime_seconds':1,'peak_memory_bytes':1,'peak_swap_bytes':None})
            with patch('src.arc.lt_batch_recovery.os.killpg',side_effect=ProcessLookupError): reconcile(j,'a')
            self.assertEqual(j.pending(1),[])
            self.assertEqual(json.loads((root/'a.result.json').read_text())['status'],'complete')

import json,tempfile,unittest
from pathlib import Path
from src.arc.lt_batch_journal import BatchJournal
from src.arc.lt_supervised_batch import supervised_batch

class SupervisedBatchTests(unittest.TestCase):
    def test_success_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            j=BatchJournal(Path(d),['a'],'id')
            def worker(task,path):
                path.write_text(json.dumps({'task_id':task,'model':'14b','status':'complete','candidates':[[[[1]]],[[[1]]]],'tokens':3,'runtime_seconds':1,'peak_memory_bytes':1,'peak_swap_bytes':None}))
                return {'status':'complete','leader_reaped':True,'group_absent':True,'returncode':0}
            self.assertEqual(supervised_batch(j,'14b',10,worker),1)
            self.assertEqual(supervised_batch(j,'14b',10,worker),0)
    def test_timeout_is_not_a_fabricated_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            j=BatchJournal(Path(d),['a'],'id')
            worker=lambda *args:{'status':'timeout','leader_reaped':True,'group_absent':True,'returncode':-9}
            self.assertEqual(supervised_batch(j,'14b',10,worker),1)
            row=json.loads((Path(d)/'a.result.json').read_text())
            self.assertEqual((row['status'],row['candidates'],row['tokens']),('timeout',[],None))
    def test_unverified_cleanup_prevents_result(self):
        with tempfile.TemporaryDirectory() as d:
            j=BatchJournal(Path(d),['a'],'id')
            with self.assertRaises(ValueError): supervised_batch(j,'14b',1,lambda *args:{'status':'timeout','leader_reaped':True,'group_absent':False})
            with self.assertRaises(ValueError): j.pending(1)

import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src.arc.data import Verifier
from src.arc.lt_batch_journal import BatchJournal
from src.cli.lt_benchmark_score import score_directory

class BenchmarkScoreTests(unittest.TestCase):
    def test_interrupted_task_does_not_hide_partial_score(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest={'sha256':'a'*64,'task_ids':['a','b'],'solutions_sha256':'b'*64,'protocol':'fixture'}
            (root/'manifest.json').write_text(json.dumps(manifest))
            j=BatchJournal(root/'14b',['a','b'],'a'*64+':14b')
            j.start('a');j.finish('a',{'task_id':'a','model':'14b','status':'complete','candidates':[[[[1]]],[[[1]]]],'tokens':1,'runtime_seconds':1,'peak_memory_bytes':1,'peak_swap_bytes':None});j.start('b')
            with patch('src.cli.lt_benchmark_score.verify'),patch('src.cli.lt_benchmark_score.load_arc',return_value=({'a':1,'b':1},Verifier({'a':[[[1]]],'b':[[[2]]]}))):
                report=score_directory(root)
            self.assertEqual(report['models']['14b']['attempted'],1)
            self.assertEqual(report['models']['14b']['pass_at_1'],1)
            self.assertEqual(report['evidence']['started_without_result'],['14b:b'])

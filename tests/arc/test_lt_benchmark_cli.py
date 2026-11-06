import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src.cli.lt_benchmark import prepare,run

class BenchmarkCLITests(unittest.TestCase):
    def test_prepare_resume_and_model_order(self):
        task={'train':[{'input':[[1]],'output':[[1]]}],'test':[{'input':[[2]]}]}
        manifest={'task_ids':['a'],'sha256':'a'*64,'batch_size':10,'task_timeout_seconds':5}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data/raw').mkdir(parents=True)
            (root/'data/raw/arc-agi_training_challenges.json').write_text(json.dumps({'a':task}))
            dest=root/'run'
            def fake(journal,model,limit,worker):
                ids=journal.pending(limit)
                for task_id in ids:
                    journal.start(task_id);journal.finish(task_id,{'task_id':task_id,'model':model,'status':'complete'})
                return len(ids)
            with patch('src.cli.lt_benchmark.ROOT',root),patch('src.cli.lt_benchmark.build',return_value=manifest),patch('src.cli.lt_benchmark.verify'),patch('src.cli.lt_benchmark.verify_model',return_value=Path('/model')),patch('src.cli.lt_benchmark.supervised_batch',side_effect=fake):
                self.assertEqual(prepare(dest),'a'*64)
                export=json.loads((dest/'export.json').read_text())
                export['tasks']['a']['test'][0]['input']=[[9]]
                (dest/'export.json').write_text(json.dumps(export))
                with self.assertRaisesRegex(ValueError, 'export identity'):run(dest,'14b')
                export['tasks']['a']['test'][0]['input']=[[2]]
                (dest/'export.json').write_text(json.dumps(export))
                with self.assertRaises(ValueError):run(dest,'32b')
                self.assertEqual(run(dest,'14b'),1)
                self.assertEqual(run(dest,'14b'),0)
                self.assertEqual(run(dest,'32b'),1)

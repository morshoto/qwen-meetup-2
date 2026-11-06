import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src.arc.lt_public import build_public_export
from src.cli.lt_batch_worker import main

class WorkerCLITests(unittest.TestCase):
    def test_only_public_task_is_given_to_backend(self):
        item={'train':[{'input':[[1]],'output':[[1]]}],'test':[{'input':[[2]]}]}
        digest=hashlib.sha256(json.dumps(item,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        receipt={'comparison_sha256':'a'*64,'selected_task_sha256':{'t':digest}}
        export=build_public_export({'t':item},receipt)
        class Backend:
            def __init__(self, path): pass
            def generate(self,prompt,*,kind,candidate):
                self.raw='[[[2]]]';self.stats={'prompt_tokens':2,'completion_tokens':1,'peak_memory_gb':1.0,'finish_reason':'stop'}
                self.assert_no_answer=prompt
                return self.raw
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'e').write_text(json.dumps(export));(root/'r').write_text(json.dumps(receipt))
            argv=['worker','--export',str(root/'e'),'--receipt',str(root/'r'),'--output',str(root/'out'),'--task','t','--model','14b','--model-path','/model']
            with patch('sys.argv',argv), patch('src.cli.lt_batch_worker.MLXBackend',Backend): main()
            row=json.loads((root/'out').read_text());self.assertEqual(row['candidates'],[[[[2]]],[[[2]]]])
            with patch('sys.argv',argv), patch('src.cli.lt_batch_worker.MLXBackend',Backend), self.assertRaises(FileExistsError): main()

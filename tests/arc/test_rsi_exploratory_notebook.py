"""The notebook can present a partial run without presenting it as complete."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


class ExploratoryNotebookTests(unittest.TestCase):
    def test_incomplete_branch_uses_read_only_partial_view(self):
        notebook=json.loads(Path('nb/results/13_rsi_exploratory_50.ipynb').read_text())
        code=''.join(notebook['cells'][3]['source'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); registration=root/'registration.json'; split=root/'split.json'; run=root/'run'; run.mkdir()
            registration.write_text('{"fixture":true}')
            split.write_text(json.dumps({'exploratory_task_ids':['t'],'confirmatory_task_ids':['c']}))
            read=Mock(return_value={'data':{'fixture':'data'},'summary':{'complete':False},
                                    'failure_type':'TimeoutError'})
            env={'RESULTS_PATH':root/'missing.json','PARTIAL_RUN_PATH':run,'PARTIAL_REGISTRATION_PATH':registration,
                 'PARTIAL_ADAPTER_SHA256':'a'*64,'SPLIT_PATH':split,'json':json,
                 'read_incomplete_exploratory':read,'display':lambda x:None,'Markdown':lambda x:x}
            exec(compile(code,'notebook-cell','exec'),env)
            self.assertEqual(env['data'],{'fixture':'data'})
            self.assertFalse(env['summary']['complete'])
            self.assertEqual(read.call_args.kwargs['candidate_adapter'],'a'*64)

    def test_default_results_branch_rejects_incomplete_marker(self):
        notebook=json.loads(Path('nb/results/13_rsi_exploratory_50.ipynb').read_text())
        code=''.join(notebook['cells'][3]['source'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); results=root/'exploratory-results.json';results.write_text('{}')
            (root/'exploratory.incomplete.json').write_text('{}')
            env={'RESULTS_PATH':results,'PARTIAL_RUN_PATH':None,'json':json,
                 'summarize_exploratory':Mock(return_value={'complete':True}),
                 'display':lambda x:None,'Markdown':lambda x:x}
            with self.assertRaises(ValueError): exec(compile(code,'notebook-cell','exec'),env)
            env['summarize_exploratory'].assert_not_called()

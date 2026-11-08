import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.lt_single_manifest import build, verify

class ColabRegistrationTests(unittest.TestCase):
    @patch('src.arc.lt_single_manifest.build_phase_a')
    def test_fp8_context_is_registered_and_tampering_rejected(self, phase):
        phase.return_value={'task_ids':['one'],'sha256':'a'*64,'models':{},'candidate_count':2}
        model={'model_id':'Qwen/Qwen3.8-27B-FP8','revision':'b'*40,'files_sha256':{'config.json':'c'*64},'chat_template_sha256':'d'*64}
        backend={'name':'vllm','version':'0.33.0','max_model_len':18432}
        manifest=build(Path('.'),{'model':model,'backend':backend})
        self.assertEqual(manifest['backend_settings']['max_model_len'],18432)
        verify(Path('.'),manifest)
        changed=copy.deepcopy(manifest);changed['backend']['max_model_len']=8192
        with self.assertRaises(ValueError):verify(Path('.'),changed)

    def test_worker_receives_registered_context(self):
        from src.cli.lt_single import worker_command
        manifest={'backend':{'name':'vllm','version':'0.33.0','max_model_len':18432}}
        argv=worker_command(manifest,Path('/tmp/run'),'one',Path('/tmp/out'),Path('/model'))
        self.assertEqual(argv[argv.index('--max-model-len')+1], '18432')

    @patch('src.arc.lt_single_manifest.build_phase_a')
    def test_public_only_verification_uses_registered_solutions_digest(self, phase):
        phase.return_value={'task_ids':['one'],'sha256':'a'*64,'models':{},'candidate_count':2,'solutions_sha256':'e'*64}
        model={'model_id':'Qwen/Qwen3.8-27B-FP8','revision':'b'*40,'files_sha256':{'config.json':'c'*64},'chat_template_sha256':'d'*64}
        manifest=build(Path('.'),model)
        verify(Path('.'),manifest,public_only=True)
        self.assertEqual(phase.call_args.kwargs, {'solutions_sha256':'e'*64})
        phase.return_value['solutions_sha256']='f'*64
        with self.assertRaises(ValueError): verify(Path('.'),manifest,public_only=True)

class ColabContextTests(unittest.TestCase):
    def test_context_preflight_stops_before_generation(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock
        from src.arc.lt_vllm_backend import VLLMBackend
        backend=object.__new__(VLLMBackend)
        backend.max_model_len=5
        backend.max_tokens=4
        backend.tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:'prompt',encode=lambda *a,**k:[1,2])
        backend.llm=SimpleNamespace(generate=Mock())
        with self.assertRaisesRegex(ValueError,'exceed registered context'):
            backend.generate(json.dumps({'task':{'test':[{'input':[[1]]}]}}),kind='direct',candidate=0)
        backend.llm.generate.assert_not_called()

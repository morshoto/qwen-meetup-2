"""Separate canary proposals must not depend on Oracle generation."""
import unittest
from pathlib import Path

from src.arc.rsi_canary_dataset import prepare_canary_dataset
from src.arc.rsi_technical_canary import propose_technical_canary
from tests.arc.test_rsi_mlx_dataset import Dataset
import tests.arc.test_rsi_sft_format as fixtures


class TechnicalCanaryTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.SFTFormatTests('test_rows_match_inference_prefix_and_mask'); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root=self.f.root
        self.canary=prepare_canary_dataset(root=self.root,forbidden_tasks=self.f.tasks,tokenizer=self.f.tokenizer,render=self.f.render)
        self.ids={'model_manifest':'a'*64,'runtime_manifest':'b'*64,'environment_manifest':'c'*64,
                  'source_inventory':'d'*64,'tokenizer':'e'*64,'tokenizer_config':'f'*64}
        self.limits={'timeout_seconds':300,'minimum_disk_gib':20,'minimum_free_memory_percent':20,
                     'maximum_swap_growth_mib':512,'maximum_process_rss_gib':32,'sample_interval_seconds':5}
        self.revisions={'14b':'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4','32b':'bcaaf7f538adf166c1080a2befdb4f6019f66639'}
        for revision in self.revisions.values(): (self.root/'models'/revision).mkdir(parents=True)
    def loader(self,path,tokenizer,config):
        import json
        rows=[json.loads(line) for line in (path/'train.jsonl').read_text().splitlines()]
        return Dataset(rows,self.f.render),[],[]
    def propose(self,model='14b',**changes):
        args=dict(root=self.root,model_key=model,model_dir=self.root/'models'/self.revisions[model],
                  adapter_dir=self.root/f'technical-{model}.adapter',forbidden_tasks=self.f.tasks,
                  tokenizer=self.f.tokenizer,render=self.f.render,load_local_dataset=self.loader,
                  expected_dataset_sha256=self.canary['dataset_jsonl_sha256'],identity_hashes=self.ids,
                  resource_limits=self.limits,seed=42)
        args.update(changes);return propose_technical_canary(**args)
    def test_both_models_have_isolated_one_step_nonpromotion_proposals(self):
        a=self.propose('14b');b=self.propose('32b')
        self.assertEqual(a['model_revision'],self.revisions['14b']); self.assertEqual(b['model_revision'],self.revisions['32b'])
        for item in (a,b):
            self.assertEqual(item['config']['iters'],1); self.assertFalse(item['execution_ready'])
            self.assertFalse(item['training_allowed']); self.assertFalse(item['candidate_reuse_allowed'])
            self.assertEqual(item['data_source'],'fixed_synthetic_canary')
        self.assertNotEqual(a['config']['adapter_path'],b['config']['adapter_path'])
    def test_rejects_malformed_identity_and_seed(self):
        for changes in ({'model_key':[]}, {'identity_hashes':{}}, {'seed':True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.propose(**changes)

    def test_rejects_wrong_model_or_existing_adapter(self):
        with self.assertRaises(ValueError): self.propose('14b',model_dir=self.root/'models'/self.revisions['32b'])
        (self.root/'technical-14b.adapter').mkdir()
        with self.assertRaises(ValueError): self.propose()
    def test_rejects_unsafe_limits_or_generation_dataset(self):
        with self.assertRaises(ValueError): self.propose(resource_limits=dict(self.limits,maximum_swap_growth_mib=513))
        with self.assertRaises(ValueError): self.propose(adapter_dir=self.root/'preflight-canary.quarantine'/'adapter')
        with self.assertRaises(ValueError): self.propose(expected_dataset_sha256='0'*64)

class TechnicalCanaryIdentityTests(TechnicalCanaryTests):
    def test_model_specific_local_identity_checks_remain_non_authorizing(self):
        import hashlib
        import json
        from unittest.mock import patch
        from src.arc.rsi_technical_canary import verify_technical_canary_identities
        manifests={'model_manifest':{},'runtime_manifest':{'interpreter_realpath':'/python','interpreter_sha256':'1'*64},
                   'environment_manifest':{'interpreter_realpath':'/python','interpreter_sha256':'1'*64},'source_inventory':{}}
        raw={key:json.dumps(value).encode() for key,value in manifests.items()}
        for model in ('14b','32b'):
            model_dir=self.root/'models'/self.revisions[model]
            for name in ('tokenizer.json','tokenizer_config.json'): (model_dir/name).write_text(name)
            ids={key:hashlib.sha256(value).hexdigest() for key,value in raw.items()}
            ids.update(tokenizer=hashlib.sha256(b'tokenizer.json').hexdigest(),tokenizer_config=hashlib.sha256(b'tokenizer_config.json').hexdigest())
            proposal=self.propose(model,identity_hashes=ids)
            args=dict(proposal=proposal,manifest_bytes=raw,source_root=self.root,python_executable='/python',
                      root=self.root,forbidden_tasks=self.f.tasks,tokenizer=self.f.tokenizer,render=self.f.render,
                      load_local_dataset=self.loader)
            with patch('src.arc.rsi_technical_canary.verify_model_manifest') as verify_model, \
                 patch('src.arc.rsi_technical_canary.verify_runtime_identity'), \
                 patch('src.arc.rsi_technical_canary.verify_environment'), \
                 patch('src.arc.rsi_technical_canary.verify_source_inventory'):
                result=verify_technical_canary_identities(**args)
                verify_model.assert_called_once_with(model_dir.resolve(),{},model_key=model)
                self.assertFalse(result['execution_ready']); self.assertFalse(result['training_allowed'])
                self.assertFalse(result['independently_verified']); self.assertFalse(result['promotion_allowed'])
                proposal['config']['iters']=2
                with self.assertRaises(ValueError): verify_technical_canary_identities(**args)
                proposal['config']['iters']=1
                with self.assertRaises(ValueError): verify_technical_canary_identities(**dict(args,manifest_bytes=dict(raw,model_manifest=b'{} ')))

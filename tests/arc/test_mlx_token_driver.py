import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from src.arc.mlx_token_driver import MLXTokenDriver, adapter_sha256


class MLXTokenDriverTests(unittest.TestCase):
    def test_rejects_nonlocal_model_before_import(self):
        with self.assertRaisesRegex(ValueError, 'existing local directory'):
            MLXTokenDriver('/definitely/missing/model',temperature=0,seed=3)

    def test_driver_passes_exact_ids_and_limit_to_mlx(self):
        calls={}
        tokenizer=types.SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: [3,4,5],
            decode=lambda ids: ','.join(map(str,ids)), eos_token_ids={99})
        mx=types.ModuleType('mlx.core')
        mx.array=lambda ids: tuple(ids)
        mx.random=types.SimpleNamespace(seed=lambda value: calls.update(seed=value))
        mlx=types.ModuleType('mlx');mlx.core=mx
        lm=types.ModuleType('mlx_lm'); lm.__path__=[]
        generate=types.ModuleType('mlx_lm.generate')
        def step(ids, model, *, max_tokens, sampler):
            calls.update(ids=ids,model=model,max_tokens=max_tokens,sampler=sampler)
            yield (9,None)
        generate.generate_step=step
        sample=types.ModuleType('mlx_lm.sample_utils');sample.make_sampler=lambda temp: ('sampler',temp)
        utils=types.ModuleType('mlx_lm.utils');utils.load=lambda path: ('model',tokenizer)
        modules={'mlx':mlx,'mlx.core':mx,'mlx_lm':lm,'mlx_lm.generate':generate,'mlx_lm.sample_utils':sample,'mlx_lm.utils':utils}
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules,modules):
            driver=MLXTokenDriver(d,temperature=0.2,seed=42)
            ids=driver.encode_messages([{'role':'user','content':'x'}])
            self.assertEqual(ids,[3,4,5])
            self.assertEqual(list(driver.generate_ids(ids,7)),[9])
            self.assertEqual(calls,{'seed':42,'ids':(3,4,5),'model':'model','max_tokens':7,'sampler':('sampler',0.2)})
            self.assertEqual(driver.decode([1,2]),'1,2')
            self.assertEqual(driver.stop_token_ids,frozenset({99}))

    def test_thinking_off_is_passed_to_the_tokenizer(self):
        calls={}
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                calls.update(kwargs); return [1,2]
        driver=MLXTokenDriver.__new__(MLXTokenDriver)
        driver._tokenizer=Tokenizer();driver.thinking='off';driver.supports_thinking_switch=True
        self.assertEqual(driver.encode_messages([{'role':'user','content':'x'}]),[1,2])
        self.assertIs(calls['enable_thinking'],False)

    def test_thinking_on_is_passed_to_the_tokenizer(self):
        calls={}
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                calls.update(kwargs); return [1,2]
        driver=MLXTokenDriver.__new__(MLXTokenDriver)
        driver._tokenizer=Tokenizer();driver.thinking='off';driver.supports_thinking_switch=True
        self.assertEqual(driver.encode_messages([{'role':'user','content':'x'}],thinking_mode='on'),[1,2])
        self.assertIs(calls['enable_thinking'],True)

    def test_thinking_canary_requires_distinct_off_and_on_token_ids(self):
        class Tokenizer:
            chat_template = 'enable_thinking'
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3] if kwargs['enable_thinking'] else [1, 2]
        driver=MLXTokenDriver.__new__(MLXTokenDriver)
        driver._tokenizer=Tokenizer();driver.supports_thinking_switch=True;driver.thinking='off'
        canary=driver._run_thinking_canary()
        self.assertTrue(canary['different_token_ids'])
        self.assertNotEqual(canary['off_sha256'],canary['on_sha256'])
        class IgnoredSwitchTokenizer:
            chat_template = 'enable_thinking'
            def apply_chat_template(self, messages, **kwargs): return [1,2]
        driver._tokenizer=IgnoredSwitchTokenizer();driver.supports_thinking_switch=True
        self.assertFalse(driver._run_thinking_canary()['different_token_ids'])

    def test_adapter_path_and_expected_hash_are_bound_at_model_load(self):
        calls={}
        tokenizer=types.SimpleNamespace(apply_chat_template=lambda messages, **kwargs: [1],decode=lambda ids:'',eos_token_ids={2})
        mx=types.ModuleType('mlx.core');mx.array=lambda ids:tuple(ids);mx.random=types.SimpleNamespace(seed=lambda value:None)
        mlx=types.ModuleType('mlx');mlx.core=mx
        lm=types.ModuleType('mlx_lm');lm.__path__=[]
        generate=types.ModuleType('mlx_lm.generate')
        def step(ids,model,*,max_tokens,sampler):
            yield (2,None)
        generate.generate_step=step
        sample=types.ModuleType('mlx_lm.sample_utils');sample.make_sampler=lambda temp:temp
        utils=types.ModuleType('mlx_lm.utils')
        def load(path,**kwargs):
            calls.update(path=path,kwargs=kwargs);return 'model',tokenizer
        utils.load=load
        modules={'mlx':mlx,'mlx.core':mx,'mlx_lm':lm,'mlx_lm.generate':generate,
                 'mlx_lm.sample_utils':sample,'mlx_lm.utils':utils}
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'model';model.mkdir()
            adapter=Path(d)/'adapter';adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            (adapter/'adapters.safetensors').write_bytes(b'weights')
            expected=adapter_sha256(adapter)
            with patch.dict(sys.modules,modules):
                driver=MLXTokenDriver(model,temperature=0,seed=1,adapter_path=adapter,
                                      expected_adapter_sha256=expected)
            self.assertEqual(calls['path'],str(model.resolve()))
            self.assertEqual(calls['kwargs']['adapter_path'],str(adapter.resolve()))
            self.assertEqual(driver.adapter_sha256,expected)

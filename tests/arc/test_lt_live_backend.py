import json
import unittest
from types import SimpleNamespace
from src.arc.lt_live_backend import MLXBackend, PROMPT

class BackendTests(unittest.TestCase):
    def test_prompt_controls_and_raw_usage(self):
        b=MLXBackend.__new__(MLXBackend); seen={}; seeds=[]
        def template(messages, **kw): seen.update(messages=messages, options=kw); return 'formatted'
        b.tokenizer=SimpleNamespace(apply_chat_template=template); b.model=object()
        b.mx=SimpleNamespace(random=SimpleNamespace(seed=seeds.append));b.seed=10;b.max_tokens=4096;b.temperature=.6
        b.make_sampler=lambda **kw: kw
        b.stream_generate=lambda *a,**kw: iter([SimpleNamespace(text='[{"output":[[2]]}]',prompt_tokens=4,generation_tokens=5,peak_memory=1.0,finish_reason='stop')])
        result=b.generate(json.dumps({'instruction':'old','task':{'test':[{'input':[[1]]}]}}),kind='direct',candidate=1)
        self.assertEqual(json.loads(result),[[[2]]]);self.assertEqual(seeds,[11])
        self.assertEqual(json.loads(seen['messages'][0]['content'])['instruction'],PROMPT)
        self.assertFalse(seen['options']['enable_thinking']);self.assertEqual(b.stats['completion_tokens'],5)
        self.assertEqual(b.raw,'[{"output":[[2]]}]')

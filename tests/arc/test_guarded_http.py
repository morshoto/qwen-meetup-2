import json
import tempfile
import time
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from src.arc.guarded_generation import GuardedGenerator
from src.arc.guarded_http import GuardedHTTPServer
from src.arc.token_ledger import TokenLedger


class FakeDriver:
    def __init__(self): self.generated=0; self.seed=None
    def set_seed(self,seed): self.seed=seed
    def encode_messages(self,messages): return [1,2,3]
    def generate_ids(self,ids,max_tokens):
        for i in range(100): self.generated+=1; yield i
    def decode(self,ids): return ','.join(map(str,ids))


class GuardedHTTPTests(unittest.TestCase):
    def test_thinking_on_is_advertised_and_selected_per_request(self):
        from src.arc.http_backend import OpenAICompatibleBackend, verify_guarded_contract
        class SwitchDriver(FakeDriver):
            thinking = 'off'
            thinking_modes_supported = ('server_default', 'off', 'on')
            thinking_canary = {'supported': True, 'different_token_ids': True,
                               'off_sha256': 'a' * 64, 'on_sha256': 'b' * 64,
                               'off_token_count': 3, 'on_token_count': 4}
            def encode_messages(self, messages, *, thinking_mode=None):
                self.thinking = thinking_mode or self.thinking
                return [1, 2, 3]
        with tempfile.TemporaryDirectory() as d:
            driver = SwitchDriver()
            ledger = TokenLedger(Path(d) / 'ledger.sqlite', 10000)
            server = GuardedHTTPServer(('127.0.0.1', 0),
                GuardedGenerator(ledger, driver, max_prompt_tokens=10), model='test',
                revision='r', temperature=0.2, max_new_tokens=4)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                base = f'http://127.0.0.1:{server.server_port}/v1'
                contract = verify_guarded_contract(base, 'test', 'r', 10000, 4, 2, 'off',
                    expected_driver=server.driver_type, required_thinking_modes=('on',))
                self.assertIn('on', contract['thinking_modes'])
                backend = OpenAICompatibleBackend(model='test', revision='r', base_url=base,
                    temperature=0.2, max_tokens=2, seed=7, timeout=2, thinking='on', guarded=True)
                self.assertEqual(backend.generate('prompt', kind='text', candidate=0), '0,1')
                self.assertEqual(driver.thinking, 'on')
            finally:
                server.shutdown(); thread.join(); server.server_close()

    def test_server_enforces_ledger_before_generation(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=TokenLedger(Path(d)/'ledger.sqlite',8); driver=FakeDriver()
            guard=GuardedGenerator(ledger,driver,max_prompt_tokens=10)
            server=GuardedHTTPServer(('127.0.0.1',0),guard,model='test',revision='r',temperature=0.2,max_new_tokens=4)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}/v1'
            payload={'model':'test','messages':[{'role':'user','content':'x'}],'temperature':0.2,'max_tokens':4,'seed':1,'stream':False,'deadline_monotonic':time.monotonic()+30}
            def post():
                req=urllib.request.Request(base+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                return urllib.request.urlopen(req,timeout=2)
            try:
                with post() as response: result=json.load(response)
                self.assertEqual(result['choices'][0]['message']['content'],'0,1,2,3')
                self.assertEqual(result['choices'][0]['finish_reason'],'length')
                self.assertEqual(result['usage'],{'prompt_tokens':3,'completion_tokens':4,'total_tokens':7})
                self.assertEqual(driver.generated,4); self.assertEqual(ledger.reserved(),8)
                with self.assertRaises(urllib.error.HTTPError) as exc: post()
                self.assertEqual(exc.exception.code,429);self.assertEqual(driver.generated,4)
                from src.arc.http_backend import BackendError, OpenAICompatibleBackend
                backend=OpenAICompatibleBackend(model='test',revision='r',base_url=base,temperature=0.2,max_tokens=4,seed=1,timeout=2,guarded=True)
                with self.assertRaisesRegex(BackendError,'budget_violation'):
                    backend.generate('x',kind='direct',candidate=0)
                self.assertEqual(driver.generated,4)
                with urllib.request.urlopen(base+'/token-budget',timeout=2) as response: status=json.load(response)
                self.assertEqual((status['token_limit'],status['reserved_tokens']),(8,8))
            finally:
                server.shutdown();thread.join();server.server_close()

    def test_contract_rejects_different_source_and_server_configuration(self):
        from src.arc.http_backend import BackendError, verify_guarded_contract
        from src.arc.token_ledger import TokenBudgetError
        with tempfile.TemporaryDirectory() as d:
            ledger=TokenLedger(Path(d)/'ledger.sqlite',1000);guard=GuardedGenerator(ledger,FakeDriver(),max_prompt_tokens=10)
            server=GuardedHTTPServer(('127.0.0.1',0),guard,model='test',revision='r',temperature=0.2,max_new_tokens=4)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                base=f'http://127.0.0.1:{server.server_port}/v1'
                with self.assertRaisesRegex(BackendError,'budget_unverified'):
                    verify_guarded_contract(base,'test','r',1000,4,2)
                contract=verify_guarded_contract(base,'test','r',1000,4,2,expected_driver=server.driver_type)
                self.assertEqual(contract['ledger_id'],ledger.identity())
                self.assertEqual(contract['adapter_sha256'],'none')
                self.assertEqual(verify_guarded_contract(base,'test','r',1000,4,2,
                                 expected_driver=server.driver_type,expected_max_prompt_tokens=10)['max_prompt_tokens'],10)
                with self.assertRaisesRegex(BackendError,'budget_unverified'):
                    verify_guarded_contract(base,'test','r',1000,4,2,
                                            expected_driver=server.driver_type,expected_max_prompt_tokens=11)
                with self.assertRaisesRegex(BackendError,'budget_unverified'):
                    verify_guarded_contract(base,'test','r',1000,4,2,expected_driver=server.driver_type,
                                            expected_adapter_sha256='f'*64)
                with self.assertRaisesRegex(BackendError,'budget_unverified'):
                    verify_guarded_contract(base,'test','r',1000,4,2,'off',expected_driver=server.driver_type)
                server.source_sha256={'bad':'hash'}
                with self.assertRaisesRegex(BackendError,'budget_unverified'):
                    verify_guarded_contract(base,'test','r',1000,4,2,expected_driver=server.driver_type)
                with self.assertRaises(TokenBudgetError):
                    GuardedHTTPServer(('127.0.0.1',0),guard,model='other',revision='r',temperature=0.2,max_new_tokens=4)
            finally:
                server.shutdown();thread.join();server.server_close()

    def test_non_loopback_bind_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            guard=GuardedGenerator(TokenLedger(Path(d)/'ledger',100),FakeDriver(),max_prompt_tokens=10)
            with self.assertRaises(ValueError): GuardedHTTPServer(('0.0.0.0',0),guard,model='x',revision='r',temperature=0,max_new_tokens=1)

    def test_guarded_server_integrates_with_experiment_and_rejects_ledger_swap(self):
        from dataclasses import replace
        from unittest.mock import patch
        import subprocess
        from src.arc.sandbox import FakeSandbox, DockerSandbox, IDENTITY_CODE
        from src.core.config import load_experiment_config
        from src.pipelines.p3a import run_real
        class TaskDriver:
            def set_seed(self,seed): pass
            def encode_messages(self,messages): return list(messages[-1]['content'].encode())
            def generate_ids(self,ids,max_tokens):
                try:
                    prompt=json.loads(bytes(ids).decode())
                except json.JSONDecodeError:
                    output=IDENTITY_CODE
                else:
                    output=json.dumps([x['input'] for x in prompt['task']['test']]) if 'JSON array' in prompt['instruction'] else IDENTITY_CODE
                yield from output.encode()
            def decode(self,ids): return bytes(ids).decode()
        with tempfile.TemporaryDirectory() as d:
            ledger=TokenLedger(Path(d)/'server.sqlite',100000);driver=TaskDriver()
            guard=GuardedGenerator(ledger,driver,max_prompt_tokens=8192)
            server=GuardedHTTPServer(('127.0.0.1',0),guard,model='test',revision='rev',temperature=0.0,max_new_tokens=512)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                cfg=load_experiment_config('configs/pilot.cpu.yml')
                cfg=replace(cfg, experiment=replace(cfg.experiment,conditions=['A','B'],generations=0,evaluation_generations=[0]),
                    model=replace(cfg.model,id='test',revision='rev',backend='guarded_mlx',base_url=f'http://127.0.0.1:{server.server_port}/v1'),
                    agent=replace(cfg.agent,sandbox='docker',candidates=1),rsi=replace(cfg.rsi,exploration_per_generation=1),
                    tracking=replace(cfg.tracking,output_dir=d),budget=replace(cfg.budget,max_candidates=1))
                original=subprocess.run
                def preflight(args,**kwargs):
                    if args[0]=='docker': return subprocess.CompletedProcess(args,0,stdout='sha256:test')
                    return original(args,**kwargs)
                from src.arc.http_backend import verify_guarded_contract
                def test_contract(*args,**kwargs): return verify_guarded_contract(*args,expected_driver=server.driver_type,**kwargs)
                with patch('src.pipelines.p3a.subprocess.run',side_effect=preflight), patch.object(DockerSandbox,'run',lambda self, code, grid: FakeSandbox().run(code,grid)), patch('src.pipelines.p3a.verify_guarded_contract',side_effect=test_contract):
                    path=run_real(cfg)
                    self.assertEqual(json.loads((path/'run_state.json').read_text())['status'],'budget_exhausted')
                    self.assertGreater(ledger.reserved(),0)
                    server.guard.ledger=TokenLedger(Path(d)/'replacement.sqlite',100000)
                    run_real(cfg,resume=True,run_dir=path)
                    state=json.loads((path/'run_state.json').read_text())
                    self.assertEqual(state['status'],'model_unavailable')
                    self.assertIn('ledger changed',state['error'])
            finally:
                server.shutdown();thread.join();server.server_close()

    def test_guarded_server_with_real_docker_when_available(self):
        import shutil
        import subprocess
        from dataclasses import replace
        from src.arc.sandbox import IDENTITY_CODE
        from src.core.config import load_experiment_config
        from src.pipelines.p3a import run_real
        if not shutil.which('docker'):
            self.skipTest('Docker CLI unavailable')
        try:
            check=subprocess.run(['docker','image','inspect','arc-sandbox:local'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
        except (OSError,subprocess.TimeoutExpired):
            self.skipTest('Docker unavailable')
        if check.returncode:
            self.skipTest('Docker runner image unavailable')
        class TaskDriver:
            def set_seed(self,seed): pass
            def encode_messages(self,messages): return list(messages[-1]['content'].encode())
            def generate_ids(self,ids,max_tokens):
                try:
                    prompt=json.loads(bytes(ids).decode())
                except json.JSONDecodeError:
                    output=IDENTITY_CODE
                else:
                    output=json.dumps([x['input'] for x in prompt['task']['test']]) if 'JSON array' in prompt['instruction'] else IDENTITY_CODE
                yield from output.encode()
            def decode(self,ids): return bytes(ids).decode()
        with tempfile.TemporaryDirectory() as d:
            ledger=TokenLedger(Path(d)/'server.sqlite',100000)
            server=GuardedHTTPServer(('127.0.0.1',0),GuardedGenerator(ledger,TaskDriver(),max_prompt_tokens=8192),model='test',revision='rev',temperature=0.0,max_new_tokens=512)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                cfg=load_experiment_config('configs/pilot.cpu.yml')
                cfg=replace(cfg,experiment=replace(cfg.experiment,conditions=['A','B'],generations=0,evaluation_generations=[0]),
                    model=replace(cfg.model,id='test',revision='rev',backend='guarded_mlx',base_url=f'http://127.0.0.1:{server.server_port}/v1'),
                    agent=replace(cfg.agent,sandbox='docker',candidates=1,timeout_seconds=5),
                    rsi=replace(cfg.rsi,exploration_per_generation=1),tracking=replace(cfg.tracking,output_dir=d))
                from unittest.mock import patch
                from src.arc.http_backend import verify_guarded_contract
                def test_contract(*args,**kwargs): return verify_guarded_contract(*args,expected_driver=server.driver_type,**kwargs)
                with patch('src.pipelines.p3a.verify_guarded_contract',side_effect=test_contract):
                    path=run_real(cfg)
                results=json.loads((path/'results.json').read_text())
                self.assertEqual([x['score']['pass_at_1'] for x in results],[True,True])
                self.assertGreater(json.loads((path/'budget.json').read_text())['containers'],0)
                env=json.loads((path/'environment.json').read_text())
                self.assertIn('TaskDriver',env['guarded_driver_type'])
                self.assertEqual(env['token_budget_guarantee'],'guarded_contract_unverified_runtime')
                self.assertLessEqual(ledger.reserved(),cfg.budget.max_tokens)
                self.assertEqual(run_real(cfg,resume=True,run_dir=path),path)
            finally:
                server.shutdown();thread.join();server.server_close()

class HardDeadlineTests(unittest.TestCase):
    def test_worker_process_terminates_and_keeps_reservation(self):
        import subprocess
        import sys
        import time
        with tempfile.TemporaryDirectory() as d:
            ledger_path=str(Path(d)/'ledger.sqlite')
            script='''
import sys,time
from src.arc.guarded_generation import GuardedGenerator
from src.arc.guarded_http import GuardedHTTPServer
from src.arc.token_ledger import TokenLedger
class SlowDriver:
 def set_seed(self,seed): pass
 def encode_messages(self,messages): return [1,2,3]
 def generate_ids(self,ids,max_tokens):
  time.sleep(5)
  yield 1
 def decode(self,ids): return 'x'
ledger=TokenLedger(sys.argv[1],100)
server=GuardedHTTPServer(('127.0.0.1',0),GuardedGenerator(ledger,SlowDriver(),max_prompt_tokens=10),model='test',revision='r',temperature=0,max_new_tokens=2,max_request_seconds=3)
print(server.server_port,flush=True)
server.serve_forever()
'''
            proc=subprocess.Popen([sys.executable,'-u','-c',script,ledger_path],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                port=int(proc.stdout.readline().strip())
                payload={'model':'test','messages':[{'role':'user','content':'x'}],'temperature':0,'max_tokens':2,'seed':0,'stream':False,'deadline_monotonic':time.monotonic()+0.4}
                req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with self.assertRaises((OSError,urllib.error.URLError)):
                    urllib.request.urlopen(req,timeout=3)
                self.assertNotEqual(proc.wait(timeout=3),0)
                self.assertEqual(TokenLedger(ledger_path,100).reserved(),6)
            finally:
                if proc.poll() is None: proc.kill();proc.wait()
                if proc.stdout: proc.stdout.close()
                if proc.stderr: proc.stderr.close()

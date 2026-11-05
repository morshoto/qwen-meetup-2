import json
import base64
import stat
from dataclasses import replace
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch
from src.arc.http_backend import OpenAICompatibleBackend, BackendError, final_answer
from src.arc.sandbox import DockerSandbox, IDENTITY_CODE


class BackendTests(unittest.TestCase):
    def test_thinking_and_remote_rejection(self):
        self.assertEqual(final_answer('<think>private</think>\n```python\ndef transform(grid): return grid\n```'), 'def transform(grid): return grid')
        self.assertEqual(final_answer('```python\ndef transform(grid): return grid\n```<|im_end|>'), 'def transform(grid): return grid')
        with self.assertRaises(BackendError): final_answer('<think>unfinished')
        with self.assertRaises(ValueError):
            OpenAICompatibleBackend(model='x', revision='r', base_url='https://example.com/v1', temperature=0, max_tokens=20, seed=1, timeout=1)

    def test_http_records_request_and_hashes(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.server.requests.append(request)
                raw = json.dumps({'model': 'test', 'choices': [{'message': {'content': '<think>private</think>[[[1]]]'}}], 'usage': {'total_tokens': 9}}).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
            def log_message(self, *args): pass
        server = HTTPServer(('127.0.0.1', 0), Handler); server.requests = []
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                log = Path(d)/'calls.jsonl'
                client = OpenAICompatibleBackend(model='test', revision='r', base_url=f'http://127.0.0.1:{server.server_port}/v1', temperature=0, max_tokens=20, seed=42, timeout=2, log_path=log)
                self.assertEqual(client.generate('hello', kind='direct', candidate=0), '[[[1]]]')
                self.assertEqual(client.generate('hello', kind='direct', candidate=1), '[[[1]]]')
                entries = [json.loads(x) for x in log.read_text().splitlines()]
                responses_path = Path(d)/'llm_responses.jsonl'
                responses = [json.loads(x) for x in responses_path.read_text().splitlines()]
                self.assertEqual([x['seed'] for x in server.requests], [42, 43])
                self.assertEqual(entries[0]['usage']['total_tokens'], 9)
                self.assertTrue(entries[0]['response_sha256'])
                self.assertEqual(entries[0]['request_id'], responses[0]['request_id'])
                self.assertEqual(base64.b64decode(responses[0]['raw_http_body_base64']),
                                 json.dumps({'model': 'test', 'choices': [{'message': {'content': '<think>private</think>[[[1]]]'}}], 'usage': {'total_tokens': 9}}).encode())
                self.assertEqual(responses[0]['generated_text'], '<think>private</think>[[[1]]]')
                self.assertEqual(responses[0]['diagnostic']['intermediate']['final_answer'], '[[[1]]]')
                self.assertNotIn('hello', responses_path.read_text())
                self.assertEqual(stat.S_IMODE(responses_path.stat().st_mode), 0o600)
                self.assertNotIn('private', log.read_text())
        finally:
            server.shutdown(); thread.join(); server.server_close()

    def test_http_error_and_model_mismatch(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path.endswith('/bad/chat/completions'):
                    self.send_response(429); self.end_headers(); return
                raw=json.dumps({'model':'wrong','choices':[{'message':{'content':'x'}}]}).encode()
                self.send_response(200); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            def client(suffix):
                return OpenAICompatibleBackend(model='expected', revision='r', base_url=f'http://127.0.0.1:{server.server_port}/{suffix}', temperature=0, max_tokens=10, seed=0, timeout=2)
            with self.assertRaisesRegex(BackendError, 'model_mismatch'): client('v1').generate('x',kind='direct',candidate=0)
            with self.assertRaisesRegex(BackendError, 'rate_limit'): client('bad').generate('x',kind='direct',candidate=0)
        finally:
            server.shutdown(); thread.join(); server.server_close()

    def test_guarded_http_error_detail_is_bounded_and_logged(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw=json.dumps({'error':'prompt token limit exceeded before model dispatch'}).encode()
                self.send_response(400); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                log=Path(d)/'requests.jsonl'
                backend=OpenAICompatibleBackend(model='test', revision='r', base_url=f'http://127.0.0.1:{server.server_port}/v1', temperature=0, max_tokens=20, seed=1, timeout=2, log_path=log, guarded=True)
                with self.assertRaisesRegex(BackendError, 'http_error'):
                    backend.generate('x',kind='code',candidate=0)
                error=json.loads(log.read_text())['error']
                self.assertEqual(error['status'], 400)
                self.assertEqual(error['detail'], 'prompt token limit exceeded before model dispatch')
        finally:
            server.shutdown();thread.join();server.server_close()

    def test_timeout_is_logged(self):
        import time
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                time.sleep(0.2)
                try: self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
                except (BrokenPipeError, ConnectionResetError): pass
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                log=Path(d)/'requests.jsonl'
                backend=OpenAICompatibleBackend(model='test', revision='r', base_url=f'http://127.0.0.1:{server.server_port}/v1', temperature=0, max_tokens=20, seed=1, timeout=0.05, log_path=log)
                with self.assertRaises(TimeoutError): backend.generate('x',kind='direct',candidate=0)
                self.assertEqual(json.loads(log.read_text())['error']['kind'], 'timeout')
        finally:
            server.shutdown();thread.join();server.server_close()

    def test_server_overreport_stops_with_budget_error(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw=json.dumps({'model':'test', 'choices':[{'message':{'content':'[]'}}], 'usage':{'total_tokens':999999}}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            backend=OpenAICompatibleBackend(model='test', revision='r', base_url=f'http://127.0.0.1:{server.server_port}/v1', temperature=0, max_tokens=20, seed=1, timeout=2)
            with self.assertRaisesRegex(BackendError, 'budget_violation'):
                backend.generate('x',kind='direct',candidate=0)
        finally:
            server.shutdown();thread.join();server.server_close()

    def test_ollama_native_thinking_disabled(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.server.payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                raw=json.dumps({'model':'test', 'message':{'content':'[[[1]]]'}, 'prompt_eval_count':5, 'eval_count':3}).encode()
                self.send_response(200); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                log=Path(d)/'requests.jsonl'
                backend=OpenAICompatibleBackend(model='test', revision='r', base_url=f'http://127.0.0.1:{server.server_port}', temperature=0, max_tokens=20, seed=1, timeout=2, thinking='off', api_style='ollama_native', log_path=log)
                self.assertEqual(backend.generate('x',kind='direct',candidate=0), '[[[1]]]')
                self.assertIs(server.payload['think'],False)
                self.assertEqual(json.loads(log.read_text())['usage']['total_tokens'],8)
        finally:
            server.shutdown();thread.join();server.server_close()

    def test_no_docker_fails_closed(self):
        with patch('src.arc.sandbox.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'refusing'):
                DockerSandbox().run(IDENTITY_CODE, [[1]])


def docker_available():
    if not shutil.which('docker'): return False
    try:
        return subprocess.run(['docker', 'image', 'inspect', 'arc-sandbox:local'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(docker_available(), 'Docker daemon or prebuilt runner image unavailable')
class DockerIntegrationTests(unittest.TestCase):
    def setUp(self): self.sandbox = DockerSandbox(timeout=3, memory_mb=128)
    def test_success_and_isolation(self):
        self.assertEqual(self.sandbox.run(IDENTITY_CODE, [[1, 2]]), [[1, 2]])
        for code in ["def transform(grid):\n open('/tmp/escape','w').write('x')\n return grid", "def transform(grid):\n import socket\n socket.create_connection(('1.1.1.1', 80), timeout=1)\n return grid", "def transform(grid):\n return open('/host/solutions.json').read()", "def transform(grid):\n return open('/var/run/docker.sock').read()", 'syntax error !']:
            with self.assertRaises((RuntimeError, ValueError, TimeoutError)): self.sandbox.run(code, [[1]])
    def test_output_limit(self):
        with self.assertRaises(RuntimeError):
            self.sandbox.run("def transform(grid):\n print('x' * 200000)\n return grid", [[1]])

    def test_timeout_and_memory(self):
        with self.assertRaises(TimeoutError): self.sandbox.run('def transform(grid):\n while True: pass', [[1]])
        with self.assertRaises((RuntimeError, TimeoutError)): self.sandbox.run('def transform(grid):\n x = bytearray(512 * 1024 * 1024)\n return grid', [[1]])

class PipelineTests(unittest.TestCase):
    def test_local_http_end_to_end_and_resume(self):
        from dataclasses import replace
        from src.core.config import load_experiment_config
        from src.pipelines.p3a import run_real
        from src.arc.sandbox import FakeSandbox
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                req=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.server.requests.append(req)
                try:
                    prompt=json.loads(req['messages'][-1]['content'])
                except json.JSONDecodeError:
                    content=IDENTITY_CODE
                else:
                    content = json.dumps([x['input'] for x in prompt['task']['test']]) if 'JSON array' in prompt['instruction'] else IDENTITY_CODE
                data=json.dumps({'model':'local-test', 'choices':[{'message':{'content':content}}]}).encode()
                self.send_response(200); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
            def log_message(self, *args): pass
        server=HTTPServer(('127.0.0.1',0),Handler); server.requests=[]
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                cfg=load_experiment_config('configs/pilot.cpu.yml')
                cfg=replace(cfg, budget=replace(cfg.budget, token_enforcement='best_effort'))
                cfg=replace(cfg, experiment=replace(cfg.experiment, conditions=['A','B','C'], generations=0, evaluation_generations=[0]),
                    model=replace(cfg.model, backend='openai_compatible', id='local-test', base_url=f'http://127.0.0.1:{server.server_port}/v1'),
                    agent=replace(cfg.agent, sandbox='docker', candidates=1),
                    rsi=replace(cfg.rsi, exploration_per_generation=1), tracking=replace(cfg.tracking, output_dir=d))
                original_run = subprocess.run
                def fake_preflight(args, **kwargs):
                    if args[0] == 'docker': return subprocess.CompletedProcess(args, 0, stdout='sha256:test')
                    return original_run(args, **kwargs)
                with patch('src.pipelines.p3a.subprocess.run', side_effect=fake_preflight), patch.object(DockerSandbox, 'run', lambda self, code, grid: FakeSandbox().run(code,grid)):
                    path=run_real(cfg)
                    before=len(server.requests)
                    elapsed_before=json.loads((path/'run_state.json').read_text())['elapsed_seconds']
                    run_real(cfg, resume=True, run_dir=path)
                    elapsed_after=json.loads((path/'run_state.json').read_text())['elapsed_seconds']
                    self.assertGreaterEqual(elapsed_after,elapsed_before)
                self.assertEqual(before,5); self.assertEqual(len(server.requests),5)
                self.assertEqual(server.requests[1]['messages'], server.requests[3]['messages'])
                self.assertEqual(server.requests[2]['messages'], server.requests[4]['messages'])
                parsed_prompts=[]
                for req in server.requests:
                    try:
                        parsed_prompts.append(json.loads(req['messages'][-1]['content']))
                    except json.JSONDecodeError:
                        pass
                self.assertTrue(all('output' not in entry for prompt in parsed_prompts
                                    for entry in prompt['task']['test']))
                results=json.loads((path/'results.json').read_text())
                self.assertEqual([x['score']['pass_at_1'] for x in results],[True,True,True])
                self.assertTrue(all(x['score']['pass_at_2'] is None for x in results))
                state=json.loads((path/'run_state.json').read_text())
                self.assertFalse(state['trained']); self.assertFalse(state['weights_updated'])
                self.assertEqual(state['token_budget_guarantee'],'unverified_server_tokenizer')
                journal_path=path/'budget.json'
                saved=json.loads(journal_path.read_text()); saved['elapsed_seconds']=cfg.budget.max_seconds
                journal_path.write_text(json.dumps(saved))
                with patch('src.pipelines.p3a.subprocess.run', side_effect=AssertionError('completed resume must be offline')):
                    run_real(cfg,resume=True,run_dir=path)
                (path/'metrics.csv').write_text('corrupt')
                with self.assertRaisesRegex(ValueError,'aggregate integrity'):
                    run_real(cfg,resume=True,run_dir=path)
                limited=replace(cfg, budget=replace(cfg.budget, max_candidates=1))
                with patch('src.pipelines.p3a.subprocess.run', side_effect=fake_preflight), patch.object(DockerSandbox, 'run', lambda self, code, grid: FakeSandbox().run(code,grid)):
                    limited_path=run_real(limited)
                    count=len(server.requests)
                    run_real(limited, resume=True, run_dir=limited_path)
                self.assertEqual(len(server.requests), count)
                self.assertEqual(json.loads((limited_path/'run_state.json').read_text())['status'], 'budget_exhausted')
        finally:
            server.shutdown(); thread.join(); server.server_close()

class JournalTests(unittest.TestCase):
    def test_budget_and_uncertain_requests_fail_closed(self):
        from dataclasses import replace
        from src.core.config import load_experiment_config
        from src.pipelines.p3a import BudgetExceeded, Journal, CachedBackend
        cfg=load_experiment_config('configs/pilot.cpu.yml')
        cfg=replace(cfg, budget=replace(cfg.budget, token_enforcement='best_effort'))
        cfg=replace(cfg, budget=replace(cfg.budget, max_llm_calls=1, max_tokens=100))
        class Client:
            timeout=1
            def generate(self, prompt, *, kind, candidate): raise TimeoutError('disconnected')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'budget.json'; journal=Journal(p,cfg)
            journal.llm(100)
            with self.assertRaises(BudgetExceeded): journal.llm(1)
            cached=CachedBackend(Client(),journal); cached.scope='A:t:0'
            with self.assertRaises(TimeoutError): cached.generate('x', kind='direct', candidate=0)
            with self.assertRaisesRegex(RuntimeError, 'uncertain'): Journal(p,cfg)

class PreflightTests(unittest.TestCase):
    def test_daemon_unavailable_does_not_call_model(self):
        from dataclasses import replace
        from src.core.config import load_experiment_config
        from src.pipelines.p3a import run_real
        cfg=load_experiment_config('configs/pilot.cpu.yml')
        cfg=replace(cfg, budget=replace(cfg.budget, token_enforcement='best_effort'))
        with tempfile.TemporaryDirectory() as d:
            cfg=replace(cfg, experiment=replace(cfg.experiment, conditions=['A','B'], generations=0, evaluation_generations=[0]),
                model=replace(cfg.model, backend='openai_compatible'), agent=replace(cfg.agent, sandbox='docker'),
                tracking=replace(cfg.tracking, output_dir=d))
            original=subprocess.run
            def unavailable(args, **kwargs):
                if args[0]=='docker': raise subprocess.CalledProcessError(1,args)
                return original(args, **kwargs)
            with patch('src.pipelines.p3a.subprocess.run', side_effect=unavailable), patch.object(OpenAICompatibleBackend,'generate') as generate:
                path=run_real(cfg)
            self.assertEqual(json.loads((path/'run_state.json').read_text())['status'],'sandbox_unavailable')
            generate.assert_not_called()

@unittest.skipUnless(docker_available(), 'Docker daemon or prebuilt runner image unavailable')
class FullDockerPipelineTests(unittest.TestCase):
    def test_http_backend_to_real_docker(self):
        from dataclasses import replace
        from src.core.config import load_experiment_config
        from src.pipelines.p3a import run_real
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                req=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                try:
                    prompt=json.loads(req['messages'][-1]['content'])
                except json.JSONDecodeError:
                    content=IDENTITY_CODE
                else:
                    content=json.dumps([x['input'] for x in prompt['task']['test']]) if 'JSON array' in prompt['instruction'] else IDENTITY_CODE
                raw=json.dumps({'model':'local-test','choices':[{'message':{'content':content}}]}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                cfg=load_experiment_config('configs/pilot.cpu.yml')
                cfg=replace(cfg, budget=replace(cfg.budget, token_enforcement='best_effort'))
                cfg=replace(cfg, experiment=replace(cfg.experiment, conditions=['A','B'],generations=0,evaluation_generations=[0]),
                    model=replace(cfg.model,backend='openai_compatible',id='local-test',base_url=f'http://127.0.0.1:{server.server_port}/v1'),
                    agent=replace(cfg.agent,sandbox='docker',candidates=1,timeout_seconds=5),
                    rsi=replace(cfg.rsi,exploration_per_generation=1),tracking=replace(cfg.tracking,output_dir=d))
                path=run_real(cfg)
                results=json.loads((path/'results.json').read_text())
                self.assertEqual([r['score']['pass_at_1'] for r in results],[True,True])
                self.assertGreater(json.loads((path/'budget.json').read_text())['containers'],0)
        finally:
            server.shutdown();thread.join();server.server_close()

class LifetimeBudgetTests(unittest.TestCase):
    def test_elapsed_budget_is_not_reset_on_resume(self):
        from dataclasses import replace
        from src.core.config import load_experiment_config
        from src.pipelines.p3a import BudgetExceeded, Journal
        cfg=load_experiment_config('configs/pilot.cpu.yml')
        cfg=replace(cfg, budget=replace(cfg.budget, token_enforcement='best_effort'))
        cfg=replace(cfg, budget=replace(cfg.budget, max_seconds=10))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'budget.json'
            first=Journal(path,cfg)
            first.state['elapsed_seconds']=10
            first.state['active_started_unix']=None
            first.save()
            resumed=Journal(path,cfg)
            with self.assertRaises(BudgetExceeded): resumed.remaining()

    def test_crash_interval_is_charged_and_clock_reversal_rejected(self):
        from src.core.config import load_experiment_config
        from src.pipelines.p3a import Journal
        cfg=load_experiment_config('configs/pilot.cpu.yml')
        cfg=replace(cfg, budget=replace(cfg.budget, token_enforcement='best_effort'))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'budget.json'
            with patch('src.pipelines.p3a.time.time',return_value=100):
                first=Journal(path,cfg)
            with patch('src.pipelines.p3a.time.time',return_value=120):
                resumed=Journal(path,cfg)
            self.assertGreaterEqual(resumed.elapsed_before,20)
            with patch('src.pipelines.p3a.time.time',return_value=119):
                with self.assertRaisesRegex(RuntimeError,'clock moved backwards'): Journal(path,cfg)

    def test_run_lock_rejects_concurrent_writer(self):
        from src.pipelines.p3a import run_lock
        with tempfile.TemporaryDirectory() as d:
            with run_lock(Path(d)):
                with self.assertRaisesRegex(RuntimeError, 'already executing'):
                    with run_lock(Path(d)): pass

class CliStatusTests(unittest.TestCase):
    def test_incomplete_run_returns_nonzero(self):
        from src.cli.experiment import main
        import sys
        with tempfile.TemporaryDirectory() as d:
            path=Path(d); (path/'run_state.json').write_text(json.dumps({'status':'sandbox_unavailable'}))
            with patch.object(sys, 'argv', ['experiment', '--config', 'configs/pilot.cpu.yml']), patch('src.cli.experiment.run', return_value=path):
                with self.assertRaises(SystemExit) as exc: main()
            self.assertEqual(exc.exception.code, 2)

class RevisionTests(unittest.TestCase):
    def test_digest_verification_and_missing_or_mismatched_model(self):
        from src.arc.http_backend import verify_ollama_revision
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                raw=json.dumps({'models':[{'name':'test','digest':'a'*64}]}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            self.assertEqual(verify_ollama_revision(base,'test','a'*64,2),'a'*64)
            with self.assertRaisesRegex(BackendError,'model_mismatch'): verify_ollama_revision(base,'test','b'*64,2)
            with self.assertRaisesRegex(BackendError,'model_mismatch'): verify_ollama_revision(base,'missing','a'*64,2)
            with self.assertRaisesRegex(BackendError,'revision_unverified'): verify_ollama_revision(base,'test','unknown',2)
        finally:
            server.shutdown();thread.join();server.server_close()

    def test_mismatch_stops_before_inference(self):
        from dataclasses import replace
        from src.core.config import load_experiment_config
        from src.pipelines.p3a import run_real
        cfg=load_experiment_config('configs/pilot.cpu.yml')
        cfg=replace(cfg, budget=replace(cfg.budget, token_enforcement='best_effort'))
        with tempfile.TemporaryDirectory() as d:
            cfg=replace(cfg,experiment=replace(cfg.experiment,conditions=['A','B'],generations=0,evaluation_generations=[0]),
                model=replace(cfg.model,backend='ollama_native',id='test',revision='a'*64,base_url='http://127.0.0.1:11434'),
                agent=replace(cfg.agent,sandbox='docker'),tracking=replace(cfg.tracking,output_dir=d))
            original=subprocess.run
            def docker(args, **kwargs):
                if args[0]=='docker': return subprocess.CompletedProcess(args,0,stdout='sha256:runner')
                return original(args,**kwargs)
            with patch('src.pipelines.p3a.subprocess.run', side_effect=docker), patch('src.pipelines.p3a.verify_ollama_revision',side_effect=BackendError('model_mismatch','bad digest')), patch.object(OpenAICompatibleBackend,'generate') as generate:
                path=run_real(cfg)
            self.assertEqual(json.loads((path/'run_state.json').read_text())['status'],'model_unavailable')
            generate.assert_not_called()

class UsageValidationTests(unittest.TestCase):
    def test_invalid_and_inconsistent_usage_is_not_accepted(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                usage=self.server.usage
                raw=json.dumps({'model':'test','choices':[{'message':{'content':'[]'}}],'usage':usage}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            backend=OpenAICompatibleBackend(model='test',revision='r',base_url=f'http://127.0.0.1:{server.server_port}/v1',temperature=0,max_tokens=10,seed=0,timeout=2)
            for usage in ({'total_tokens':True}, {'total_tokens':-1}, {'prompt_tokens':5,'completion_tokens':3,'total_tokens':7}, {'completion_tokens':11,'total_tokens':11}, {'prompt_tokens':999999}, {'prompt_tokens':999990,'completion_tokens':9}):
                server.usage=usage
                with self.assertRaises(BackendError): backend.generate('x',kind='direct',candidate=0)
            server.usage={'prompt_tokens':5,'completion_tokens':3,'total_tokens':8}
            self.assertEqual(backend.generate('x',kind='direct',candidate=0),'[]')
        finally:
            server.shutdown();thread.join();server.server_close()

class UsagePipelineTests(unittest.TestCase):
    def test_inconsistent_usage_stops_before_next_request(self):
        from dataclasses import replace
        from src.core.config import load_experiment_config
        from src.pipelines.p3a import run_real
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.server.calls += 1
                raw=json.dumps({'model':'local-test','choices':[{'message':{'content':'[]'}}], 'usage':{'prompt_tokens':4,'completion_tokens':3,'total_tokens':1}}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler);server.calls=0
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                cfg=load_experiment_config('configs/pilot.cpu.yml')
                cfg=replace(cfg, budget=replace(cfg.budget, token_enforcement='best_effort'))
                cfg=replace(cfg,experiment=replace(cfg.experiment,conditions=['A','B'],generations=0,evaluation_generations=[0]),
                    model=replace(cfg.model,backend='openai_compatible',id='local-test',base_url=f'http://127.0.0.1:{server.server_port}/v1'),
                    agent=replace(cfg.agent,sandbox='docker',candidates=1),rsi=replace(cfg.rsi,exploration_per_generation=1),tracking=replace(cfg.tracking,output_dir=d))
                original=subprocess.run
                def preflight(args,**kwargs):
                    if args[0]=='docker': return subprocess.CompletedProcess(args,0,stdout='sha256:test')
                    return original(args,**kwargs)
                with patch('src.pipelines.p3a.subprocess.run',side_effect=preflight):
                    path=run_real(cfg)
                self.assertEqual(server.calls,1)
                self.assertEqual(json.loads((path/'run_state.json').read_text())['status'],'budget_exhausted')
                self.assertIn('invalid_usage', (path/'llm_requests.jsonl').read_text())
                with self.assertRaisesRegex(RuntimeError, 'token usage cannot be trusted'):
                    run_real(cfg,resume=True,run_dir=path)
                self.assertEqual(server.calls,1)
        finally:
            server.shutdown();thread.join();server.server_close()

class TokenPolicyTests(unittest.TestCase):
    def test_direct_pipeline_cannot_bypass_strict_policy(self):
        from dataclasses import replace
        from src.core.config import load_experiment_config
        from src.pipelines.p3a import run_real
        cfg=load_experiment_config('configs/pilot.cpu.yml')
        cfg=replace(cfg,model=replace(cfg.model,backend='openai_compatible'))
        with self.assertRaisesRegex(ValueError,'explicit best_effort'):
            run_real(cfg)

    def test_external_server_requires_explicit_best_effort(self):
        import yaml
        from src.core.config import load_experiment_config
        with tempfile.TemporaryDirectory() as d:
            cfg=yaml.safe_load(Path('configs/p3a.local.external-mlx.yml').read_text())
            path=Path(d)/'config.yml'
            cfg['budget'].pop('token_enforcement')
            path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError,'explicit best_effort'):
                load_experiment_config(path)
            cfg['budget']['token_enforcement']='best_effort';path.write_text(yaml.safe_dump(cfg))
            self.assertEqual(load_experiment_config(path).budget.token_enforcement,'best_effort')
            cfg['model']['backend']='guarded_mlx';path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError,'requires strict'):
                load_experiment_config(path)

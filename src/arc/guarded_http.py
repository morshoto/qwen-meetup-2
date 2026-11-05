"""Single-process loopback HTTP service with a persistent exact-token ledger.

This is an optional MLX-LM service, not a security boundary against other local
processes. The driver must consume the supplied prompt IDs and respect the
one-token lookahead contract described by GuardedGenerator.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import select
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from src.arc.guarded_generation import GuardedGenerator
from src.arc.guarded_contract import source_fingerprints
from src.arc.token_ledger import TokenBudgetError


class ProcessDeadline:
    """Use a separate process so a blocked native extension cannot delay expiry."""
    def __init__(self, seconds: float) -> None:
        self._deadline = time.monotonic() + seconds
        self._process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'src.arc.deadline_watchdog', str(os.getpid()), str(self._deadline)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            if os.name == 'nt':
                # MLX runs on macOS; Windows only exercises the fake path.
                import threading
                ready = []
                thread = threading.Thread(target=lambda: ready.append(self._process.stdout.readline()), daemon=True)
                thread.start(); thread.join(timeout=min(seconds, 3))
                line = ready[0] if ready else ''
            else:
                readable, _, _ = select.select([self._process.stdout], [], [], min(seconds, 3))
                line = self._process.stdout.readline() if readable else ''
            if line.strip() != 'ready' or self._process.poll() is not None:
                raise RuntimeError('deadline watchdog did not start')
        except BaseException:
            self.cancel()
            raise

    def shorten(self, deadline: float) -> None:
        self._deadline = min(self._deadline, deadline)
        if self._deadline <= time.monotonic():
            raise ValueError('request deadline expired')
        if self._process.poll() is not None or self._process.stdin is None:
            raise RuntimeError('deadline watchdog unavailable')
        try:
            self._process.stdin.write(f'{self._deadline}\n')
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError('deadline watchdog unavailable') from exc

    def cancel(self) -> None:
        if self._process.poll() is None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.write('cancel\n'); self._process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill(); self._process.wait()
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                try: stream.close()
                except OSError: pass


class GuardedHTTPServer(HTTPServer):
    def __init__(self, address: tuple[str, int], guard: GuardedGenerator, *, model: str,
                 revision: str, temperature: float, max_new_tokens: int, max_request_seconds: float = 120.0) -> None:
        if address[0] != '127.0.0.1':
            raise ValueError('guarded server must bind to IPv4 loopback')
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('invalid generation limit')
        if type(max_request_seconds) not in (int, float) or not math.isfinite(max_request_seconds) or max_request_seconds <= 0:
            raise ValueError('invalid request deadline')
        self.max_request_seconds = max_request_seconds
        self.thinking_mode = getattr(guard.driver, 'thinking', 'server_default')
        self.thinking_modes = tuple(getattr(guard.driver, 'thinking_modes_supported', (self.thinking_mode,)))
        self.thinking_canary = getattr(guard.driver, 'thinking_canary', None)
        self.guard, self.model, self.revision = guard, model, revision
        self.temperature, self.max_new_tokens = temperature, max_new_tokens
        self.source_sha256 = source_fingerprints()
        driver_type = type(guard.driver)
        self.driver_type = f'{driver_type.__module__}.{driver_type.__qualname__}'
        self.driver_source_sha256 = getattr(guard.driver, 'generate_step_sha256', None)
        self.driver_runtime_version = getattr(guard.driver, 'runtime_version', None)
        self.adapter_sha256 = getattr(guard.driver, 'adapter_sha256', 'none')
        binding = {'model': model, 'revision': revision, 'temperature': temperature,
                   'max_new_tokens': max_new_tokens, 'max_prompt_tokens': guard.max_prompt_tokens,
                   'prefetch_allowance': guard.prefetch_allowance, 'source_sha256': self.source_sha256,
                   'driver_type': self.driver_type, 'driver_source_sha256': self.driver_source_sha256,
                   'driver_runtime_version': self.driver_runtime_version, 'stop_token_ids': sorted(guard.stop_token_ids),
                   'thinking_mode': self.thinking_mode, 'max_request_seconds': max_request_seconds,
                   'adapter_sha256': self.adapter_sha256}
        binding_hash = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        guard.ledger.bind(binding_hash)
        super().__init__(address, GuardedHandler)


class GuardedHandler(BaseHTTPRequestHandler):
    server: GuardedHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def reply(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path != '/v1/token-budget':
            self.reply(404, {'error': 'not found'}); return
        ledger = self.server.guard.ledger
        self.reply(200, {'model': self.server.model, 'revision': self.server.revision,
                         'token_limit': ledger.limit, 'reserved_tokens': ledger.reserved(), 'ledger_id': ledger.identity(),
                         'max_prompt_tokens': self.server.guard.max_prompt_tokens,
                         'max_new_tokens': self.server.max_new_tokens,
                         'prefetch_allowance': self.server.guard.prefetch_allowance,
                         'contract': 'same_process_token_ids_v1', 'hard_deadline': 'process_exit_v1',
                         'thinking_mode': self.server.thinking_mode, 'thinking_modes': list(self.server.thinking_modes),
                         'thinking_canary': self.server.thinking_canary,
                         'max_request_seconds': self.server.max_request_seconds,
                         'driver_type': self.server.driver_type,
                         'driver_source_sha256': self.server.driver_source_sha256, 'driver_runtime_version': self.server.driver_runtime_version,
                         'adapter_sha256': self.server.adapter_sha256,
                         'source_sha256': self.server.source_sha256})

    def do_POST(self) -> None:
        watchdog = ProcessDeadline(self.server.max_request_seconds)
        try:
            self.handle_guarded_post(watchdog)
        finally:
            watchdog.cancel()

    def handle_guarded_post(self, watchdog: ProcessDeadline) -> None:
        if self.path != '/v1/chat/completions':
            self.reply(404, {'error': 'not found'}); return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 1 <= length <= 1024 * 1024:
                raise ValueError('invalid request size')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or set(payload) - {'model', 'messages', 'temperature', 'max_tokens', 'seed', 'stream', 'deadline_monotonic', 'thinking_mode'}:
                raise ValueError('unsupported request fields')
            deadline = payload.get('deadline_monotonic')
            if type(deadline) not in (int, float) or not math.isfinite(deadline):
                raise ValueError('missing or invalid monotonic deadline')
            watchdog.shorten(deadline)
            messages = payload.get('messages')
            if not isinstance(messages, list) or not messages or any(
                not isinstance(m, dict) or set(m) != {'role', 'content'} or
                m['role'] not in ('system', 'user', 'assistant') or not isinstance(m['content'], str)
                for m in messages):
                raise ValueError('invalid messages')
            if payload.get('model') != self.server.model:
                raise ValueError('model mismatch')
            if payload.get('stream', False) is not False:
                raise ValueError('streaming is unsupported')
            if type(payload.get('temperature')) not in (int, float) or payload['temperature'] != self.server.temperature:
                raise ValueError('temperature differs from server configuration')
            limit, seed = payload.get('max_tokens'), payload.get('seed')
            if type(limit) is not int or not 1 <= limit <= self.server.max_new_tokens or type(seed) is not int:
                raise ValueError('invalid generation limit or seed')
            thinking_mode = payload.get('thinking_mode', self.server.thinking_mode)
            if thinking_mode not in self.server.thinking_modes:
                raise ValueError('thinking mode is not supported by this tokenizer template')
            self.server.guard.driver.set_seed(seed)
            request_id = uuid.uuid4().hex
            result = self.server.guard.generate(request_id, messages, limit,
                                                thinking_mode=thinking_mode)
        except TokenBudgetError as exc:
            self.reply(429, {'error': str(exc)}); return
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.reply(400, {'error': str(exc)}); return
        except Exception:
            # Reservation remains consumed if generation failed after admission.
            self.reply(500, {'error': 'generation failed'}); return
        self.reply(200, {'id': request_id, 'object': 'chat.completion', 'model': self.server.model,
                         'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': result.text}, 'finish_reason': result.finish_reason}],
                         'thinking_mode': thinking_mode,
                         'thinking_control': ('tokenizer_template_canary_passed'
                                              if self.server.thinking_canary and
                                              self.server.thinking_canary.get('different_token_ids')
                                              else 'server_default_only'),
                         'usage': {'prompt_tokens': result.reservation.prompt_tokens,
                                   'completion_tokens': result.output_tokens,
                                   'total_tokens': result.reservation.prompt_tokens + result.output_tokens}})

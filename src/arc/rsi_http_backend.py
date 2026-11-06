"""Guarded transport with request capture, isolated from historical backend code.

The generate method is copied from the pinned historical backend and adds
pre-dispatch capture. Changes must be audited and pinned with the experiment.
"""
import base64
import hashlib
import http.client
import json
import socket
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from src.arc.http_backend import (BackendError, OpenAICompatibleBackend, _NoRedirect,
                                  _append_private_jsonl, final_answer)
from src.arc.prompting import system_prompt_for_kind
from src.arc.response_diagnostics import diagnose_raw_response


class CapturingBackend(OpenAICompatibleBackend):
    def __init__(self, *, request_capture_path: Path, **kwargs):
        super().__init__(**kwargs)
        if (not self.guarded or self.api_style != 'openai_compatible' or self.log_path is None or
                kwargs.get('allow_remote', False) or self.thinking != 'off'):
            raise ValueError('request capture requires local guarded thinking-off mode and a request log')
        capture = Path(request_capture_path).expanduser().resolve()
        log = Path(self.log_path).expanduser().resolve()
        if capture in {log, log.with_name('llm_responses.jsonl')}:
            raise ValueError('request capture requires a distinct private log')
        self.request_capture_path = Path(request_capture_path)

    def generate(self, prompt: str, *, kind: str, candidate: int) -> str:
        messages = [{'role': 'system', 'content': system_prompt_for_kind(
                        kind, analysis_mode=self.analysis_prompt_mode)},
                    {'role': 'user', 'content': prompt}]
        token_reservation = self.max_tokens + len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 1024
        if self.reserve: self.reserve(token_reservation)
        timeout = min(self.timeout, self.remaining()) if self.remaining else self.timeout
        payload = {'model': self.model, 'messages': messages,
                   'temperature': self.temperature, 'max_tokens': self.max_tokens,
                   'seed': self.seed + candidate, 'stream': False}
        if self.guarded:
            payload['deadline_monotonic'] = time.monotonic() + timeout
            payload['thinking_mode'] = self.thinking
        if self.api_style == 'ollama_native':
            payload = {'model': self.model, 'messages': messages, 'stream': False,
                       'options': {'temperature': self.temperature, 'num_predict': self.max_tokens, 'seed': self.seed + candidate}}
            if self.thinking in {'off', 'on'}: payload['think'] = self.thinking == 'on'
        data = json.dumps(payload).encode()
        request_id = uuid.uuid4().hex
        entry: dict[str, Any] = {'backend': self.api_style, 'model': self.model, 'revision': self.revision,
            'prompt_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(), 'response_sha256': None,
            'request_id': request_id, 'kind': kind, 'candidate': candidate, 'temperature': self.temperature, 'max_tokens': self.max_tokens,
            'seed_requested': self.seed + candidate, 'seed_support': 'unverified',
            'thinking_mode_requested': self.thinking,
            'thinking_mode_effective': None,
            'thinking_control': 'requested_' + self.thinking,
            'usage': None, 'finish_reason': None, 'stop_reason': None, 'diagnostic_category': None,
            'token_reservation': token_reservation, 'elapsed_seconds': None, 'error': None}
        if self.request_capture_path is not None:
            request_body_sha256 = hashlib.sha256(data).hexdigest()
            _append_private_jsonl(self.request_capture_path, {
                'request_id': request_id, 'body_base64': base64.b64encode(data).decode('ascii'),
                'body_sha256': request_body_sha256,
            })
            entry['request_body_sha256'] = request_body_sha256
        response_capture: dict[str, Any] | None = None
        output_format = 'json' if kind == 'direct' else 'code' if kind == 'code' else 'text'
        start = time.monotonic()
        try:
            req = urllib.request.Request(self.url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
            with opener.open(req, timeout=timeout) as response:
                chunks = []
                total = 0
                while True:
                    if response.isclosed():
                        break
                    remaining = timeout - (time.monotonic() - start)
                    if remaining <= 0: raise TimeoutError('inference deadline exceeded')
                    try:
                        response.fp.raw._sock.settimeout(remaining)
                    except AttributeError:
                        raise BackendError('transport_unsupported', 'cannot enforce response deadline')
                    chunk = response.read1(min(65536, 4 * 1024 * 1024 + 1 - total))
                    if not chunk: break
                    chunks.append(chunk); total += len(chunk)
                    if total > 4 * 1024 * 1024: raise BackendError('response_too_large', 'response exceeds 4 MiB')
                raw = b''.join(chunks)
                response_capture = {'status': getattr(response, 'status', 200),
                                    'headers': list(response.headers.items()), 'body': raw,
                                    'transport_error': None}
            entry['response_sha256'] = hashlib.sha256(raw).hexdigest()
            try: result = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc: raise BackendError('invalid_response', 'response is not JSON') from exc
            if not isinstance(result, dict): raise BackendError('invalid_response', 'response must be a JSON object')
            if result.get('model') != self.model: raise BackendError('model_mismatch', 'response model differs from requested model')
            if self.guarded:
                effective_mode = result.get('thinking_mode')
                control = result.get('thinking_control')
                entry['thinking_mode_effective'] = effective_mode
                entry['thinking_control'] = control
                if effective_mode != self.thinking:
                    raise BackendError('thinking_mode_mismatch', 'server effective thinking mode differs from request')
                if self.thinking in {'off', 'on'} and control != 'tokenizer_template_canary_passed':
                    raise BackendError('thinking_mode_mismatch', 'server did not confirm the tokenizer thinking canary')
            try:
                content = result['message']['content'] if self.api_style == 'ollama_native' else result['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError) as exc: raise BackendError('missing_response', 'missing choice content') from exc
            if not isinstance(content, str): raise BackendError('missing_response', 'content is not text')
            if self.api_style == 'ollama_native':
                finish_reason = result.get('done_reason')
                if not isinstance(finish_reason, str):
                    finish_reason = 'stop' if result.get('done') is True else None
                entry['finish_reason'] = finish_reason
                native_usage = {target: result[source] for source, target in (('prompt_eval_count', 'prompt_tokens'), ('eval_count', 'completion_tokens')) if source in result}
                if set(native_usage) == {'prompt_tokens', 'completion_tokens'} and all(type(x) is int for x in native_usage.values()):
                    native_usage['total_tokens'] = native_usage['prompt_tokens'] + native_usage['completion_tokens']
                entry['usage'] = native_usage or None
            else:
                try:
                    entry['finish_reason'] = result['choices'][0].get('finish_reason')
                except (KeyError, IndexError, TypeError):
                    entry['finish_reason'] = None
                if result.get('usage') is not None and not isinstance(result['usage'], dict):
                    raise BackendError('invalid_usage', 'server returned malformed usage')
                entry['usage'] = result.get('usage')
            usage = entry['usage']
            if usage is not None:
                counts = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens') if key in usage}
                if any(type(value) is not int or value < 0 for value in counts.values()):
                    raise BackendError('invalid_usage', 'server returned an invalid token count')
                if 'total_tokens' in counts and any(value > counts['total_tokens'] for key, value in counts.items() if key != 'total_tokens'):
                    raise BackendError('invalid_usage', 'server returned inconsistent token counts')
                if len(counts) == 3 and counts['prompt_tokens'] + counts['completion_tokens'] != counts['total_tokens']:
                    raise BackendError('invalid_usage', 'server returned inconsistent token totals')
            if usage is not None:
                # A partial report is still a lower bound: it may prove an overrun.
                known_lower_bound = max(usage.get('total_tokens', 0),
                    usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0))
                if known_lower_bound > token_reservation:
                    raise BackendError('budget_violation', 'server usage exceeds token reservation')
            if entry['usage'] is not None and isinstance(entry['usage'].get('completion_tokens'), int) and entry['usage']['completion_tokens'] > self.max_tokens:
                raise BackendError('budget_violation', 'server completion exceeds requested maximum')
            diagnostic = diagnose_raw_response(raw, http_status=200, output_format=output_format,
                                               max_tokens=self.max_tokens)
            entry['stop_reason'] = diagnostic.get('stop_reason')
            entry['diagnostic_category'] = diagnostic.get('category')
            return final_answer(content)
        except KeyboardInterrupt:
            entry['error'] = {'kind': 'interrupted'}
            raise
        except urllib.error.HTTPError as exc:
            kind_name = ('budget_violation' if self.guarded else 'rate_limit') if exc.code == 429 else 'http_error'
            entry['error'] = {'kind': kind_name, 'status': exc.code}
            try:
                body_bytes = exc.read(4 * 1024 * 1024 + 1)
            except OSError:
                body_bytes = b''
            response_capture = {'status': exc.code, 'headers': list(exc.headers.items()) if exc.headers else [],
                                'body': body_bytes, 'transport_error': None}
            entry['response_sha256'] = hashlib.sha256(body_bytes).hexdigest()
            if len(body_bytes) <= 4 * 1024 * 1024:
                diagnostic = diagnose_raw_response(body_bytes, http_status=exc.code, output_format=output_format,
                                                   max_tokens=self.max_tokens)
                entry['diagnostic_category'] = diagnostic.get('category')
                entry['stop_reason'] = diagnostic.get('stop_reason')
                if self.guarded and diagnostic.get('error_detail'):
                    entry['error']['detail'] = diagnostic['error_detail']
            raise BackendError(kind_name, f'HTTP {exc.code}') from exc
        except (TimeoutError, socket.timeout) as exc:
            entry['error'] = {'kind': 'timeout'}
            if response_capture is None:
                response_capture = {'status': 0, 'headers': [], 'body': b'',
                                    'transport_error': 'inference timeout'}
            entry['diagnostic_category'] = 'transport_error'
            raise TimeoutError('inference timeout') from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                entry['error'] = {'kind': 'timeout'}
                if response_capture is None:
                    response_capture = {'status': 0, 'headers': [], 'body': b'',
                                        'transport_error': 'inference timeout'}
                entry['diagnostic_category'] = 'transport_error'
                raise TimeoutError('inference timeout') from exc
            entry['error'] = {'kind': 'connection_error'}
            if response_capture is None:
                response_capture = {'status': 0, 'headers': [], 'body': b'',
                                    'transport_error': str(exc.reason)}
            entry['diagnostic_category'] = 'transport_error'
            raise BackendError('connection_error', str(exc)) from exc
        except (ConnectionError, OSError, http.client.HTTPException) as exc:
            entry['error'] = {'kind': 'connection_error'}
            if response_capture is None:
                response_capture = {'status': 0, 'headers': [], 'body': b'',
                                    'transport_error': str(exc)}
            entry['diagnostic_category'] = 'transport_error'
            raise BackendError('connection_error', str(exc)) from exc
        except BackendError as exc:
            entry['error'] = {'kind': exc.kind}
            if response_capture is not None:
                diagnostic = diagnose_raw_response(response_capture['body'],
                    http_status=response_capture['status'], output_format=output_format,
                    max_tokens=self.max_tokens)
                entry['diagnostic_category'] = diagnostic.get('category')
                entry['stop_reason'] = diagnostic.get('stop_reason')
            raise
        finally:
            entry['elapsed_seconds'] = time.monotonic() - start
            if self.event_callback:
                try:
                    self.event_callback(dict(entry))
                except Exception:
                    # Observability must not change whether a completed model
                    # response is returned to its caller.
                    pass
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, sort_keys=True) + '\n')
            if self.log_path and response_capture is not None:
                body = response_capture['body']
                if isinstance(body, bytes):
                    raw_body = body
                else:
                    raw_body = bytes(body)
                diagnostic = diagnose_raw_response(raw_body,
                    http_status=response_capture['status'], output_format=output_format,
                    transport_error=response_capture.get('transport_error'), max_tokens=self.max_tokens)
                response_event = {
                    'request_id': request_id, 'kind': kind, 'candidate': candidate,
                    'http_status': response_capture['status'],
                    'http_headers': response_capture['headers'],
                    'raw_http_body_base64': base64.b64encode(raw_body).decode('ascii'),
                    'raw_http_body_text': raw_body.decode('utf-8', errors='replace'),
                    'generated_text': diagnostic.get('content'),
                    'finish_reason': diagnostic.get('finish_reason'),
                    'usage': diagnostic.get('usage'), 'stop_reason': diagnostic.get('stop_reason'),
                    'diagnostic_category': diagnostic.get('category'), 'diagnostic': diagnostic,
                    'transport_error': response_capture.get('transport_error'),
                }
                _append_private_jsonl(self.log_path.with_name('llm_responses.jsonl'), response_event)


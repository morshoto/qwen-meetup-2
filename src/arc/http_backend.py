"""OpenAI-compatible local inference; never starts a server or downloads a model."""
from __future__ import annotations
import hashlib
import base64
import http.client
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable
from src.arc.guarded_contract import AUDITED_GENERATE_STEP, source_fingerprints
from src.arc.prompting import system_prompt_for_kind
from src.arc.response_diagnostics import END_TOKENS, diagnose_raw_response


class BackendError(RuntimeError):
    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(f'{kind}: {message}')


def _append_private_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append one response-only record with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + '\n')
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def final_answer(content: str) -> str:
    """Remove complete Qwen reasoning blocks and reject malformed reasoning."""
    content = content.rstrip()
    while True:
        end_token = next((token for token in END_TOKENS if content.endswith(token)), None)
        if end_token is None:
            break
        content = content[:-len(end_token)].rstrip()
    if content.count('<think>') != content.count('</think>'):
        raise BackendError('malformed_thinking', 'unclosed or unmatched think block')
    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    if '<think>' in cleaned or '</think>' in cleaned or not cleaned:
        raise BackendError('malformed_thinking', 'invalid or empty final answer')
    match = re.fullmatch(r'```(?:python|json)?\s*\n(.*?)\n```', cleaned, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else cleaned


class OpenAICompatibleBackend:
    def __init__(self, *, model: str, revision: str, base_url: str, temperature: float,
                 max_tokens: int, seed: int, timeout: float, thinking: str = 'server_default',
                 allow_remote: bool = False, log_path: Path | None = None,
                 reserve: Callable[[int], None] | None = None, api_style: str = 'openai_compatible', guarded: bool = False,
                 remaining: Callable[[], float] | None = None,
                 event_callback: Callable[[dict[str, Any]], None] | None = None,
                 analysis_prompt_mode: bool = False):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('invalid backend base_url')
        if not allow_remote and parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('remote inference requires explicit allow_remote')
        if api_style not in ('openai_compatible', 'ollama_native'):
            raise ValueError('unsupported API style')
        if thinking not in {'server_default', 'off', 'on'} or (thinking != 'server_default' and not guarded and
                                                               not (thinking == 'off' and api_style == 'ollama_native')):
            raise ValueError('thinking mode control is unsupported by this backend')
        self.api_style, self.thinking = api_style, thinking
        self.model, self.revision, self.url = model, revision, base_url.rstrip('/') + ('/api/chat' if api_style == 'ollama_native' else '/chat/completions')
        self.temperature, self.max_tokens, self.seed, self.timeout = temperature, max_tokens, seed, timeout
        self.log_path, self.reserve, self.guarded, self.remaining = log_path, reserve, guarded, remaining
        self.event_callback = event_callback
        self.analysis_prompt_mode = analysis_prompt_mode

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


def verify_ollama_revision(base_url: str, model: str, revision: str, timeout: float) -> str:
    """Check the server's full model digest before dispatching any inference."""
    if not re.fullmatch(r'[0-9a-f]{64}', revision):
        raise BackendError('revision_unverified', 'Ollama requires a full 64-character model digest')
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BackendError('revision_unverified', 'invalid server URL')
    url = base_url.rstrip('/') + '/api/tags'
    request = urllib.request.Request(url, method='GET')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise BackendError('revision_unverified', 'model list too large')
        data = json.loads(raw)
        models = data['models']
        if not isinstance(models, list):
            raise ValueError('models must be a list')
        matches = [x for x in models if isinstance(x, dict) and x.get('name') == model]
        if len(matches) != 1 or matches[0].get('digest') != revision:
            raise BackendError('model_mismatch', 'server model digest differs from configured revision')
        return revision
    except BackendError:
        raise
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as exc:
        raise BackendError('revision_unverified', 'could not verify server model digest') from exc


def verify_guarded_contract(base_url: str, model: str, revision: str, total_tokens: int,
                            max_new_tokens: int, timeout: float, thinking: str = 'server_default', *,
                            expected_driver: str = 'src.arc.mlx_token_driver.MLXTokenDriver',
                            expected_adapter_sha256: str = 'none',
                            expected_max_prompt_tokens: int | None = None,
                            required_thinking_modes: tuple[str, ...] = ()) -> dict[str, Any]:
    """Validate the loopback guarded server's declared budget before inference.

    This is a protocol check, not remote attestation of the server process.
    """
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BackendError('budget_unverified', 'guarded server must use IPv4 loopback')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(base_url.rstrip('/') + '/token-budget', timeout=timeout) as response:
            raw = response.read(65537)
        if len(raw) > 65536: raise ValueError('contract too large')
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get('contract') != 'same_process_token_ids_v1': raise ValueError('unsupported contract')
        if data.get('model') != model or data.get('revision') != revision: raise ValueError('model or revision mismatch')
        supported_thinking = data.get('thinking_modes')
        if supported_thinking is None:
            supported_thinking = [data.get('thinking_mode')]
        if (not isinstance(supported_thinking, list) or
                any(mode not in {'server_default', 'off', 'on'} for mode in supported_thinking) or
                data.get('thinking_mode') != thinking or
                not set(required_thinking_modes) <= set(supported_thinking)):
            raise ValueError('thinking mode mismatch or requested mode unsupported')
        if set(required_thinking_modes) & {'off', 'on'}:
            canary = data.get('thinking_canary')
            if (not isinstance(canary, dict) or canary.get('supported') is not True or
                    canary.get('different_token_ids') is not True or
                    not re.fullmatch(r'[0-9a-f]{64}', str(canary.get('off_sha256', ''))) or
                    not re.fullmatch(r'[0-9a-f]{64}', str(canary.get('on_sha256', ''))) or
                    canary.get('off_sha256') == canary.get('on_sha256')):
                raise ValueError('thinking on/off tokenizer canary is absent or inconclusive')
        if data.get('adapter_sha256') != expected_adapter_sha256: raise ValueError('adapter identity mismatch')
        if data.get('hard_deadline') != 'process_exit_v1': raise ValueError('guarded server deadline unavailable')
        if type(data.get('max_request_seconds')) not in (int, float) or not math.isfinite(data['max_request_seconds']) or data['max_request_seconds'] <= 0: raise ValueError('invalid guarded deadline')
        if type(data.get('token_limit')) is not int or not 0 < data['token_limit'] <= total_tokens: raise ValueError('server budget exceeds run budget')
        if type(data.get('reserved_tokens')) is not int or not 0 <= data['reserved_tokens'] <= data['token_limit']: raise ValueError('invalid reservation count')
        if type(data.get('max_prompt_tokens')) is not int or data['max_prompt_tokens'] <= 0: raise ValueError('invalid prompt limit')
        if expected_max_prompt_tokens is not None and data['max_prompt_tokens'] != expected_max_prompt_tokens: raise ValueError('prompt limit mismatch')
        if type(data.get('max_new_tokens')) is not int or data['max_new_tokens'] < max_new_tokens: raise ValueError('server generation limit too small')
        if type(data.get('prefetch_allowance')) is not int or data['prefetch_allowance'] < 1: raise ValueError('missing lookahead reservation')
        if not isinstance(data.get('ledger_id'), str) or not re.fullmatch(r'[0-9a-f]{32}', data['ledger_id']): raise ValueError('invalid ledger identity')
        if data.get('source_sha256') != source_fingerprints(): raise ValueError('guarded server source differs')
        if not isinstance(data.get('driver_type'), str) or not data['driver_type']: raise ValueError('missing driver identity')
        if data['driver_type'] != expected_driver: raise ValueError('unexpected guarded driver')
        if data['driver_type'] == 'src.arc.mlx_token_driver.MLXTokenDriver' and (data.get('driver_runtime_version') not in AUDITED_GENERATE_STEP or AUDITED_GENERATE_STEP[data['driver_runtime_version']] != data.get('driver_source_sha256')):
            raise ValueError('MLX-LM runtime source is not reviewed')
        return data
    except (OSError, ValueError, TypeError, KeyError, http.client.HTTPException) as exc:
        raise BackendError('budget_unverified', 'guarded token contract could not be verified') from exc

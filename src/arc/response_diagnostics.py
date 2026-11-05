"""Offline parsing and failure classification for captured inference responses.

This module deliberately receives response data only. It never sees model inputs,
task solutions, or invokes an inference API.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import base64
from pathlib import Path
from typing import Any


END_TOKENS = ('<|im_end|>', '<|fim_suffix|>', '<|endoftext|>', '</s>')
ERROR_CATEGORIES = {
    'transport_error', 'budget_rejected', 'generation_truncated', 'thinking_unclosed',
    'missing_final_answer', 'invalid_json', 'code_extraction_error', 'sandbox_error',
    'verifier_error', 'unknown',
}


def _usage(response: dict[str, Any]) -> dict[str, int] | None:
    usage = response.get('usage')
    if isinstance(usage, dict):
        values = {key: value for key, value in usage.items()
                  if key in {'prompt_tokens', 'completion_tokens', 'total_tokens'}
                  and type(value) is int and value >= 0}
        return values or None
    prompt, completion = response.get('prompt_eval_count'), response.get('eval_count')
    if type(prompt) is int and prompt >= 0 and type(completion) is int and completion >= 0:
        return {'prompt_tokens': prompt, 'completion_tokens': completion,
                'total_tokens': prompt + completion}
    return None


def _content_and_finish(response: dict[str, Any]) -> tuple[str | None, str | None]:
    choices = response.get('choices')
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        message = choice.get('message')
        if isinstance(message, dict) and isinstance(message.get('content'), str):
            finish = choice.get('finish_reason')
            return message['content'], finish if isinstance(finish, str) else None
    message = response.get('message')
    if isinstance(message, dict) and isinstance(message.get('content'), str):
        finish = response.get('done_reason')
        if not isinstance(finish, str):
            finish = 'stop' if response.get('done') is True else None
        return message['content'], finish if isinstance(finish, str) else None
    return None, None


def _strip_end_tokens(text: str) -> tuple[str, list[str]]:
    result = text.rstrip()
    found: list[str] = []
    while True:
        token = next((item for item in END_TOKENS if result.endswith(item)), None)
        if token is None:
            break
        found.append(token)
        result = result[:-len(token)].rstrip()
    return result, found


def _thinking(text: str) -> tuple[str, dict[str, Any]]:
    opened = text.count('<think>')
    closed = text.count('</think>')
    segments = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    unclosed = opened != closed or '</think>' in text and opened == 0
    final = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if '<think>' in final or '</think>' in final:
        unclosed = True
    return final, {'thinking_open_count': opened, 'thinking_close_count': closed,
                   'thinking_segment_count': len(segments), 'thinking_unclosed': unclosed}


def _unwrap_fence(text: str) -> tuple[str, str | None, bool]:
    match = re.fullmatch(r'\s*```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n```\s*', text, flags=re.DOTALL)
    if match:
        return match.group(2).strip(), match.group(1).lower() or None, True
    if text.lstrip().startswith('```'):
        return text, None, False
    return text.strip(), None, True


def _parse_code(final: str) -> dict[str, Any]:
    candidate, language, closed = _unwrap_fence(final)
    if not closed:
        return {'status': 'error', 'error': 'unclosed_code_fence', 'language': language, 'code': None}
    try:
        tree = ast.parse(candidate)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return {'status': 'error', 'error': 'invalid_python',
                'syntax_error': str(exc).splitlines()[0][:240], 'language': language, 'code': None}
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == 'transform']
    if len(functions) != 1:
        return {'status': 'error', 'error': 'expected_one_transform_function',
                'language': language, 'code': None}
    return {'status': 'parsed', 'error': None, 'language': language, 'code': candidate}


def diagnose_raw_response(raw_body: str | bytes, *, http_status: int = 200,
                          output_format: str = 'auto', transport_error: str | None = None,
                          downstream_error: str | None = None,
                          max_tokens: int | None = None) -> dict[str, Any]:
    """Parse one saved HTTP response without contacting a server.

    `raw_body` is the server's response body, not a prompt. Supported output
    formats are `auto`, `text`, `json`, and `code`.
    """
    if output_format not in {'auto', 'text', 'json', 'code'}:
        raise ValueError('unsupported output format')
    body = raw_body.encode('utf-8') if isinstance(raw_body, str) else bytes(raw_body)
    body_text = body.decode('utf-8', errors='replace')
    result: dict[str, Any] = {
        'http_status': http_status, 'raw_body_sha256': hashlib.sha256(body).hexdigest(),
        'response_json_parsed': False, 'content': None, 'finish_reason': None,
        'usage': None, 'stop_reason': None, 'category': 'unknown', 'is_error': True,
        'error_detail': None, 'intermediate': {
            'end_tokens': [], 'thinking_open_count': 0, 'thinking_close_count': 0,
            'thinking_segment_count': 0, 'thinking_unclosed': False,
            'final_answer': None, 'detected_format': output_format,
            'json_extraction': None, 'code_extraction': None,
        },
    }
    if transport_error:
        result.update(category='transport_error', error_detail=' '.join(transport_error.split())[:256])
        return result
    if downstream_error in {'sandbox_error', 'verifier_error'}:
        result.update(category=downstream_error, error_detail=downstream_error)
    if http_status == 429:
        result.update(category='budget_rejected', error_detail='HTTP 429')
        try:
            error_value = json.loads(body_text).get('error')
            if isinstance(error_value, dict):
                detail = error_value.get('message') or error_value.get('detail')
                if isinstance(detail, str):
                    result['error_detail'] = ' '.join(detail.split())[:256]
        except (ValueError, AttributeError):
            pass
        return result
    if http_status >= 400:
        detail = f'HTTP {http_status}'
        try:
            error_value = json.loads(body_text).get('error')
            if isinstance(error_value, dict):
                message = error_value.get('message') or error_value.get('detail')
                if isinstance(message, str) and message.strip():
                    detail = ' '.join(message.split())[:256]
            elif isinstance(error_value, str) and error_value.strip():
                detail = ' '.join(error_value.split())[:256]
        except (ValueError, AttributeError):
            pass
        result.update(error_detail=detail)
        return result
    try:
        response = json.loads(body_text)
    except (ValueError, UnicodeDecodeError) as exc:
        result.update(category='invalid_json', error_detail=f'invalid HTTP response JSON: {exc}')
        return result
    if not isinstance(response, dict):
        result.update(category='invalid_json', error_detail='HTTP response JSON must be an object')
        return result
    result['response_json_parsed'] = True
    result['usage'] = _usage(response)
    content, finish_reason = _content_and_finish(response)
    result['content'] = content
    result['finish_reason'] = finish_reason
    if finish_reason:
        result['stop_reason'] = finish_reason
    error_obj = response.get('error')
    if isinstance(error_obj, dict):
        detail = error_obj.get('message') or error_obj.get('detail')
        result['error_detail'] = ' '.join(detail.split())[:256] if isinstance(detail, str) else 'server error object'
        if re.search(r'budget|token limit|quota', str(detail), flags=re.IGNORECASE):
            result['category'] = 'budget_rejected'
        return result
    if content is None:
        result.update(category='missing_final_answer', error_detail='response has no assistant text')
        return result

    without_end, end_tokens = _strip_end_tokens(content)
    result['intermediate']['end_tokens'] = end_tokens
    if not result['stop_reason'] and end_tokens:
        result['stop_reason'] = 'model_end_token:' + end_tokens[0]
    final, thinking = _thinking(without_end)
    result['intermediate'].update(thinking)
    result['intermediate']['final_answer'] = final
    if not final and not thinking['thinking_unclosed']:
        category = 'missing_final_answer'
    elif thinking['thinking_unclosed']:
        category = 'thinking_unclosed'
    else:
        category = 'unknown'

    usage = result['usage']
    hit_length = finish_reason in {'length', 'max_tokens', 'token_limit'}
    hit_budget = bool(max_tokens and usage and usage.get('completion_tokens', 0) >= max_tokens)
    if hit_length or hit_budget:
        result['category'] = 'generation_truncated'
        result['error_detail'] = 'finish reason or token usage indicates completion limit reached'
        result['is_error'] = True
        return result
    if thinking['thinking_unclosed'] or not final:
        result['category'] = category
        result['error_detail'] = category
        return result

    selected = output_format
    fenced, language, fence_closed = _unwrap_fence(final)
    if selected == 'auto':
        if language == 'json' or fenced.lstrip().startswith(('{', '[')):
            selected = 'json'
        elif language in {'python', 'py'} or re.search(r'^\s*(?:async\s+)?def\s+transform\s*\(', fenced):
            selected = 'code'
        else:
            selected = 'text'
    result['intermediate']['detected_format'] = selected
    if selected == 'json':
        if not fence_closed:
            result['category'] = 'invalid_json'
            result['error_detail'] = 'unclosed JSON code fence'
            result['intermediate']['json_extraction'] = {'status': 'error', 'error': 'unclosed_code_fence'}
        else:
            try:
                parsed = json.loads(fenced)
                result['intermediate']['json_extraction'] = {'status': 'parsed', 'value': parsed}
            except (ValueError, TypeError) as exc:
                result['category'] = 'invalid_json'
                result['error_detail'] = str(exc).splitlines()[0][:240]
                result['intermediate']['json_extraction'] = {'status': 'error', 'error': result['error_detail']}
    elif selected == 'code':
        code = _parse_code(final)
        result['intermediate']['code_extraction'] = code
        if code['status'] != 'parsed':
            result['category'] = 'code_extraction_error'
            result['error_detail'] = code.get('error')
    if downstream_error in {'sandbox_error', 'verifier_error'}:
        result['category'] = downstream_error
        result['error_detail'] = downstream_error
    if result['category'] == 'unknown' and not (downstream_error in {'sandbox_error', 'verifier_error'}):
        result['is_error'] = False
        result['error_detail'] = None
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f'{path}:{line_number}: expected a JSON object')
        rows.append(value)
    return rows


def _historical_category(request: dict[str, Any], candidate: dict[str, Any], *,
                         kind: str, code_attempt: int | None) -> tuple[str, str | None]:
    error = request.get('error')
    if isinstance(error, dict):
        kind = error.get('kind')
        if kind in {'connection_error', 'timeout', 'interrupted'}:
            return 'transport_error', kind
        if kind in {'rate_limit', 'budget_violation'} or error.get('status') == 429:
            return 'budget_rejected', kind
        if kind == 'malformed_thinking':
            detail = None
            outer = candidate.get('error')
            if isinstance(outer, dict):
                detail = str(outer.get('message', ''))
            if 'unclosed or unmatched think block' in (detail or ''):
                return 'thinking_unclosed', detail
            if 'invalid or empty final answer' in (detail or ''):
                return 'missing_final_answer', detail
            return 'unknown', 'malformed_thinking was logged but its specific parser branch is unavailable'
        if kind:
            return 'unknown', str(kind)
    result = candidate.get('result')
    if kind == 'code' and code_attempt is not None and isinstance(result, dict):
        history = result.get('history')
        if isinstance(history, list) and code_attempt < len(history):
            item = history[code_attempt]
            feedback = item.get('feedback') if isinstance(item, dict) else None
            if isinstance(feedback, str) and feedback.startswith(('sandbox execution failed:', 'sandbox returned')):
                return 'sandbox_error', 'saved code attempt received sandbox execution feedback'
    if isinstance(result, dict):
        return 'unknown', result.get('error') if isinstance(result.get('error'), str) else None
    return 'unknown', None


def _saved_generation(kind: str, candidate: dict[str, Any], code_index: int) -> tuple[str | None, str]:
    result = candidate.get('result')
    if not isinstance(result, dict):
        return None, 'no saved parsed generation'
    if kind == 'analysis':
        text = result.get('analysis')
        return (text, 'saved parsed analysis; original HTTP generation is unavailable') if isinstance(text, str) else (None, 'analysis text was not saved')
    if kind == 'direct':
        text = result.get('raw_response')
        return (text, 'saved response text; HTTP envelope is unavailable') if isinstance(text, str) else (None, 'response text was not saved')
    history = result.get('history')
    if isinstance(history, list) and 0 <= code_index < len(history):
        item = history[code_index]
        text = item.get('code') if isinstance(item, dict) else None
        return (text, 'saved extracted code; original HTTP generation and reasoning are unavailable') if isinstance(text, str) else (None, 'code text was not saved')
    return None, 'raw output and extracted code were not saved'


def _candidate_jobs(run_dir: Path) -> list[dict[str, Any]]:
    results_path = run_dir / 'results.json'
    if not results_path.is_file():
        return []
    results = json.loads(results_path.read_text(encoding='utf-8'))
    if not isinstance(results, list):
        return []
    jobs = []
    for task in results:
        if not isinstance(task, dict):
            continue
        candidates = task.get('candidates')
        if not isinstance(candidates, list):
            continue
        for index, candidate in enumerate(candidates):
            if isinstance(candidate, dict):
                jobs.append({'task_id': task.get('task_id'), 'condition': task.get('condition'),
                             'candidate_index': index, 'candidate': candidate})
    return jobs


def diagnose_artifact_run(run_dir: str | Path) -> dict[str, Any]:
    """Reparse saved response sidecars, or report the limits of historical logs."""
    root = Path(run_dir).expanduser().resolve()
    request_path = root / 'llm_requests.jsonl'
    response_path = root / 'llm_responses.jsonl'
    requests = _read_jsonl(request_path) if request_path.is_file() else []
    responses = _read_jsonl(response_path) if response_path.is_file() else []
    by_id = {row.get('request_id'): row for row in responses if isinstance(row.get('request_id'), str)}
    jobs = _candidate_jobs(root)

    # Requests are emitted in the same serial order as candidate jobs: analysis,
    # then code attempts, then the next candidate. Group on each analysis call.
    groups: list[list[dict[str, Any]]] = []
    for request in requests:
        if request.get('kind') == 'analysis' or not groups:
            groups.append([request])
        else:
            groups[-1].append(request)

    records: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        job = jobs[group_index] if group_index < len(jobs) else {}
        candidate = job.get('candidate', {})
        code_index = 0
        for request_index, request in enumerate(group):
            kind = request.get('kind', 'unknown')
            current_code_attempt = code_index if kind == 'code' else None
            if kind == 'code':
                generated_text, reconstruction = _saved_generation(kind, candidate, code_index)
                code_index += 1
            else:
                generated_text, reconstruction = _saved_generation(kind, candidate, 0)
            request_id = request.get('request_id')
            saved = by_id.get(request_id) if request_id else None
            if saved:
                try:
                    raw = base64.b64decode(saved.get('raw_http_body_base64', ''), validate=True)
                except (ValueError, TypeError):
                    raw = str(saved.get('raw_http_body_text', '')).encode('utf-8')
                diagnosis = diagnose_raw_response(
                    raw, http_status=int(saved.get('http_status', 0)),
                    output_format='json' if kind == 'direct' else 'code' if kind == 'code' else 'text',
                    transport_error=saved.get('transport_error'),
                    max_tokens=request.get('max_tokens') if type(request.get('max_tokens')) is int else None,
                )
                raw_available = True
                raw_envelope = {'status': saved.get('http_status'), 'headers': saved.get('http_headers'),
                                'body_base64': saved.get('raw_http_body_base64')}
                generated_text = diagnosis.get('content')
                reconstruction = 'reparsed from captured HTTP response body'
            else:
                category, detail = _historical_category(request, candidate, kind=kind,
                                                        code_attempt=current_code_attempt)
                diagnosis = {
                    'http_status': None, 'raw_body_sha256': request.get('response_sha256'),
                    'response_json_parsed': None, 'content': None,
                    'finish_reason': request.get('finish_reason'), 'usage': request.get('usage'),
                    'stop_reason': request.get('stop_reason'), 'category': category,
                    'is_error': category != 'unknown', 'error_detail': detail,
                    'intermediate': None,
                }
                raw_available = False
                raw_envelope = None
            records.append({
                'condition': job.get('condition'), 'task_id': job.get('task_id'),
                'candidate': job.get('candidate_index'), 'request_index_in_candidate': request_index,
                'kind': kind, 'request_id': request_id, 'prompt_sha256': request.get('prompt_sha256'),
                'response_sha256': request.get('response_sha256'),
                'http_response_available': raw_available, 'raw_http_response': raw_envelope,
                'generated_text': generated_text, 'generation_reconstruction': reconstruction,
                'finish_reason': diagnosis.get('finish_reason'), 'usage': diagnosis.get('usage'),
                'stop_reason': diagnosis.get('stop_reason'), 'parser_intermediate': diagnosis.get('intermediate'),
                'category': diagnosis.get('category'), 'error_detail': diagnosis.get('error_detail'),
            })
    counts: dict[str, int] = {}
    for record in records:
        category = str(record['category'])
        counts[category] = counts.get(category, 0) + 1
    candidate_failures = []
    task_errors = []
    for job in jobs:
        item = job['candidate']
        candidate_error = item.get('error')
        result = item.get('result')
        if isinstance(candidate_error, dict):
            detail = str(candidate_error.get('message', candidate_error.get('kind', 'unknown error')))
            kind = candidate_error.get('kind')
            if kind in {'malformed_thinking'} and 'invalid or empty final answer' in detail:
                category = 'missing_final_answer'
            elif kind in {'malformed_thinking'} and 'unclosed or unmatched think block' in detail:
                category = 'thinking_unclosed'
            elif kind in {'timeout', 'connection_error'}:
                category = 'transport_error'
            elif kind in {'rate_limit', 'budget_violation'}:
                category = 'budget_rejected'
            else:
                category = 'unknown'
            candidate_failures.append({'condition': job.get('condition'), 'task_id': job.get('task_id'),
                                       'candidate': job.get('candidate_index'), 'category': category,
                                       'detail': detail})
        elif isinstance(result, dict) and result.get('error'):
            history = result.get('history') if isinstance(result.get('history'), list) else []
            sandbox_error = any(isinstance(row, dict) and isinstance(row.get('feedback'), str) and
                                row['feedback'].startswith(('sandbox execution failed:', 'sandbox returned'))
                                for row in history)
            category = 'sandbox_error' if sandbox_error else 'unknown'
            candidate_failures.append({'condition': job.get('condition'), 'task_id': job.get('task_id'),
                                       'candidate': job.get('candidate_index'), 'category': category,
                                       'detail': result.get('error')})
    results_file = root / 'results.json'
    if results_file.is_file():
        for task in json.loads(results_file.read_text(encoding='utf-8')):
            if isinstance(task, dict) and task.get('score') is None:
                task_errors.append({'condition': task.get('condition'), 'task_id': task.get('task_id'),
                                    'candidate_failures': sum(1 for failure in candidate_failures
                                        if failure['condition'] == task.get('condition') and failure['task_id'] == task.get('task_id'))})
    candidate_counts: dict[str, int] = {}
    for failure in candidate_failures:
        category = failure['category']
        candidate_counts[category] = candidate_counts.get(category, 0) + 1
    return {
        'protocol': 'p3b_response_diagnostics_v1', 'run_dir': str(root),
        'raw_responses_available': len(responses), 'request_log_entries': len(requests),
        'candidate_records': len(jobs), 'reconstructed_request_groups': len(groups),
        'taxonomy_counts': counts, 'candidate_failure_counts': candidate_counts,
        'candidate_failures': candidate_failures, 'task_errors': task_errors,
        'raw_response_limitations': [] if responses else [
            'Historical llm_requests.jsonl retained response SHA-256 and usage but not HTTP status, headers, raw body, finish_reason, or stop_reason.',
            'Saved analysis and code fields are post-parser values; hidden reasoning and the exact generated byte sequence cannot be reconstructed.',
        ],
        'requests': records,
    }


def diagnose_artifact_tree(root: str | Path) -> dict[str, Any]:
    """Diagnose one run or every nested run under a comparison artifact."""
    path = Path(root).expanduser().resolve()
    runs = [path] if (path / 'llm_requests.jsonl').is_file() else sorted(
        candidate.parent for candidate in path.rglob('llm_requests.jsonl'))
    return {'protocol': 'p3b_response_diagnostics_tree_v1',
            'source_root': str(path), 'runs': [diagnose_artifact_run(run) for run in runs]}

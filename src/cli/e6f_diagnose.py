"""Reproduce the six sealed E6 execution failures without model calls."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from src.arc.e6_curriculum import _quality_grids, verify_oracle
from src.arc.http_backend import final_answer
from src.arc.sandbox import DockerSandbox
from src.pipelines.e6 import _verify_seal

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / 'artifacts/e6/e6-local-bootstrap-e526bae72f'
OUTPUT = PROJECT / 'artifacts/e6f/diagnosis'


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    _verify_seal(SOURCE)
    seal = hashlib.sha256((SOURCE / 'artifact_hashes.json').read_bytes()).hexdigest()
    identity = json.loads((SOURCE / 'identity.json').read_text())
    agent = identity['config']['agent']
    sandbox = DockerSandbox(image=identity['sandbox_image_id'], timeout=agent['timeout_seconds'],
                            memory_mb=agent['memory_mb'], cpus=agent['cpus'], pids=agent['pids'],
                            max_output_bytes=agent['max_output_bytes'], max_code_bytes=agent['max_code_bytes'])
    requests = {r['candidate']: r for r in _rows(SOURCE / 'logs/e6/llm_requests.jsonl')}
    responses = {r['request_id']: r for r in _rows(SOURCE / 'logs/e6/llm_responses.jsonl')}
    failures = [r for r in _rows(SOURCE / 'gen-0/rejections.jsonl')
                if r.get('reason', '').startswith('sandbox execution failed:')]
    if len(failures) != 6:
        raise ValueError('expected the six sealed sandbox failures')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    os.chmod(OUTPUT, 0o700)
    private = OUTPUT / 'private'
    private.mkdir(mode=0o700)
    results = []
    for failure in failures:
        attempt = failure['attempt']
        request = requests[attempt]
        response = responses[request['request_id']]
        raw = response['raw_http_body_text']
        if hashlib.sha256(raw.encode()).hexdigest() != request['response_sha256']:
            raise ValueError('sealed Qwen response hash mismatch')
        code = final_answer(response['generated_text']).strip()
        if hashlib.sha256(code.encode()).hexdigest() != failure['code_sha256']:
            raise ValueError('candidate hash mismatch')
        starts = 0

        class CountedSandbox:
            def run(self, program, grid):
                nonlocal starts
                starts += 1
                return sandbox.run(program, grid)

        category = 'unknown_insufficient_evidence'
        exception_type = None
        detail = ''
        try:
            verify_oracle(code, _quality_grids(66000 + attempt, 7), sandbox=CountedSandbox())
            reproducibility = 'not_reproduced'
        except Exception as exc:
            detail = f'{type(exc).__name__}: {exc}'
            match = re.search(r'(?m)^([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception))(?::|$)', str(exc))
            if isinstance(exc, TimeoutError):
                category = 'timeout_or_resource_limit'
            elif str(exc).startswith('sandbox execution failed:') and match:
                category = 'runtime_exception'
                exception_type = match.group(1)
            elif type(exc).__name__ == 'BackendError':
                category = 'sandbox_infrastructure_error'
            reproducibility = 'reproduced' if category == 'runtime_exception' else 'inconclusive'
        trace_path = private / f'attempt-{attempt}.log'
        trace_path.write_text(detail + '\n', encoding='utf-8')
        os.chmod(trace_path, 0o600)
        results.append({'attempt': attempt, 'request_id': request['request_id'],
                        'code_sha256': failure['code_sha256'],
                        'source': 'gen-0/rejections.jsonl and logs/e6/llm_requests.jsonl',
                        'original_stage': 'oracle sandbox execution',
                        'original_exit_code': None,
                        'original_stderr': 'bounded traceback prefix only; final exception not retained',
                        'classification': category, 'reproduced_exception_type': exception_type,
                        'reproducibility': reproducibility, 'reproduction_sandbox_starts': starts,
                        'resource_limits': {'timeout_seconds': agent['timeout_seconds'],
                                            'memory_mb': agent['memory_mb'], 'cpus': agent['cpus'],
                                            'pids': agent['pids'], 'image_id': identity['sandbox_image_id']}})
    _verify_seal(SOURCE)
    if hashlib.sha256((SOURCE / 'artifact_hashes.json').read_bytes()).hexdigest() != seal:
        raise ValueError('source seal changed')
    summary = {'protocol': 'e6f_failure_diagnostic_v1', 'source_artifact': str(SOURCE),
               'source_seal_sha256': seal, 'model_calls': 0, 'cases': len(results),
               'classification_counts': dict(Counter(r['classification'] for r in results)),
               'reproduction_sandbox_starts': sum(r['reproduction_sandbox_starts'] for r in results),
               'failures': results}
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    with (OUTPUT / 'failures.csv').open('w', newline='') as stream:
        keys = ['attempt', 'request_id', 'code_sha256', 'original_stage', 'original_exit_code',
                'classification', 'reproduced_exception_type', 'reproducibility', 'reproduction_sandbox_starts']
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    lines = ['# E6-F offline diagnosis', '', f"Cases: {len(results)}; model calls: 0.",
             f"Categories: {summary['classification_counts']}.",
             'The historical log retained only a bounded traceback prefix and no exit code.',
             'Reproduction uses the original candidate and deterministic validation inputs inside Docker.',
             'Detailed reproduction errors are in owner-only private logs.', '']
    lines.extend(f"- Attempt {r['attempt']}: {r['classification']}, {r['reproducibility']}; "
                 f"exception type: {r['reproduced_exception_type'] or 'unknown'}." for r in results)
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n')
    seal = {str(path.relative_to(OUTPUT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(OUTPUT.rglob('*')) if path.is_file() and path.name != 'seal.json'}
    (OUTPUT / 'seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True) + '\n')
    os.chmod(OUTPUT / 'seal.json', 0o600)
    print(json.dumps({'output': str(OUTPUT), 'classification_counts': summary['classification_counts'],
                      'sandbox_starts': summary['reproduction_sandbox_starts']}))


if __name__ == '__main__':
    main()

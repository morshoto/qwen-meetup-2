"""Captured local HTTP for matched arms; adapter claims are not authenticated."""
import math
import os
import re
from pathlib import Path

from src.arc.rsi_http_backend import CapturingBackend
from src.arc.http_backend import BackendError
from src.arc.rsi_slot_lifecycle import _directory

_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


class CapturedComparisonTransport:
    def __init__(self, *, root, parent_url, candidate_url, parent_adapter, candidate_adapter, max_request_seconds):
        if ((parent_adapter != 'none' and (not isinstance(parent_adapter, str) or not re.fullmatch(r'[0-9a-f]{64}', parent_adapter))) or
                not isinstance(candidate_adapter, str) or not re.fullmatch(r'[0-9a-f]{64}', candidate_adapter) or
                parent_adapter == candidate_adapter or type(max_request_seconds) not in (int, float) or
                not math.isfinite(max_request_seconds) or max_request_seconds <= 0):
            raise ValueError('invalid comparison transport configuration')
        self.root = Path(root)
        self.urls = {'parent': parent_url, 'candidate': candidate_url}
        self.adapters = {'parent': parent_adapter, 'candidate': candidate_adapter}
        self.max_request_seconds = max_request_seconds
        # Validate both endpoints before creating transport directories.
        for arm in self.urls:
            self._backend(arm, max_request_seconds, None)
        with _directory(self.root) as directory:
            for arm in self.urls:
                try:
                    os.mkdir(f'{arm}.responses', 0o700, dir_fd=directory)
                except FileExistsError:
                    pass
                with _directory(self.root / f'{arm}.responses'):
                    pass

    def _backend(self, arm, timeout, callback):
        return CapturingBackend(model=_MODEL, revision=_REVISION, base_url=self.urls[arm],
            temperature=0.2, max_tokens=512, seed=0, timeout=timeout, thinking='off', guarded=True,
            log_path=self.root / f'{arm}.responses' / 'requests.jsonl',
            request_capture_path=self.root / f'{arm}.capture.jsonl', event_callback=callback)

    def __call__(self, prompt, *, arm, adapter, task_id, sample, seed, timeout):
        if (arm not in self.adapters or adapter != self.adapters[arm] or not isinstance(task_id, str) or not task_id or
                type(sample) is not int or sample not in (0, 1) or type(seed) is not int or not 0 <= seed < 2**31 or
                type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('invalid comparison request routing')
        events = []
        backend = self._backend(arm, min(timeout, self.max_request_seconds), events.append)
        try:
            code = backend.generate(prompt, kind='e5_self_solve', candidate=seed)
        except BackendError as exc:
            if exc.kind != 'malformed_thinking':
                raise
            if len(events) != 1 or not isinstance(events[0].get('request_id'), str):
                raise ValueError('comparison request event unavailable') from exc
            if events[0].get('finish_reason') not in ('stop', 'length'):
                raise BackendError('invalid_response', 'missing model completion status') from exc
            return {'code': None, 'request_id': events[0]['request_id'], 'failure': 'invalid_model_output'}
        if len(events) != 1 or not isinstance(events[0].get('request_id'), str):
            raise ValueError('comparison request event unavailable')
        if events[0].get('finish_reason') == 'length':
            return {'code': None, 'request_id': events[0]['request_id'], 'failure': 'invalid_model_output'}
        if events[0].get('finish_reason') != 'stop':
            raise BackendError('invalid_response', 'missing model completion status')
        return {'code': code, 'request_id': events[0]['request_id']}

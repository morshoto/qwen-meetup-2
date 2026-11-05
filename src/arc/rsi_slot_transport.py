"""Bind scheduled calls to guarded captured HTTP; not live authorization."""
import math

from src.arc.rsi_http_backend import CapturingBackend

_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


class CapturedSlotTransport:
    """A fresh backend per attempt prevents shared deadline/seed mutation.

    The collector reserves tokens before invoking this adapter. The backend is
    configured without a second reservation callback. HTTP capture does not
    authenticate the server, model weights, or adapter.
    """
    def __init__(self, *, base_url, log_path, request_capture_path, max_request_seconds):
        if (type(max_request_seconds) not in (int, float) or not math.isfinite(max_request_seconds)
                or max_request_seconds <= 0):
            raise ValueError('invalid request time limit')
        self.options = dict(model=_MODEL, revision=_REVISION, base_url=base_url,
                            temperature=0.2, max_tokens=512, seed=12000, thinking='off',
                            guarded=True, log_path=log_path, request_capture_path=request_capture_path)
        self.max_request_seconds = max_request_seconds
        self._backend(max_request_seconds)

    def _backend(self, timeout, event_callback=None):
        return CapturingBackend(**self.options, timeout=timeout, event_callback=event_callback)

    def __call__(self, prompt, *, seed, timeout):
        if (type(seed) is not int or seed - 12000 not in (*range(30), *range(100, 130)) or
                type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('invalid scheduled seed or remaining time')
        events = []
        backend = self._backend(min(timeout, self.max_request_seconds), events.append)
        code = backend.generate(prompt, kind='e5_generator', candidate=seed - 12000)
        if len(events) != 1 or not isinstance(events[0].get('request_id'), str):
            raise ValueError('request event is unavailable')
        return {'code': code, 'request_id': events[0]['request_id']}

"""Incrementally observe MLX loss reports; not proof of training provenance."""
import math
import re
import threading

_REPORT = re.compile(r'^Iter ([1-9][0-9]*): Train loss ([^,\s]+),')


class TrainingLossWatch:
    def __init__(self):
        self._lock = threading.Lock()
        self._buffer = b''
        self._status = 'ok'
        self._last_step = 0
        self._last_loss = None

    def feed(self, chunk):
        with self._lock:
            if self._status != 'ok':
                return
            if not isinstance(chunk, bytes) or len(chunk) > 1024 * 1024:
                self._status = 'unavailable'
                return
            self._buffer += chunk
            while b'\n' in self._buffer:
                line, self._buffer = self._buffer.split(b'\n', 1)
                if len(line) > 16384:
                    self._status = 'unavailable'
                    return
                try:
                    text = line.decode('utf-8')
                except UnicodeDecodeError:
                    self._status = 'unavailable'
                    return
                if 'Train loss' not in text:
                    continue
                match = _REPORT.match(text)
                try:
                    if match is None or int(match[1]) <= self._last_step:
                        raise ValueError('invalid loss report')
                    loss = float(match[2])
                except ValueError:
                    self._status = 'unavailable'
                    return
                if not math.isfinite(loss):
                    self._status = 'nonfinite_loss'
                    return
                self._last_step, self._last_loss = int(match[1]), loss
            if len(self._buffer) > 16384:
                self._status = 'unavailable'

    def finish(self):
        with self._lock:
            if self._buffer and self._status == 'ok':
                self._status = 'unavailable'

    def snapshot(self):
        with self._lock:
            return {'status': self._status, 'last_step': self._last_step, 'last_loss': self._last_loss}

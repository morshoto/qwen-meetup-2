"""Hash permitted answers without decoding or retaining withheld values."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest


def permitted_fingerprints(path, allowed, withheld, expected_sha256):
    """Parse only allowed top-level values from a hash-pinned solutions object.

    Other values are scanned solely to skip their JSON nesting; their bytes are
    neither accumulated nor decoded. The source hash is checked before and after.
    """
    allowed, withheld = set(allowed), set(withheld)
    if not allowed or allowed & withheld or any(not isinstance(k, str) or not k for k in allowed | withheld):
        raise ValueError('invalid permitted or withheld task inventory')
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('invalid solution source')
    result, seen = {}, set()
    with path.open('rb') as stream:
        def source_hash():
            stream.seek(0)
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
            return h.hexdigest()

        if source_hash() != expected_sha256:
            raise ValueError('solution source changed')
        stream.seek(0)

        def nonspace():
            c = stream.read(1)
            while c and c in b' \t\r\n':
                c = stream.read(1)
            return c

        def key_string():
            raw = bytearray(b'"'); escaped = False
            while True:
                c = stream.read(1)
                if not c: raise ValueError('unterminated solution key')
                raw.extend(c)
                if escaped: escaped = False
                elif c == b'\\': escaped = True
                elif c == b'"': return json.loads(bytes(raw))

        def array_value(capture):
            raw = bytearray(b'[') if capture else None
            stack = [b']']; quoted = False; escaped = False
            while stack:
                c = stream.read(1)
                if not c: raise ValueError('unterminated solution value')
                if raw is not None: raw.extend(c)
                if quoted:
                    if escaped: escaped = False
                    elif c == b'\\': escaped = True
                    elif c == b'"': quoted = False
                elif c == b'"': quoted = True
                elif c == b'[': stack.append(b']')
                elif c == b'{': stack.append(b'}')
                elif c in (b']', b'}'):
                    if stack.pop() != c: raise ValueError('invalid solution nesting')
            return json.loads(bytes(raw)) if raw is not None else None

        if nonspace() != b'{': raise ValueError('solution root must be an object')
        c = nonspace()
        while c != b'}':
            if c != b'"': raise ValueError('invalid solution key')
            key = key_string()
            if not isinstance(key, str) or key in seen: raise ValueError('duplicate solution key')
            seen.add(key)
            if nonspace() != b':' or nonspace() != b'[': raise ValueError('invalid solution entry')
            value = array_value(key in allowed)
            if key in allowed: result[key] = digest(value)
            c = nonspace()
            if c == b',': c = nonspace()
            elif c != b'}': raise ValueError('invalid solution separator')
        if nonspace() or not allowed | withheld <= seen or source_hash() != expected_sha256:
            raise ValueError('solution inventory or source changed')
    return result

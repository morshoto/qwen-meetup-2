"""Persistent atomic reservations for an in-process, exactly tokenized backend.

The caller must tokenize the exact input consumed by the model and enforce the
reserved generation bound in the generation process. A reservation is never
refunded, including when a request crashes or returns early.
"""
from __future__ import annotations

import hashlib
from contextlib import closing
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path


class TokenBudgetError(RuntimeError):
    """A request cannot safely be admitted to the token budget."""


@dataclass(frozen=True)
class Reservation:
    request_id: str
    prompt_tokens: int
    generation_limit: int
    prefetch_allowance: int
    reserved_tokens: int
    used_after: int


class TokenLedger:
    """A SQLite transaction reserves the worst-case model work before dispatch."""

    def __init__(self, path: str | Path, limit: int) -> None:
        if type(limit) is not int or limit <= 0:
            raise ValueError('token limit must be a positive integer')
        self.path = str(path)
        if self.path == ':memory:':
            raise ValueError('token ledger requires persistent storage')
        self.limit = limit
        with closing(self._connect()) as db:
            db.execute('CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK (id = 1), token_limit INTEGER NOT NULL, reserved INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, prompt_hash TEXT NOT NULL, prompt_tokens INTEGER NOT NULL, generation_limit INTEGER NOT NULL, prefetch_allowance INTEGER NOT NULL, reserved_tokens INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK (id = 1), ledger_id TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS binding (id INTEGER PRIMARY KEY CHECK (id = 1), binding_hash TEXT NOT NULL)')
            db.execute('INSERT OR IGNORE INTO metadata VALUES (1, ?)', (uuid.uuid4().hex,))
            db.execute('INSERT OR IGNORE INTO budget (id, token_limit, reserved) VALUES (1, ?, 0)', (limit,))
            current = db.execute('SELECT token_limit FROM budget WHERE id = 1').fetchone()
            if current is None or current[0] != limit:
                raise TokenBudgetError('token limit differs from the existing ledger')

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.execute('PRAGMA synchronous=FULL')
        return db

    def reserve(self, request_id: str, prompt_token_ids: list[int], generation_limit: int,
                *, prefetch_allowance: int = 1) -> Reservation:
        if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
            raise ValueError('invalid request ID')
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids or any(type(x) is not int or x < 0 for x in prompt_token_ids):
            raise ValueError('prompt must contain nonnegative integer token IDs')
        if type(generation_limit) is not int or generation_limit <= 0 or type(prefetch_allowance) is not int or prefetch_allowance < 0:
            raise ValueError('invalid generation or prefetch limit')
        amount = len(prompt_token_ids) + generation_limit + prefetch_allowance
        # Hash fixed-width token IDs so different token sequences cannot share a key.
        try:
            encoded = b''.join(x.to_bytes(8, 'big') for x in prompt_token_ids)
        except OverflowError as exc:
            raise ValueError('token ID exceeds 64 bits') from exc
        prompt_hash = hashlib.sha256(encoded).hexdigest()
        with closing(self._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                if db.execute('SELECT 1 FROM requests WHERE request_id = ?', (request_id,)).fetchone():
                    raise TokenBudgetError('request ID already reserved; automatic replay is forbidden')
                row = db.execute('SELECT token_limit, reserved FROM budget WHERE id = 1').fetchone()
                if row is None or row[0] != self.limit:
                    raise TokenBudgetError('invalid ledger budget')
                used = row[1]
                if amount > self.limit - used:
                    raise TokenBudgetError('token budget exhausted before model dispatch')
                db.execute('INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?)',
                           (request_id, prompt_hash, len(prompt_token_ids), generation_limit, prefetch_allowance, amount))
                db.execute('UPDATE budget SET reserved = ? WHERE id = 1', (used + amount,))
                db.execute('COMMIT')
            except BaseException:
                db.execute('ROLLBACK')
                raise
        return Reservation(request_id, len(prompt_token_ids), generation_limit, prefetch_allowance, amount, used + amount)

    def reserved(self) -> int:
        with closing(self._connect()) as db:
            return int(db.execute('SELECT reserved FROM budget WHERE id = 1').fetchone()[0])

    def identity(self) -> str:
        with closing(self._connect()) as db:
            return str(db.execute('SELECT ledger_id FROM metadata WHERE id = 1').fetchone()[0])

    def bind(self, binding_hash: str) -> None:
        if not isinstance(binding_hash, str) or len(binding_hash) != 64 or any(c not in '0123456789abcdef' for c in binding_hash):
            raise ValueError('invalid server binding hash')
        with closing(self._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                db.execute('INSERT OR IGNORE INTO binding VALUES (1, ?)', (binding_hash,))
                saved = db.execute('SELECT binding_hash FROM binding WHERE id = 1').fetchone()
                if saved is None or saved[0] != binding_hash:
                    raise TokenBudgetError('ledger belongs to a different server configuration')
                db.execute('COMMIT')
            except BaseException:
                db.execute('ROLLBACK')
                raise

"""Token-ID boundary for a future locally controlled generation server.

The driver and tokenizer are trusted components in the same process. This
module does not claim to enforce limits on an arbitrary remote HTTP server.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol
from src.arc.token_ledger import Reservation, TokenLedger


class TokenDriver(Protocol):
    """A lazy driver may compute at most one lookahead token per request."""
    def encode_messages(self, messages: list[dict[str, str]]) -> list[int]: ...
    def generate_ids(self, prompt_ids: list[int], max_tokens: int) -> Iterator[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class GuardedResult:
    text: str
    output_tokens: int
    finish_reason: str
    reservation: Reservation


class GuardedGenerator:
    def __init__(self, ledger: TokenLedger, driver: TokenDriver, *, max_prompt_tokens: int,
                 prefetch_allowance: int = 1, stop_token_ids: frozenset[int] = frozenset()) -> None:
        if type(max_prompt_tokens) is not int or max_prompt_tokens <= 0:
            raise ValueError('max prompt tokens must be a positive integer')
        if type(prefetch_allowance) is not int or prefetch_allowance < 1:
            raise ValueError('at least one lookahead token must be reserved')
        self.ledger, self.driver = ledger, driver
        self.max_prompt_tokens, self.prefetch_allowance = max_prompt_tokens, prefetch_allowance
        self.stop_token_ids = stop_token_ids

    def generate(self, request_id: str, messages: list[dict[str, str]], max_tokens: int,
                 *, thinking_mode: str | None = None) -> GuardedResult:
        if thinking_mode is None:
            ids = self.driver.encode_messages(messages)
        else:
            try:
                ids = self.driver.encode_messages(messages, thinking_mode=thinking_mode)
            except TypeError:
                if thinking_mode != getattr(self.driver, 'thinking', 'server_default'):
                    raise ValueError('token driver does not support per-request thinking control')
                ids = self.driver.encode_messages(messages)
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError('driver returned invalid prompt token IDs')
        if len(ids) > self.max_prompt_tokens:
            raise ValueError('prompt token limit exceeded before model dispatch')
        reservation = self.ledger.reserve(request_id, ids, max_tokens, prefetch_allowance=self.prefetch_allowance)
        tokens: list[int] = []
        iterator = iter(self.driver.generate_ids(ids, max_tokens))
        try:
            for _ in range(max_tokens):
                try:
                    token = next(iterator)
                except StopIteration:
                    break
                if type(token) is not int or token < 0:
                    raise ValueError('driver returned an invalid generated token')
                tokens.append(token)
                if token in self.stop_token_ids:
                    break
        finally:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()
        visible = tokens[:-1] if tokens and tokens[-1] in self.stop_token_ids else tokens
        finish_reason = 'length' if len(tokens) == max_tokens and not (tokens and tokens[-1] in self.stop_token_ids) else 'stop'
        return GuardedResult(self.driver.decode(visible), len(tokens), finish_reason, reservation)

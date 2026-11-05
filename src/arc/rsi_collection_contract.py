"""Check declared collection server contracts; not process attestation."""
import json
import math
import re
import threading
import time
from pathlib import Path

from src.arc.http_backend import verify_guarded_contract
from src.arc.prompting import system_prompt_for_kind
from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_slot_transport import CapturedSlotTransport, _MODEL, _REVISION


class ContractCheckedSlotTransport:
    def __init__(self, transport, *, budget, registration_sha256, generation, parent_adapter,
                 expected_max_prompt_tokens, expected_ledger_id, expected_initial_reserved_tokens):
        if (not isinstance(transport, CapturedSlotTransport) or type(budget) is not CollectionBudget or
                budget.identity != {'registration_sha256': registration_sha256, 'generation': generation} or
                (generation == 0 and parent_adapter != 'none') or
                (generation == 1 and (not isinstance(parent_adapter, str) or not re.fullmatch(r'[0-9a-f]{64}', parent_adapter))) or
                type(generation) is not int or generation not in (0, 1) or
                type(expected_max_prompt_tokens) is not int or expected_max_prompt_tokens <= 0 or
                not isinstance(expected_ledger_id, str) or not re.fullmatch(r'[0-9a-f]{32}', expected_ledger_id) or
                type(expected_initial_reserved_tokens) is not int or not 0 <= expected_initial_reserved_tokens <= 200000):
            raise ValueError('invalid collection contract configuration')
        root = Path(budget.root).resolve()
        if any(not Path(transport.options[name]).resolve().is_relative_to(root)
               for name in ('log_path', 'request_capture_path')):
            raise ValueError('collection logs must belong to the budget directory')
        self.transport, self.budget = transport, budget
        self.generation, self.parent_adapter = generation, parent_adapter
        self.max_prompt_tokens, self.ledger_id = expected_max_prompt_tokens, expected_ledger_id
        self.lock = threading.Lock()
        local = budget.snapshot()
        self.previous_local = (local['model_calls'], local['reserved_tokens'])
        initial = self._check(time.monotonic() + transport.max_request_seconds, local)
        if initial['reserved_tokens'] != expected_initial_reserved_tokens:
            raise ValueError('initial collection ledger differs')
        self.token_limit, self.previous_server = initial['token_limit'], initial['reserved_tokens']

    def _check(self, deadline, local):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('collection contract deadline exceeded')
        value = verify_guarded_contract(self.transport.options['base_url'], _MODEL, _REVISION,
            200000, 512, min(remaining, self.transport.max_request_seconds), 'off',
            expected_adapter_sha256=self.parent_adapter, expected_max_prompt_tokens=self.max_prompt_tokens,
            required_thinking_modes=('off',))
        if (not isinstance(value, dict) or value.get('ledger_id') != self.ledger_id or
                type(value.get('token_limit')) is not int or not 0 < value['token_limit'] <= 200000 or
                type(value.get('reserved_tokens')) is not int or
                not 0 <= value['reserved_tokens'] <= min(value['token_limit'], local['reserved_tokens'])):
            raise ValueError('collection server ledger differs from budget')
        if hasattr(self, 'token_limit') and (value['token_limit'] != self.token_limit or
                                             value['reserved_tokens'] < self.previous_server):
            raise ValueError('collection server ledger changed or regressed')
        return value

    def __call__(self, prompt, *, seed, timeout):
        valid_seeds = range(12000 + 100 * self.generation, 12030 + 100 * self.generation)
        if (not isinstance(prompt, str) or type(seed) is not int or seed not in valid_seeds or
                type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('invalid collection request')
        deadline = time.monotonic() + timeout
        with self.lock:
            local = self.budget.snapshot()
            messages = [{'role': 'system', 'content': system_prompt_for_kind('e5_generator')},
                        {'role': 'user', 'content': prompt}]
            reservation = 512 + len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 1024
            if (local['model_calls'] != self.previous_local[0] + 1 or
                    local['reserved_tokens'] != self.previous_local[1] + reservation):
                raise ValueError('collection request lacks its exact local reservation')
            self.previous_local = (local['model_calls'], local['reserved_tokens'])
            state = self._check(deadline, local)
            self.previous_server = state['reserved_tokens']
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('collection dispatch deadline exceeded')
            return self.transport(prompt, seed=seed, timeout=remaining)

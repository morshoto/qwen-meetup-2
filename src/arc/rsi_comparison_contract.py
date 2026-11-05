"""Check declared guarded server contracts, not process or model attestation."""
import math
import threading
import time

from src.arc.http_backend import verify_guarded_contract
from src.arc.rsi_comparison_transport import CapturedComparisonTransport, _MODEL, _REVISION


class ContractCheckedComparisonTransport:
    def __init__(self, transport, *, expected_max_prompt_tokens):
        if (not isinstance(transport, CapturedComparisonTransport) or
                transport.urls['parent'] == transport.urls['candidate'] or
                type(expected_max_prompt_tokens) is not int or expected_max_prompt_tokens <= 0):
            raise ValueError('invalid guarded comparison configuration')
        self.transport = transport
        self.expected_max_prompt_tokens = expected_max_prompt_tokens
        self.lock = threading.Lock()
        self.initial = self._check(time.monotonic() + transport.max_request_seconds, initial=True)
        self.previous = {arm: item['reserved_tokens'] for arm, item in self.initial.items()}

    def _check(self, deadline, *, initial=False):
        contracts = {}
        for arm in ('parent', 'candidate'):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('comparison contract deadline exceeded')
            value = verify_guarded_contract(self.transport.urls[arm], _MODEL, _REVISION,
                16000000, 512, min(remaining, self.transport.max_request_seconds), 'off',
                expected_adapter_sha256=self.transport.adapters[arm],
                expected_max_prompt_tokens=self.expected_max_prompt_tokens, required_thinking_modes=('off',))
            if (not isinstance(value, dict) or not isinstance(value.get('ledger_id'), str) or
                    type(value.get('token_limit')) is not int or type(value.get('reserved_tokens')) is not int or
                    not 0 <= value['reserved_tokens'] <= value['token_limit']):
                raise ValueError('invalid guarded comparison contract')
            if initial and value['reserved_tokens'] != 0:
                raise ValueError('comparison server ledger is not fresh')
            if not initial:
                old = self.initial[arm]
                if (value['ledger_id'] != old['ledger_id'] or value['token_limit'] != old['token_limit'] or
                        value['reserved_tokens'] < self.previous[arm]):
                    raise ValueError('comparison server ledger changed or regressed')
            contracts[arm] = value
        if (contracts['parent']['ledger_id'] == contracts['candidate']['ledger_id'] or
                sum(contracts[arm]['token_limit'] for arm in contracts) > 16000000):
            raise ValueError('comparison server ledgers or combined budgets differ')
        return contracts

    def __call__(self, prompt, *, arm, adapter, task_id, sample, seed, timeout):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('invalid comparison timeout')
        deadline = time.monotonic() + timeout
        with self.lock:
            contracts = self._check(deadline)
            self.previous = {name: item['reserved_tokens'] for name, item in contracts.items()}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('comparison dispatch deadline exceeded')
            return self.transport(prompt, arm=arm, adapter=adapter, task_id=task_id,
                                  sample=sample, seed=seed, timeout=remaining)

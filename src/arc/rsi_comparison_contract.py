"""Check declared guarded server contracts, not process or model attestation."""
import json
import math
from pathlib import Path
import threading
import time

from src.arc.http_backend import verify_guarded_contract
from src.arc.prompting import system_prompt_for_kind
from src.arc.rsi_comparison_budget import ComparisonBudget
from src.arc.rsi_comparison_transport import CapturedComparisonTransport, _MODEL, _REVISION


class ContractCheckedComparisonTransport:
    budget_type = ComparisonBudget
    token_limit = 16000000

    def __init__(self, transport, *, budget, registration_sha256, generation, expected_max_prompt_tokens):
        if (not isinstance(transport, CapturedComparisonTransport) or type(budget) is not self.budget_type or
                budget.identity != {'registration_sha256': registration_sha256, 'generation': generation} or
                type(generation) is not int or generation not in (0, 1) or
                Path(transport.root).resolve() != Path(budget.root).resolve() or
                transport.urls['parent'] == transport.urls['candidate'] or
                type(expected_max_prompt_tokens) is not int or expected_max_prompt_tokens <= 0):
            raise ValueError('invalid guarded comparison configuration')
        self.transport, self.budget = transport, budget
        self.expected_max_prompt_tokens = expected_max_prompt_tokens
        self.lock = threading.Lock()
        local = budget.snapshot()
        if any(local[key] != 0 for key in ('model_calls', 'reserved_tokens', 'sandbox_runs')):
            raise ValueError('comparison local budget is not fresh')
        self.previous_local = (local['model_calls'], local['reserved_tokens'])
        self.initial = self._check(time.monotonic() + transport.max_request_seconds, local, initial=True)
        self.previous = {arm: item['reserved_tokens'] for arm, item in self.initial.items()}

    def _check(self, deadline, local, *, initial=False):
        contracts = {}
        for arm in ('parent', 'candidate'):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('comparison contract deadline exceeded')
            value = verify_guarded_contract(self.transport.urls[arm], _MODEL, _REVISION,
                self.token_limit, 512, min(remaining, self.transport.max_request_seconds), 'off',
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
                sum(contracts[arm]['token_limit'] for arm in contracts) > self.token_limit or
                sum(contracts[arm]['reserved_tokens'] for arm in contracts) > local['reserved_tokens']):
            raise ValueError('comparison server ledgers or combined budgets differ')
        return contracts

    def __call__(self, prompt, *, arm, adapter, task_id, sample, seed, timeout):
        if (not isinstance(prompt, str) or arm not in self.transport.adapters or
                adapter != self.transport.adapters[arm] or not isinstance(task_id, str) or not task_id or
                type(sample) is not int or sample not in (0, 1) or type(seed) is not int or not 0 <= seed < 2**31 or
                type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('invalid comparison request')
        deadline = time.monotonic() + timeout
        with self.lock:
            local = self.budget.snapshot()
            messages = [{'role': 'system', 'content': system_prompt_for_kind('e5_self_solve')},
                        {'role': 'user', 'content': prompt}]
            reservation = 512 + len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 1024
            if (local['model_calls'] != self.previous_local[0] + 1 or
                    local['reserved_tokens'] != self.previous_local[1] + reservation):
                raise ValueError('comparison request lacks its exact local reservation')
            self.previous_local = (local['model_calls'], local['reserved_tokens'])
            contracts = self._check(deadline, local)
            self.previous = {name: item['reserved_tokens'] for name, item in contracts.items()}
            remaining = min(deadline - time.monotonic(), self.budget.remaining())
            if remaining <= 0:
                raise TimeoutError('comparison dispatch deadline exceeded')
            return self._dispatch(prompt, arm=arm, adapter=adapter, task_id=task_id,
                                  sample=sample, seed=seed, timeout=remaining)

    def _dispatch(self, prompt, *, arm, adapter, task_id, sample, seed, timeout):
        return self.transport(prompt, arm=arm, adapter=adapter, task_id=task_id,
                              sample=sample, seed=seed, timeout=timeout)

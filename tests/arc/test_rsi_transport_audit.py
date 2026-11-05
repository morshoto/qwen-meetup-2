"""Offline checks for request and response payload consistency."""
import copy
import hashlib
import json
import unittest

from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.arc.rsi_transport_audit import audit_transport


MODEL = 'mlx-community/Qwen3-14B-4bit'
REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
CODE = 'def transform(grid):\n    return grid[::-1]'


def build_evidence():
    request = {'model': MODEL, 'messages': [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT},
                {'role': 'user', 'content': 'visible prompt'}], 'temperature': 0.2, 'max_tokens': 512,
               'seed': 12000, 'stream': False, 'thinking_mode': 'off', 'deadline_monotonic': 123.0}
    response = {'model': MODEL, 'thinking_mode': 'off', 'thinking_control': 'tokenizer_template_canary_passed',
                'choices': [{'message': {'role': 'assistant', 'content': CODE}, 'finish_reason': 'stop'}]}
    request_body = json.dumps(request).encode()
    response_body = json.dumps(response).encode()
    ledger = {'request_id': 'fixture-1', 'request_body_sha256': hashlib.sha256(request_body).hexdigest(),
              'response_sha256': hashlib.sha256(response_body).hexdigest(), 'model': MODEL,
              'revision': REVISION, 'seed_requested': 12000, 'thinking_mode_effective': 'off',
              'error': None}
    return request_body, response_body, ledger


class TransportAuditTests(unittest.TestCase):
    def test_matching_payloads_are_consistent_but_not_training_eligible(self):
        request, response, ledger = build_evidence()
        result = audit_transport(request, response, ledger, expected_prompt='visible prompt',
                                 expected_code=CODE, expected_seed=12000)
        self.assertEqual(result['status'], 'transport_consistent')
        self.assertFalse(result['eligible_for_training'])
        self.assertEqual(result['request_id'], 'fixture-1')

    def test_detects_raw_body_tampering_and_missing_evidence(self):
        request, response, ledger = build_evidence()
        for req, resp in [(request + b' ', response), (request, response + b' '), (b'', response)]:
            with self.subTest(req=req == request, resp=resp == response), self.assertRaises(ValueError):
                audit_transport(req, resp, ledger, expected_prompt='visible prompt',
                                expected_code=CODE, expected_seed=12000)

    def test_rejects_wrong_prompt_seed_model_and_code(self):
        request, response, ledger = build_evidence()
        for prompt, code, seed, row in [('wrong', CODE, 12000, ledger),
                                        ('visible prompt', 'other code', 12000, ledger),
                                        ('visible prompt', CODE, 12001, ledger),
                                        ('visible prompt', CODE, 12000, dict(ledger, revision='wrong'))]:
            with self.subTest(prompt=prompt, seed=seed), self.assertRaises(ValueError):
                audit_transport(request, response, row, expected_prompt=prompt,
                                expected_code=code, expected_seed=seed)

    def test_rejects_conflicting_response_or_disabled_guard(self):
        request, response, ledger = build_evidence()
        for change in ({'thinking_mode': 'on'}, {'model': 'other'}, {'thinking_control': 'unverified'}):
            value = json.loads(response)
            value.update(change)
            changed = json.dumps(value).encode()
            row = copy.deepcopy(ledger)
            row['response_sha256'] = hashlib.sha256(changed).hexdigest()
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit_transport(request, changed, row, expected_prompt='visible prompt',
                                expected_code=CODE, expected_seed=12000)


if __name__ == '__main__':
    unittest.main()

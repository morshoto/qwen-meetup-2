import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.arc.rsi_generation_budget_draft import prepare_draft_generation_budget

ROOT = Path(__file__).resolve().parents[2]


class DraftGenerationBudgetTests(unittest.TestCase):
    def inputs(self, key):
        registration = (ROOT / f'research/rsi/oracle-generation-registration-{key}-draft.json').read_bytes()
        reference = (ROOT / f'research/rsi/audits/historical-candidates/{key}-behavior-reference.json').read_bytes()
        return registration, reference

    def test_each_model_draft_initializes_bound_setup(self):
        for key in ('14b', '32b'):
            registration, reference = self.inputs(key)
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                result = prepare_draft_generation_budget(root=root, registration_bytes=registration,
                    expected_registration_sha256=hashlib.sha256(registration).hexdigest(), reference_bytes=reference)
                state = result['budget'].snapshot()
                self.assertEqual(state['reference_model_key'], key)
                self.assertEqual(state['reference_setup_sha256'], hashlib.sha256(reference).hexdigest())
                self.assertEqual(state['reference_pool_sha256'], json.loads(reference)['pool_sha256'])
                self.assertEqual(state['sandbox_runs'], 10)
                self.assertEqual(state['reference_elapsed_seconds'], json.loads(reference)['elapsed_seconds'])
                self.assertFalse(result['execution_authorized'])
                self.assertFalse(result['training_allowed'])

    def test_invalid_bindings_fail_before_budget_creation(self):
        registration, reference = self.inputs('14b')
        other_reference = self.inputs('32b')[1]
        original = json.loads(registration)
        mutations = [dict(original, model_key='32b'), dict(original, generation=1),
                     dict(original, execution_ready=True), dict(original, training_allowed=True),
                     dict(original, limits={**original['limits'], 'sandbox_starts': 301}),
                     dict(original, provisional_behavior_reference={**original['provisional_behavior_reference'],
                                                                   'pool_sha256': '0'*64}),
                     dict(original, provisional_behavior_reference={**original['provisional_behavior_reference'],
                                                                   'local_result_sha256': '0'*64}),
                     dict(original, provisional_probe_candidates={**original['provisional_probe_candidates'],
                          'private': {**original['provisional_probe_candidates']['private'],
                                      'canonical_probe_sha256': '0'*64}})]
        cases = [(registration, other_reference, hashlib.sha256(registration).hexdigest()),
                 (registration, reference, '0'*64)]
        for item in mutations:
            raw = json.dumps(item).encode()
            cases.append((raw, reference, hashlib.sha256(raw).hexdigest()))
        for reg, ref, expected in cases:
            with self.subTest(reg=hashlib.sha256(reg).hexdigest(), ref=hashlib.sha256(ref).hexdigest()), tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    prepare_draft_generation_budget(root=root, registration_bytes=reg,
                        expected_registration_sha256=expected, reference_bytes=ref)
                self.assertFalse((Path(root) / 'g0.budget.json').exists())

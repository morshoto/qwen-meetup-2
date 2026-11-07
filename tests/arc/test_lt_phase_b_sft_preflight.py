import copy
import unittest

from src.arc.lt_phase_b_sft_preflight import audit_sft_dataset
from tests.arc import test_lt_phase_b_dataset as fixtures


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        assert kwargs == {'add_generation_prompt': True, 'enable_thinking': False, 'return_dict': False}
        return [1, 2, 3]


class Dataset:
    def __init__(self, example, tokenizer): self.example = example
    def process(self, example): return [1, 2, 3, 4, 5], 3


class SFTPreflightTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DatasetTests('test_full_correct_only_one_lineage_and_disjoint_split')
        self.fixture.setUp()
        f = self.fixture
        rows = [fixtures.row('a', [f.good, f.good]), fixtures.row('b', [f.bad, f.good]),
                fixtures.row('c', [f.good, f.bad]), fixtures.row('d', [f.bad, None])]
        self.dataset = f.build(rows)

    def audit(self, dataset=None, tokenizer=None, dataset_factory=None):
        return audit_sft_dataset(dataset=self.dataset if dataset is None else dataset,
            tasks=self.fixture.tasks, verifier=self.fixture.verifier, source=self.fixture.source, tokenizer=tokenizer or Tokenizer(), dataset_factory=dataset_factory or Dataset)

    def test_audits_exact_verified_examples_and_prefix(self):
        result = self.audit()
        self.assertEqual(result['status'], 'dataset_tokens_consistent_unverified')
        self.assertFalse(result['training_allowed'])
        self.assertEqual((result['train_count'], result['valid_count']), (2, 1))
        self.assertEqual(result['maximum_tokens'], 5)
        self.assertEqual(len(result['train_jsonl_sha256']), 64)
        self.assertNotIn('messages', result)

    def test_truncation_or_incorrect_mask_fails(self):
        class TooLong(Dataset):
            def process(self, example): return [1, 2, 3] + [4] * 510, 3
        class WrongMask(Dataset):
            def process(self, example): return [1, 2, 3, 4], 2
        class WrongPrefix(Dataset):
            def process(self, example): return [1, 2, 0, 4], 3
        for factory in (TooLong, WrongMask, WrongPrefix):
            with self.subTest(factory=factory), self.assertRaises(ValueError):
                self.audit(dataset_factory=factory)

    def test_eligible_record_must_match_registered_source(self):
        for field, value in (('source_model_revision', 'wrong'), ('source_code_sha256', 'd' * 64),
                             ('run_sha256', 'd' * 64), ('run_id', 'other'), ('adapter_state', 'on')):
            changed = copy.deepcopy(self.dataset)
            record = next(x for x in changed['journal'] if x['verification_status'] == 'eligible')
            record[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.audit(changed)

    def test_correctness_claim_is_rechecked_by_verifier(self):
        changed = copy.deepcopy(self.dataset)
        task_id = changed['split_task_ids']['train'][0]
        record = next(x for x in changed['journal'] if x['task_id'] == task_id and x['verification_status'] == 'eligible')
        record['prediction'] = [[[9]]]
        changed['train'][0]['messages'][1]['content'] = '[[[9]]]'
        with self.assertRaises(ValueError): self.audit(changed)

    def test_tampered_examples_splits_or_lineages_fail(self):
        changed = copy.deepcopy(self.dataset); changed['train'][0]['messages'][1]['content'] = '[[[9]]]'
        with self.assertRaises(ValueError): self.audit(changed)
        changed = copy.deepcopy(self.dataset); changed['split_task_ids']['valid'] = changed['split_task_ids']['train'][:1]
        with self.assertRaises(ValueError): self.audit(changed)
        changed = copy.deepcopy(self.dataset)
        eligible = [x for x in changed['journal'] if x['verification_status'] == 'eligible']
        eligible[0]['solution_lineage_sha256'] = eligible[1]['solution_lineage_sha256']
        with self.assertRaises(ValueError): self.audit(changed)
        changed = copy.deepcopy(self.dataset); changed['yield_status'] = 'collection_incomplete'
        with self.assertRaises(ValueError): self.audit(changed)


if __name__ == '__main__': unittest.main()

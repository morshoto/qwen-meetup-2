"""A structural SFT batch check must never substitute for independent audit."""
import copy
import hashlib
import unittest

from src.arc.ast_fingerprint import stable_ast_sha256
from src.arc.rsi_batch_precheck import precheck_batch


MODEL = 'mlx-community/Qwen3-14B-4bit'
REV = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


def row(i):
    code = f'def transform(grid):\n    return [[(x + {i}) % 10 for x in r] for r in grid]'
    digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
    return {'status': 'pending_independent_audit', 'eligible_for_training': False,
            'source_type': 'generator_distillation', 'parent_task_id': f't{i}',
            'generation': 0, 'parent_adapter': 'none', 'source_sha256': digest(code),
            'ast_sha256': stable_ast_sha256(code), 'behavior_sha256': digest(f'behavior-{i}'),
            'probe_set_sha256': 'a' * 64, 'reference_pool_sha256': digest(f'pool-{i}'),
            'provenance': {'source_type': 'generator_distillation', 'model_id': MODEL,
                           'model_revision': REV, 'request_id': f'request-{i}',
                           'raw_response_sha256': digest(f'response-{i}'), 'execution_mode': 'live_model'},
            'mask_prompt': True, 'loss_target': 'assistant_completion_only',
            'messages': [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'task'},
                         {'role': 'assistant', 'content': code}]}


class BatchPrecheckTests(unittest.TestCase):
    def test_fewer_than_five_does_not_reach_precheck_threshold(self):
        result = precheck_batch([row(i) for i in range(4)], generation=0,
                                parent_adapter='none', allowed_tasks={f't{i}' for i in range(10)})
        self.assertEqual(result['status'], 'insufficient_records')
        self.assertEqual(result['structurally_valid_count'], 4)
        self.assertFalse(result['training_allowed'])

    def test_five_structurally_valid_records_still_need_independent_audit(self):
        result = precheck_batch([row(i) for i in range(5)], generation=0,
                                parent_adapter='none', allowed_tasks={f't{i}' for i in range(10)})
        self.assertEqual(result['status'], 'pending_independent_audit')
        self.assertFalse(result['training_allowed'])

    def test_fake_wrong_generation_and_source_substitution_are_rejected(self):
        for mutation in ('fake', 'generation', 'boolean_generation', 'source'):
            item = row(0)
            if mutation == 'fake': item['provenance']['execution_mode'] = 'fake_fixture'
            if mutation == 'generation': item['generation'] = 1
            if mutation == 'boolean_generation': item['generation'] = False
            if mutation == 'source': item['messages'][-1]['content'] += '\n# changed'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                precheck_batch([item], generation=0, parent_adapter='none', allowed_tasks={'t0'})

    def test_rejects_duplicates_different_probes_and_private_payload(self):
        for mutation in ('task', 'ast', 'behavior', 'request', 'probes', 'private'):
            first, second = row(0), row(1)
            if mutation == 'task': second['parent_task_id'] = first['parent_task_id']
            if mutation == 'ast':
                second['messages'][-1]['content'] = first['messages'][-1]['content']
                second['source_sha256'], second['ast_sha256'] = first['source_sha256'], first['ast_sha256']
            if mutation == 'behavior': second['behavior_sha256'] = first['behavior_sha256']
            if mutation == 'request': second['provenance']['request_id'] = first['provenance']['request_id']
            if mutation == 'probes': second['probe_set_sha256'] = 'f' * 64
            if mutation == 'private': second['private'] = {'query_outputs': [[[9]]]}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                precheck_batch([first, second], generation=0, parent_adapter='none',
                               allowed_tasks={'t0', 't1'})

    def test_rejects_task_not_in_registered_schedule(self):
        with self.assertRaises(ValueError):
            precheck_batch([row(0)], generation=0, parent_adapter='none', allowed_tasks={'other'})


if __name__ == '__main__':
    unittest.main()

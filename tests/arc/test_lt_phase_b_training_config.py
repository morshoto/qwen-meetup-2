import json
import unittest
from pathlib import Path

from src.arc.lt_phase_b_training_config import build_training_config
from tests.arc import test_lt_phase_b_sft_materialize as fixtures
from tests.arc import test_lt_phase_b_dataset as dataset_fixtures


class TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SFTMaterializeTests('test_writes_private_exact_jsonl_once')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.model = self.fixture.root / 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'; self.model.mkdir()
        self.manifest = self.fixture.materialize()

    def build(self, **kwargs):
        args = {'source_root': self.fixture.root, 'run_dir': self.fixture.run, 'model_dir': self.model,
                'expected_source': self.fixture.fixture.fixture.source}
        args.update(kwargs)
        return build_training_config(**args)

    def test_disjoint_yield_proposes_fixed_final_step_configuration(self):
        result = self.build()
        self.assertEqual(result['status'], 'real_training_configuration_proposed_unverified')
        self.assertFalse(result['training_allowed'])
        self.assertFalse(result['execution_ready'])
        self.assertEqual(result['yield_status'], 'disjoint_split')
        config = result['trainer_config']
        self.assertEqual((config['iters'], config['num_layers'], config['batch_size'], config['max_seq_length']), (20, 8, 1, 512))
        self.assertEqual((config['seed'], config['learning_rate'], config['mask_prompt']), (20261111, 1e-4, True))
        self.assertEqual(config['lora_parameters'], {'rank': 8, 'scale': 2.0, 'dropout': 0.0})
        self.assertEqual(config['val_batches'], -1)
        self.assertEqual(config['save_every'], 20)
        self.assertFalse(config['test'])

    def test_changed_data_or_source_never_proposes_training(self):
        train = self.fixture.run / 'sft-data/train.jsonl'
        train.write_text('tampered')
        with self.assertRaises(ValueError): self.build()
        self.assertFalse((self.fixture.run / 'gen1-adapter').exists())

    def test_existing_adapter_and_invalid_source_fail(self):
        with self.assertRaises(ValueError): self.build(expected_source={'run_id': 'wrong'})
        (self.fixture.run / 'gen1-adapter').mkdir()
        with self.assertRaises(ValueError): self.build()

    def test_manifest_split_tampering_is_rejected(self):
        path = self.fixture.run / 'sft-data/manifest.json'
        value = json.loads(path.read_text())
        value['split_task_ids']['valid'] = value['split_task_ids']['train'][:1]
        path.write_text(json.dumps(value)); path.chmod(0o600)
        with self.assertRaises(ValueError): self.build()

    def test_overfitting_only_omits_validation_and_is_labeled(self):
        self.fixture.tmp.cleanup()
        self.fixture = fixtures.SFTMaterializeTests('test_overfit_case_omits_empty_validation_file')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.model = self.fixture.root / 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'; self.model.mkdir()
        f = self.fixture.fixture.fixture
        rows = [dataset_fixtures.row('a', [f.good, f.bad])] + [dataset_fixtures.row(x, [f.bad, None]) for x in ('b', 'c', 'd')]
        self.fixture.materialize(f.build(rows))
        result = self.build()
        self.assertEqual(result['yield_status'], 'overfitting_only')
        self.assertEqual(result['trainer_config']['val_batches'], 0)
        self.assertFalse((self.fixture.run / 'sft-data/valid.jsonl').exists())


if __name__ == '__main__': unittest.main()

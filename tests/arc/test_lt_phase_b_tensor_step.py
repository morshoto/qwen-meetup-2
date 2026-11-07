import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.lt_phase_b_tensor_step import observe_one_step
from tests.arc.test_lt_phase_b_tensor_evidence import Model


class TensorStepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = Model()
        self.patcher = patch('src.arc.lt_phase_b_tensor_evidence._expected_shapes',
                             return_value={'layer.lora_a': [2, 1], 'layer.lora_b': [1, 2]})
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def observe(self, step):
        return observe_one_step(self.model, output_dir=self.root, run_step=step,
                                flatten=lambda x: list(x.items()), evaluate=lambda _: None)

    def test_step_saves_before_after_and_unverified_comparison(self):
        result = self.observe(lambda: self.model.params['layer.lora_b'].__setitem__((0, 0), .25))
        self.assertEqual(result['changed_adapter_tensor_count'], 1)
        self.assertFalse(result['training_verified'])
        self.assertEqual(json.loads((self.root / 'tensor-evidence.json').read_text()), result)
        self.assertTrue((self.root / 'tensor-before.json').is_file())
        self.assertTrue((self.root / 'tensor-after.json').is_file())
        with self.assertRaises(FileExistsError): self.observe(lambda: None)

    def test_failed_or_base_changing_step_keeps_partial_evidence(self):
        def fail(): raise RuntimeError('fixture failure')
        with self.assertRaises(RuntimeError): self.observe(fail)
        self.assertTrue((self.root / 'tensor-before.json').is_file())
        self.assertFalse((self.root / 'tensor-evidence.json').exists())
        with tempfile.TemporaryDirectory() as other:
            def mutate():
                self.model.params['base.weight'][0, 0] = 4
                self.model.params['layer.lora_b'][0, 0] = .25
            with self.assertRaises(ValueError):
                observe_one_step(self.model, output_dir=Path(other), run_step=mutate,
                                 flatten=lambda x: list(x.items()), evaluate=lambda _: None)
            self.assertTrue((Path(other) / 'tensor-after.json').is_file())
            self.assertFalse((Path(other) / 'tensor-evidence.json').exists())


if __name__ == '__main__': unittest.main()

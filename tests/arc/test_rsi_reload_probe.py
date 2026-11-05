"""The reload probe checks identities and all loaded tensors, without authorizing training."""
import unittest
from unittest.mock import Mock, patch

from src.arc.rsi_adapter_reload import observe_model_reload


class ReloadProbeTests(unittest.TestCase):
    def test_validates_identities_around_single_adapter_load(self):
        model = Mock()
        model.parameters.return_value = 'parameters'
        load = Mock(return_value=(model, object()))
        flatten = Mock(return_value=[('base.weight', object()), ('layer.lora_a', 'a'), ('layer.lora_b', 'b')])
        evaluate = Mock()
        convert = Mock(side_effect=lambda value: value)
        expected = {'status':'loaded_adapter_factors_match_unverified','matched_tensor_count':2,
                    'reload_verified':False,'training_allowed':False,'promotion_allowed':False}
        with patch('src.arc.rsi_adapter_reload.verify_model_manifest') as verify_model, \
             patch('src.arc.rsi_adapter_reload.inspect_adapter_structure') as inspect, \
             patch('src.arc.rsi_adapter_reload.compare_reloaded_factors',return_value=expected) as compare:
            result=observe_model_reload(model_key='32b',model_dir='/model',adapter_dir='/adapter',
                expected_adapter_sha256='a'*64,model_manifest={'model_id':'example'},
                load_model=load,flatten_parameters=flatten,evaluate=evaluate,to_numpy=convert)
            self.assertEqual(verify_model.call_count,2)
            inspect.assert_called_once_with('/adapter',expected_sha256='a'*64,model_key='32b')
            load.assert_called_once_with('/model',adapter_path='/adapter',tokenizer_config={'trust_remote_code':False})
            self.assertEqual(evaluate.call_count,2)
            compare.assert_called_once_with('/adapter',expected_sha256='a'*64,model_key='32b',
                                             loaded_factors={'layer.lora_a':'a','layer.lora_b':'b'})
            self.assertFalse(result['reload_verified'])
            self.assertFalse(result['training_allowed'])
            self.assertFalse(result['promotion_allowed'])

    def test_duplicate_names_and_changed_model_fail(self):
        model = Mock(); model.parameters.return_value = object()
        args=dict(model_key='14b',model_dir='/model',adapter_dir='/adapter',expected_adapter_sha256='a'*64,
                  model_manifest={},load_model=Mock(return_value=(model, object())),evaluate=Mock(),to_numpy=lambda x:x)
        with patch('src.arc.rsi_adapter_reload.verify_model_manifest'), \
             patch('src.arc.rsi_adapter_reload.inspect_adapter_structure'):
            with self.assertRaises(ValueError):
                observe_model_reload(**args,flatten_parameters=lambda _: [('x.lora_a',1),('x.lora_a',2)])
        with patch('src.arc.rsi_adapter_reload.verify_model_manifest',side_effect=[None,ValueError('model drift')]), \
             patch('src.arc.rsi_adapter_reload.inspect_adapter_structure'), \
             patch('src.arc.rsi_adapter_reload.compare_reloaded_factors',return_value={}):
            with self.assertRaises(ValueError):
                observe_model_reload(**args,flatten_parameters=lambda _: [('x.lora_a',1)])

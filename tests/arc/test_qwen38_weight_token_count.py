import numpy as np
import pytest

from src.arc.qwen38_weight_token_count import trainer_token_count


class FakeDataset:
    def __init__(self, item): self.item = item
    def __getitem__(self, index): return self.item


def test_count_uses_actual_trainer_inputs_and_completion_mask():
    item = {'input_ids': np.array([[1, 2, 3, 4, 5]]), 'attention_mask': np.ones((1, 5), dtype=int),
            'completion_mask': np.array([[0, 0, 0, 1, 1]])}
    assert trainer_token_count([{'role': 'user'}, {'role': 'assistant'}], lambda _: FakeDataset(item)) == 5


@pytest.mark.parametrize('mask', [None, [[0, 0, 0, 0, 0]], [[0, 1, 0, 1, 1]], [[0, 0, 0, 2, 1]]])
def test_count_fails_if_completion_mask_is_missing_or_unsafe(mask):
    item = {'input_ids': np.array([[1, 2, 3, 4, 5]]), 'attention_mask': np.ones((1, 5), dtype=int)}
    if mask is not None: item['completion_mask'] = np.array(mask)
    with pytest.raises(ValueError): trainer_token_count([], lambda _: FakeDataset(item))


def test_unsigned_mask_cannot_hide_unmasked_prompt():
    item = {'input_ids': np.array([[1, 2, 3, 4, 5]]), 'attention_mask': np.ones((1, 5), dtype=int),
            'completion_mask': np.array([[0, 1, 0, 1, 1]], dtype=np.uint8)}
    with pytest.raises(ValueError): trainer_token_count([], lambda _: FakeDataset(item))


def test_counter_initializes_processor_as_mlx_vlm_load_does(tmp_path, monkeypatch):
    import sys
    import types
    from src.arc.qwen38_local_runner import MODEL_REVISION
    from src.arc.qwen38_weight_token_count import make_trainer_counter

    model = tmp_path / MODEL_REVISION
    model.mkdir()
    (model / 'config.json').write_text('{}')
    calls = []
    image_processor = object()
    processor = types.SimpleNamespace(image_processor=None)
    def load_image(path):
        calls.append(('image', path))
        return image_processor
    def load_processor(path, add_detokenizer):
        assert calls == [('image', model)]
        calls.append(('processor', path))
        return processor
    class Dataset:
        def __init__(self, rows, config, actual_processor, train_on_completions):
            assert actual_processor.image_processor is image_processor
            assert train_on_completions is True
        def __getitem__(self, index):
            return {'input_ids': np.array([[1, 2]]), 'attention_mask': np.array([[1, 1]]),
                    'completion_mask': np.array([[0, 1]])}
    monkeypatch.setitem(sys.modules, 'mlx_vlm.utils', types.SimpleNamespace(
        load_image_processor=load_image, load_processor=load_processor))
    monkeypatch.setitem(sys.modules, 'mlx_vlm.trainer.datasets', types.SimpleNamespace(VisionDataset=Dataset))
    assert make_trainer_counter(model)([]) == 2
    assert [call[0] for call in calls] == ['image', 'processor']

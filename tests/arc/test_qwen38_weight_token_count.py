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

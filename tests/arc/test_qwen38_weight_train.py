import pytest

from src.arc.qwen38_weight_train import parse_training_losses


def test_loss_log_requires_every_finite_training_step():
    log = '\n'.join(f'Iter {i}: Train loss \x1b[92m{i / 10:.2f}\x1b[0m, Learning Rate' for i in range(1, 21))
    losses = parse_training_losses(log, steps=20)
    assert len(losses) == 20 and losses[0] == .1 and losses[-1] == 2.0


@pytest.mark.parametrize('log', ['Iter 1: Train loss nan, Learning Rate',
                                 'Iter 1: Train loss 1,\nIter 1: Train loss 2,',
                                 'Iter 2: Train loss 1,', ''])
def test_loss_log_rejects_missing_duplicate_nonfinite_and_wrong_steps(log):
    with pytest.raises(ValueError): parse_training_losses(log, steps=1)

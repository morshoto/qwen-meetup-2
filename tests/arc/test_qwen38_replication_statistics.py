import pytest

from src.arc.qwen38_replication_statistics import paired_sign_test


def test_exact_sign_test_uses_only_discordant_pairs():
    assert paired_sign_test(4, 1) == {'gained': 4, 'lost': 1, 'discordant': 5, 'two_sided_p': 0.375}
    assert paired_sign_test(0, 0) == {'gained': 0, 'lost': 0, 'discordant': 0, 'two_sided_p': 1.0}
    assert paired_sign_test(0, 5)['two_sided_p'] == 0.0625
    assert paired_sign_test(5, 0) == {'gained': 5, 'lost': 0, 'discordant': 5, 'two_sided_p': 0.0625}


@pytest.mark.parametrize('gain,loss', [(-1, 0), (0, -1), (True, 0), (0, 1.0)])
def test_sign_test_rejects_invalid_counts(gain, loss):
    with pytest.raises(ValueError): paired_sign_test(gain, loss)

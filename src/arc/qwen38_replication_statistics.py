"""Exact discordant-pair uncertainty for small exploratory comparisons."""
from math import comb


def paired_sign_test(gained, lost):
    """Two-sided exact McNemar/sign test under equal discordant probabilities."""
    if type(gained) is not int or type(lost) is not int or gained < 0 or lost < 0:
        raise ValueError('invalid discordant-pair counts')
    n = gained + lost
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(gained, lost) + 1)) / (2 ** n))
    return {'gained': gained, 'lost': lost, 'discordant': n, 'two_sided_p': p}

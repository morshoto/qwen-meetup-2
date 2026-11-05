"""Deterministic descriptive paired summary of supplied task outcomes.

The caller must establish provenance, matched execution and held-out scoring;
this function cannot prove that supplied booleans are real experiment results.
"""
import math
import random


def paired_summary(base: dict[str, bool], candidate: dict[str, bool]) -> dict:
    if (not isinstance(base, dict) or not isinstance(candidate, dict) or not base or
            set(base) != set(candidate) or
            any(not isinstance(key, str) or not key for key in base) or
            any(type(value) is not bool for value in [*base.values(), *candidate.values()])):
        raise ValueError('paired task outcomes must have identical nonempty IDs and boolean results')
    ids = sorted(base)
    differences = [int(candidate[key]) - int(base[key]) for key in ids]
    n = len(ids)
    delta = 100.0 * sum(differences) / n
    rng = random.Random(42)
    samples = sorted(100.0 * sum(differences[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(10000))
    # Nearest-rank percentile interval; rank 1 maps to index 0.
    lower = samples[math.ceil(0.025 * len(samples)) - 1]
    upper = samples[math.ceil(0.975 * len(samples)) - 1]
    return {'evidence_type': 'supplied_task_outcomes', 'task_count': n,
            'base_solved': sum(base.values()), 'candidate_solved': sum(candidate.values()),
            'difference_percentage_points': delta,
            'bootstrap_95_percentage_points': [lower, upper],
            'bootstrap_resamples': 10000, 'bootstrap_seed': 42,
            'improvement_criterion_met': delta > 0 and lower > 0}

"""Conservative per-launch budget calculations; not a provider spending cap."""
import math


class Budget:
    def __init__(self, *, minutes, units, rate_per_hour):
        for value in (minutes, units, rate_per_hour):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('budget values must be finite and positive')
        self.limit_seconds = min(minutes * 60, units / rate_per_hour * 3600)
        self.rate_per_hour = rate_per_hour

    def task_timeout(self, elapsed, task_limit):
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid elapsed time')
        if type(task_limit) is not int or task_limit <= 0:
            raise ValueError('invalid task limit')
        remaining = int(self.limit_seconds - elapsed)
        return min(remaining, task_limit) if remaining >= 30 else None

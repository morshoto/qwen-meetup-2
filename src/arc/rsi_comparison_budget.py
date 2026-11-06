"""Shared comparison ceiling; caller must separately authorize execution."""
from src.arc.rsi_collection_budget import CollectionBudget


class ComparisonBudget(CollectionBudget):
    max_model_calls = 800
    max_reserved_tokens = 16000000
    max_sandbox_runs = 8000
    max_wall_seconds = 36000
    model_kinds = frozenset(('evaluation',))

    def __init__(self, root, identity):
        super().__init__(root, identity)
        self.name = f"g{identity['generation']}.comparison-budget.json"
        self.lock_name = f"g{identity['generation']}.comparison-budget.lock"

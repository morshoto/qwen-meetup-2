"""Proposed separate local limits for the 50-task exploratory comparison."""
from src.arc.rsi_collection_budget import CollectionBudget


class ExploratoryBudget(CollectionBudget):
    max_model_calls = 200
    max_reserved_tokens = 4000000
    max_sandbox_runs = 2000
    max_wall_seconds = 9000
    model_kinds = frozenset(('evaluation',))

    def __init__(self, root, identity):
        super().__init__(root, identity)
        self.name = f"g{identity['generation']}.exploratory-budget.json"
        self.lock_name = f"g{identity['generation']}.exploratory-budget.lock"

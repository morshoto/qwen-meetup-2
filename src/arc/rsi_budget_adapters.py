"""Connect cooperating components to a budget; does not authorize execution."""


def model_reserver(budget, *, kind):
    if kind not in ('generation', 'canary'):
        raise ValueError('invalid collection call kind')

    def reserve(tokens):
        budget.reserve_model(tokens, kind=kind)

    return reserve


class BudgetedSandbox:
    def __init__(self, sandbox, budget):
        self.sandbox = sandbox
        self.budget = budget

    def run(self, code, grid):
        self.budget.reserve_sandbox()
        return self.sandbox.run(code, grid)

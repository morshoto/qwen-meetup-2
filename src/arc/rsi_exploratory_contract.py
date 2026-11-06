"""Apply the guarded server contract to the exploratory local budget."""
from src.arc.rsi_comparison_contract import ContractCheckedComparisonTransport
from src.arc.rsi_exploratory_budget import ExploratoryBudget
from src.arc.rsi_exploratory_transport import CapturedExploratoryTransport


class ContractCheckedExploratoryTransport(ContractCheckedComparisonTransport):
    budget_type = ExploratoryBudget
    token_limit = 4000000

    def __init__(self, transport, *, budget, registration_sha256, generation, expected_max_prompt_tokens):
        if type(transport) is not CapturedExploratoryTransport:
            raise ValueError('captured exploratory transport required')
        self.exploratory_transport = transport
        super().__init__(transport.transport, budget=budget, registration_sha256=registration_sha256,
                         generation=generation, expected_max_prompt_tokens=expected_max_prompt_tokens)

    def __call__(self, prompt, *, arm, adapter, task_id, sample, seed, timeout):
        if arm not in ('base', 'trained'):
            raise ValueError('invalid exploratory arm')
        return super().__call__(prompt, arm='parent' if arm == 'base' else 'candidate', adapter=adapter,
                                task_id=task_id, sample=sample, seed=seed, timeout=timeout)

    def _dispatch(self, prompt, *, arm, adapter, task_id, sample, seed, timeout):
        return self.exploratory_transport(prompt, arm='base' if arm == 'parent' else 'trained', adapter=adapter,
                                          task_id=task_id, sample=sample, seed=seed, timeout=timeout)

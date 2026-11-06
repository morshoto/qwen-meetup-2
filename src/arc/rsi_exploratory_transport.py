"""Bind exploratory arm names to the captured local comparison transport."""
from src.arc.rsi_collected_slot_audit import _index
from src.arc.rsi_comparison_audit import _read_jsonl
from src.arc.rsi_comparison_transport import CapturedComparisonTransport
from src.arc.rsi_comparison_usage import _summarize_usage
from src.arc.rsi_slot_lifecycle import _directory


class CapturedExploratoryTransport:
    """Expose server-reported usage without asserting identity or provenance."""
    def __init__(self, transport):
        if type(transport) is not CapturedComparisonTransport:
            raise ValueError('captured comparison transport required')
        self.transport = transport

    def __call__(self, prompt, *, arm, adapter, task_id, sample, seed, timeout):
        if arm not in ('base', 'trained'):
            raise ValueError('invalid exploratory arm')
        mapped = 'parent' if arm == 'base' else 'candidate'
        response = self.transport(prompt, arm=mapped, adapter=adapter, task_id=task_id,
                                  sample=sample, seed=seed, timeout=timeout)
        with _directory(self.transport.root / f'{mapped}.responses') as directory:
            events = _index(_read_jsonl(directory, 'requests.jsonl'))
        event = events.get(response.get('request_id'))
        if event is None:
            raise ValueError('captured request event unavailable')
        _summarize_usage([event])
        usage = event.get('usage')
        counts = ({key:usage[key] for key in ('prompt_tokens','completion_tokens')}
                  if isinstance(usage,dict) and all(key in usage for key in ('prompt_tokens','completion_tokens'))
                  else None)
        return {**response, 'usage':counts}

"""Journal monitored 14B collection attempts one task at a time.

Callers must separately verify the frozen protocol, inputs and public export and
serialize concurrent access to the journal. No live execution is authorized by
this orchestration. Worker observations are local evidence, not attestation.
"""
import json
import math

from src.arc.lt_batch_journal import BatchJournal, _create
from src.arc.lt_supervised_batch import supervised_batch
from src.arc.rsi_training_resources import training_resource_decision, valid_resource_limits


def collect_batch(journal, *, limit, worker, resource_limits=None):
    if (not isinstance(journal, BatchJournal) or type(limit) is not int or limit <= 0 or
            not callable(worker) or (resource_limits is not None and not valid_resource_limits(resource_limits))):
        raise ValueError('invalid collection settings')
    journal.pending(1)  # Validate all prior journal entries before resuming.
    for task in journal.ids:
        result = journal.root / (task + '.result.json')
        if result.exists():
            status = json.loads(result.read_bytes()).get('status')
            if status in ('error', 'timeout'):
                return {'completed': 0, 'stop_reason': 'worker_' + status}
            if status != 'complete':
                raise ValueError('invalid prior result status')
    completed = 0
    for _ in range(limit):
        if not journal.pending(1):
            return {'completed': completed, 'stop_reason': 'complete'}
        observed = []

        def guarded(task, output):
            def on_start(pid):
                if type(pid) is not int or pid <= 0:
                    raise ValueError('invalid owned worker pid')
                _create(journal.root / (task + '.process.json'),
                        {'task_id': task, 'identity': journal.identity, 'pid': pid})
            value = worker(task, output, on_start)
            if not isinstance(value, dict):
                raise ValueError('invalid worker observation')
            _create(journal.root / (task + '.resources.json'), value)
            if value.get('status') not in ('complete', 'error', 'timeout'):
                raise ValueError('worker preflight or execution refused')
            if not (journal.root / (task + '.process.json')).exists():
                raise ValueError('worker process identity missing')
            if value['status'] == 'complete':
                observation, baseline = value.get('observation'), value.get('baseline')
                if not isinstance(observation, dict):
                    raise ValueError('worker monitoring unavailable')
                elapsed = observation.get('elapsed_seconds', value.get('elapsed_seconds'))
                timeout = resource_limits['timeout_seconds'] if resource_limits else 300
                if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= timeout or
                        observation.get('stop_reason') != 'process_exited' or observation.get('exit_code') != 0 or
                        observation.get('stream_complete') is not True or
                        observation.get('group_signals_confirmed') is not True or
                        not isinstance(observation.get('cleanup_observation'), dict) or
                        observation['cleanup_observation'].get('group_cleanup_verified') is not True or
                        training_resource_decision(baseline, observation.get('samples'), limits=resource_limits) is not None):
                    raise ValueError('worker resource evidence invalid')
            return value

        supervised_batch(journal, '14b', 1, guarded, on_result=lambda task, status: observed.append(status))
        completed += 1
        if observed != ['complete']:
            if len(observed) != 1 or observed[0] not in ('error', 'timeout'):
                raise ValueError('invalid supervised worker result')
            return {'completed': completed, 'stop_reason': 'worker_' + observed[0]}
    return {'completed': completed, 'stop_reason': 'complete' if not journal.pending(1) else 'batch_limit'}

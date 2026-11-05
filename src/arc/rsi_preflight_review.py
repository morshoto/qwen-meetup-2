"""Review local preflight observations without approving a training run."""
import math

from src.arc.rsi_training_resources import training_resource_decision, valid_resource_limits


def review_preflight_observation(*, baseline, observation, resource_limits=None):
    gates = ['execution_provenance', 'process_containment_proof', 'independent_adapter_reload',
             'preflight_registration', 'training_authorization']
    def reject(reason):
        return {'status': 'preflight_observation_rejected', 'training_allowed': False,
                'reason': reason, 'remaining_gates': gates}
    if resource_limits is not None and not valid_resource_limits(resource_limits):
        return reject('invalid_resource_limits')
    if not isinstance(observation, dict):
        return reject('invalid_observation')
    if (observation.get('stop_reason') != 'process_exited' or type(observation.get('exit_code')) is not int or
            observation['exit_code'] != 0 or observation.get('stream_complete') is not True or
            observation.get('group_signals_confirmed') is not True or observation.get('training_verified') is not False):
        return reject('process_or_stream_incomplete')
    elapsed = observation.get('elapsed_seconds')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= (resource_limits['timeout_seconds'] if resource_limits is not None else 300):
        return reject('invalid_elapsed_time')
    loss = observation.get('loss')
    if (not isinstance(loss, dict) or loss.get('status') != 'ok' or type(loss.get('last_step')) is not int or
            loss['last_step'] != 1 or type(loss.get('last_loss')) not in (int, float) or
            not math.isfinite(loss['last_loss'])):
        return reject('invalid_loss_observation')
    samples = observation.get('samples')
    resource_reason = training_resource_decision(baseline, samples, limits=resource_limits)
    if resource_reason is not None:
        return reject(resource_reason)
    if samples[-1]['monotonic_seconds'] - baseline['monotonic_seconds'] > (resource_limits['timeout_seconds'] if resource_limits is not None else 300):
        return reject('sample_window_exceeded')
    return {'status': 'preflight_observation_consistent_unverified', 'training_allowed': False,
            'observed_loss': loss['last_loss'], 'elapsed_seconds': elapsed,
            'remaining_gates': gates, 'evidence_type': 'caller_supplied_process_and_samples'}

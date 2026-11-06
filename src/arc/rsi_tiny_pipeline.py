"""Opt-in tiny fabricated-model integration fixture; not an RSI experiment."""
import json
import os
import subprocess
from pathlib import Path

from src.arc.rsi_canary_containment import fork_restricted_command
from src.arc.rsi_process_sampling import sample_baseline, sample_process
from src.arc.rsi_training_resources import training_resource_decision
from src.arc.rsi_training_stream import watch_training_stream
from src.arc.rsi_reload_stream import watch_reload_stream
from src.arc.rsi_descendant_tracking import begin_owned_observation, sample_process_rows
from src.arc.rsi_process_monitor import _exited_unreaped, _stop_group


def cleanup_fixture_process(process):
    """Stop only an owned unreaped session; observations do not verify cleanup."""
    if not isinstance(process, subprocess.Popen):
        raise ValueError('owned process required')
    if process.returncode is not None:
        return {'status': 'already_reaped_unverified', 'cleanup_verified': False,
                'escaped_processes_excluded': False}
    try:
        exited = _exited_unreaped(process)
        if exited is None:
            if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
                raise ValueError('owned session leader required')
        else:
            from src.arc.rsi_group_observation import observe_exited_group
            try:
                observe_exited_group(process)  # requires unreaped leader in its own group
            except ValueError:
                # waitid already established that this owned leader has exited.
                # A missing group observation cannot justify a group signal.
                code = process.wait(timeout=1.0)
                return {'status': 'leader_reaped_group_unverified', 'exit_code': code,
                        'group_signals_confirmed': False, 'cleanup_verified': False,
                        'escaped_processes_excluded': False}
        exit_code, signals, observation = _stop_group(process, 1.0)
    except (OSError, ChildProcessError) as exc:
        raise ValueError('owned process identity unavailable') from exc
    return {'status': 'fixture_stop_observed_unverified', 'exit_code': exit_code,
            'group_signals_confirmed': signals, 'group_observation': observation,
            'cleanup_verified': False, 'escaped_processes_excluded': False}


def run_tiny_pipeline_fixture(*, root, python_executable):
    root = Path(root).resolve(strict=True)
    model, data, adapter = root/'fabricated-model', root/'synthetic', root/'tiny-adapter'
    expected = {'model_type':'qwen3', 'hidden_size':16, 'num_hidden_layers':2, 'intermediate_size':32,
                'num_attention_heads':2, 'rms_norm_eps':1e-6, 'vocab_size':12, 'num_key_value_heads':1,
                'max_position_embeddings':128, 'rope_theta':10000.0, 'head_dim':8, 'tie_word_embeddings':True}
    if (json.loads((model/'config.json').read_text()) != expected or
            (model/'model.safetensors').stat().st_size > 1024*1024 or adapter.exists() or
            json.loads((data/'train.jsonl').read_text()) !=
            {'messages':[{'role':'user','content':'hello world'}, {'role':'assistant','content':'answer yes'}]}):
        raise ValueError('tiny fixture must use the fabricated model and fixed synthetic data')
    source = Path(__file__).resolve().parents[2]
    python = str(Path(python_executable).absolute())
    config = {'model':str(model),'train':True,'fine_tune_type':'lora','optimizer':'adam','data':str(data),
              'seed':42,'num_layers':2,'batch_size':1,'iters':1,'val_batches':0,'learning_rate':0.0001,
              'steps_per_report':1,'steps_per_eval':2,'resume_adapter_file':None,'adapter_path':str(adapter),
              'save_every':1,'max_seq_length':128,'grad_checkpoint':False,
              'lora_parameters':{'rank':8,'scale':2.0,'dropout':0.0},'mask_prompt':True,'report_to':None}
    config_file = root/'tiny-config.json'
    config_file.write_text(json.dumps(config))
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    command = fork_restricted_command([python,'-u','-B','-m','src.cli.p3b_train','--config',str(config_file),
                                       '--thinking-mode','off'])['argv']
    baseline = sample_baseline(root)
    if training_resource_decision(baseline, [baseline]) is not None:
        raise RuntimeError('host outside fixture resource limits')
    process = subprocess.Popen(command, cwd=source, env=env, start_new_session=True,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               close_fds=True)
    try:
        tracker, initial_identity = begin_owned_observation(process)
        result = watch_training_stream(process, baseline=baseline, sampler=lambda pid: sample_process(root,pid),
                                       timeout_seconds=60, interval_seconds=0.25, sample_timeout_seconds=2,
                                       terminate_grace_seconds=1,
                                       identity_sampler=lambda _: sample_process_rows(),
                                       identity_processor=tracker.observe)
    finally:
        try:
            cleanup_fixture_process(process)
        finally:
            process.stdout.close()
    result.update(tiny_adapter_reloaded=False, tiny_reloaded_factor_matches=0, promotion_allowed=False,
                  tiny_initial_identity=initial_identity, tiny_reload_cleanup_verified=False)
    if result['stop_reason'] != 'process_exited' or result['exit_code'] != 0 or not result['stream_complete']:
        return result
    if (result['group_signals_confirmed'] is not True or
            result['cleanup_observation'].get('group_cleanup_verified') is not True or
            result['escaped_processes_excluded'] is not True):
        result['tiny_reload_stop_reason'] = 'not_run_cleanup_unverified'
        return result
    probe = """import json,sys
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.utils import load
model,_=load(sys.argv[1],adapter_path=sys.argv[2],tokenizer_config={'trust_remote_code':False})
loaded={key:value for key,value in tree_flatten(model.parameters()) if key.endswith(('.lora_a','.lora_b'))}
saved=mx.load(sys.argv[2]+'/adapters.safetensors')
assert len(saved)==28 and set(loaded)==set(saved)
for key,value in saved.items():
    mx.eval(value,loaded[key])
    assert np.array_equal(np.asarray(value),np.asarray(loaded[key]))
print(json.dumps({'tiny_adapter_reloaded':True,'tiny_reloaded_factor_matches':len(saved)}))
"""
    reload_command = fork_restricted_command([python,'-B','-c',probe,str(model),str(adapter)])['argv']
    reload_baseline = sample_baseline(root)
    if training_resource_decision(reload_baseline, [reload_baseline]) is not None:
        raise RuntimeError('host outside reload fixture resource limits')
    reload_process = subprocess.Popen(reload_command, cwd=source, env=env, start_new_session=True,
                                      stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      close_fds=True)
    try:
        reload_tracker, reload_initial_identity = begin_owned_observation(reload_process)
        reload_result = watch_reload_stream(reload_process, baseline=reload_baseline,
            sampler=lambda pid: sample_process(root,pid), timeout_seconds=60, interval_seconds=0.25,
            sample_timeout_seconds=2, terminate_grace_seconds=1,
            identity_sampler=lambda _: sample_process_rows(), identity_processor=reload_tracker.observe)
    finally:
        try:
            cleanup_fixture_process(reload_process)
        finally:
            reload_process.stdout.close()
    result.update(tiny_reload_stop_reason=reload_result['stop_reason'],
                  tiny_reload_sample_count=len(reload_result['samples']),
                  tiny_reload_identity_sample_count=len(reload_result['identity_observations']),
                  tiny_reload_initial_identity=reload_initial_identity,
                  tiny_reload_elapsed_seconds=reload_result['elapsed_seconds'],
                  tiny_reload_cleanup_verified=False)
    try:
        parsed = json.loads(reload_result['output'])
    except (ValueError, TypeError):
        parsed = None
    if (reload_result['stop_reason'] == 'process_exited' and reload_result['exit_code'] == 0 and
            reload_result['stream_complete'] and
            parsed == {'tiny_adapter_reloaded':True, 'tiny_reloaded_factor_matches':28}):
        result.update(parsed)
    return result

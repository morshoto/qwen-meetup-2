"""Opt-in tiny fabricated-model integration fixture; not an RSI experiment."""
import json
import os
import subprocess
from pathlib import Path

from src.arc.rsi_canary_containment import fork_restricted_command
from src.arc.rsi_process_sampling import sample_baseline, sample_process
from src.arc.rsi_training_resources import training_resource_decision
from src.arc.rsi_training_stream import watch_training_stream


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
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        result = watch_training_stream(process, baseline=baseline, sampler=lambda pid: sample_process(root,pid),
                                       timeout_seconds=60, interval_seconds=0.25, sample_timeout_seconds=2,
                                       terminate_grace_seconds=1)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait()
        process.stdout.close()
    result.update(tiny_adapter_reloaded=False, tiny_reloaded_factor_matches=0, promotion_allowed=False)
    if result['stop_reason'] != 'process_exited' or result['exit_code'] != 0 or not result['stream_complete']:
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
    reload_result = subprocess.run(reload_command, cwd=source, env=env, capture_output=True, text=True,
                                   timeout=60, start_new_session=True)
    try:
        parsed = json.loads(reload_result.stdout)
    except (ValueError, TypeError):
        parsed = None
    if reload_result.returncode == 0 and parsed == {'tiny_adapter_reloaded':True, 'tiny_reloaded_factor_matches':28}:
        result.update(parsed)
    return result

import json
from pathlib import Path
cells = [
('markdown', '# Qwen3.8-27B H100 Colab smoke test\n\nDevelopment-only generation check. This is not the Phase A benchmark. The first cell refuses to proceed unless an H100 is attached. Runtimes consume Colab compute units while active.'),
('code', '''import os, subprocess, sys, time
print("UTC:", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
print("Python:", sys.version)
result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True)
print("GPU:", result.stdout.strip())
assert "H100" in result.stdout, "H100 required; change the runtime and retry."
print(subprocess.run(["gcc", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0])'''),
('code', '''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.33.0", "transformers>=5.8.0", "huggingface_hub", "cuda-toolkit[nvcc]==13.0.3.0", "ninja"], check=True)'''),
('code', '''import os, sys, time, pathlib, importlib.metadata
import nvidia
cuda_home = pathlib.Path(next(iter(nvidia.__path__))) / "cu13"
os.environ["CUDA_HOME"] = str(cuda_home)
os.environ["PATH"] = str(pathlib.Path(sys.executable).parent) + os.pathsep + str(cuda_home / "bin") + os.pathsep + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = str(cuda_home / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
for target, link in [("lib", cuda_home / "lib64"), ("libcudart.so.13", cuda_home / "lib" / "libcudart.so")]:
    if not link.exists(): link.symlink_to(target)
import subprocess
subprocess.run(["nvcc", "--version"], check=True)
print("vllm:", importlib.metadata.version("vllm"), "transformers:", importlib.metadata.version("transformers"))
from huggingface_hub import snapshot_download
started = time.monotonic()
model = snapshot_download("Qwen/Qwen3.8-27B", revision="37561816e66b5b9153b1b64c10968c18c1e6f4a0")
print("Pinned model downloaded in seconds:", round(time.monotonic()-started, 2))
from vllm import LLM, SamplingParams
started = time.monotonic()
llm = LLM(model=model, trust_remote_code=False, dtype="bfloat16", max_model_len=16384, tensor_parallel_size=1, language_model_only=True, enforce_eager=True, max_num_seqs=1)
print("Model load seconds:", round(time.monotonic()-started, 2))
tokenizer = llm.get_tokenizer()
prompt = tokenizer.apply_chat_template([{"role":"user","content":"Reply with the single word READY."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
started = time.monotonic()
response = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)[0]
print("Generation seconds:", round(time.monotonic()-started, 2))
print("Prompt tokens:", len(response.prompt_token_ids), "Output tokens:", len(response.outputs[0].token_ids), "Finish reason:", response.outputs[0].finish_reason)
print("Model response:", repr(response.outputs[0].text))
assert response.outputs[0].text.strip(), "Empty response"
print("H100 Qwen3.8 generation smoke test: SUCCESS")''')]
notebook = {'cells': [{'cell_type':kind, 'metadata':{}, 'source':source.splitlines(keepends=True), **({'execution_count':None,'outputs':[]} if kind=='code' else {})} for kind,source in cells], 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'gpuType':'H100'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
path=Path(__file__).with_name('qwen38-h100-smoke.ipynb')
path.write_text(json.dumps(notebook, indent=2)+'\n')
print(path)

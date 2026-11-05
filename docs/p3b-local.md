# P3-B local QLoRA

P3-B is separate from the normal `experiment` command. Fake/P3-A profiles keep
`rsi.trainer: dry_run`; a training run requires both `rsi.trainer: mlx_lora`
and `p3b.training.backend: mlx_lora` in the explicit P3-B profile. The model
directory is local and the trainer starts with Hugging Face offline switches.

The implementation calls the official `mlx_lm lora` command in a native ARM
Python environment. MLX-LM treats a quantized model path as QLoRA, accepts chat
JSONL, supports completion-only loss masking, and resumes from a saved adapter.
See the [official MLX-LM LoRA/QLoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).
The synthetic acceptance profile pins MLX-LM 0.30.7 and the local Qwen3-4B-4bit
snapshot revision. Real Exploration collection is recorded separately below;
its later Qwen3-8B pilot uses its own pinned base revision and matching configs.

## P3-B1: synthetic adapter smoke test

The synthetic pilot uses eight artificial prompt/target pairs and does not load
ARC examples or labels. It writes its own run under `artifacts/`:

```sh
uv run --offline python -m src.cli.p3b synthetic-pilot \
  --config configs/p3b.local.qlora.yml
```

The run records all training losses, elapsed time, peak child RSS, MLX/MLX-LM
versions, adapter SHA-256, base model weight hashes before and after, and a
base-versus-adapter inference probe. It marks `trained` and `weights_updated`
true only when a saved nonzero LoRA B tensor and a nonzero forward-logit delta
are measured and the base safetensors hashes remain unchanged. The trained
adapter is also loaded through the guarded HTTP server during acceptance; the
server contract binds the adapter SHA-256 before accepting inference.

Latest synthetic-only acceptance: `artifacts/p3b1-synthetic-d6299ff5b737`.
Eight artificial records trained for 20 steps (loss 8.997 to 0.042, 3.06 s,
2,612,494,336-byte peak child RSS) against the Qwen3-4B base revision
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`. Base weights kept the same SHA-256
before and after (`e240c0bdc0ebb0681bf0da0f98d9719fd6ebe269a3633f81542c13e81345651d`);
adapter SHA-256 is
`99f796ebe734b5ec0f8e13ef43795bc7330d338867cd59efeebc7ca925391168`. The
adapter changed greedy output to
`P3B1_CANARY_7F3A` with a 23.53125 maximum absolute logit delta. The guarded
HTTP server loaded that same adapter SHA and returned the canary with 29 prompt
and 12 completion tokens. `adapter_http_probe.json` is included in the sealed
artifact. This proves the synthetic adapter path only.

## P3-B2/B3: verified Exploration data

Run B alone on a fixed, disjoint 20-task Exploration window. The initial
Qwen3-4B windows cover tasks 0-219; the later Qwen3-8B windows cover tasks
240-279. Start the guarded base-model server with an empty token ledger:

```sh
/tmp/qwen-p3a-mlx-venv/bin/python -m src.cli.guarded_mlx_server \
  --model-dir /Users/shotomorisaki/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --model-id mlx-community/Qwen3-4B-4bit \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --expected-mlx-lm-version 0.30.7 \
  --expected-generate-step-sha256 1aee463145a78556a66604ace7b6d85dc10035226aa4f909cbefccfa81b49de6 \
  --ledger artifacts/p3b-b20-token-ledger.sqlite \
  --total-tokens 2000000 --max-prompt-tokens 32768 --max-new-tokens 2048 \
  --temperature 0.7 --thinking off --max-request-seconds 240 --port 8080
```

Then run the B-only collection profile:

```sh
uv run --offline python -m src.cli.experiment \
  --config configs/p3b.local.collect.yml
```

After the run, stop the server and pass its run directory to the P3-B trainer:

```sh
uv run --offline python -m src.cli.p3b train-from-run \
  --config configs/p3b.local.qlora.yml \
  --source-run artifacts/<completed-p3b-local-b20-run>
```

The collector accepts only a completed, guarded, B-only Gen 0 run with exactly
20 executed Exploration tasks. It reopens the frozen challenge and solution
files, checks the data and result hashes, reruns the independent verifier, and
rebuilds the SFT messages from correct code candidates. Test outputs are
checked to be absent from the messages. Fewer than five verified records writes
`skipped_insufficient_data`, leaves `trained=false` and `weights_updated=false`,
and preserves any previous adapter. A failed trainer similarly never promotes
its candidate.

Training output is written to a new run directory; existing adapters are never
overwritten. `--previous-adapter` is used only when a previous adapter is
explicitly provided and continuation is enabled. The adapter loader requires
an expected SHA-256; `/v1/token-budget` exposes that identity, and the guarded
client refuses a configured hash mismatch. Base and adapted generations must
run with separate guarded services because each service binds one immutable
adapter at startup.

Promotion and rollback settings are parsed and retained in the snapshot. The
original B-only run, `artifacts/p3b-local-b20-302b7536d475`, completed 20 tasks
with pass@1 0/15. The retuned run,
`artifacts/p3b-local-b20-ec3fea73fa48`, completed 20 tasks in 715.05 s with 40
candidates, 79 calls, and 726,078 client-reserved tokens. It scored 18 tasks,
had two task errors, and pass@1/pass@2 were both zero. Four HTTP 400 responses
reported `prompt token limit exceeded before model dispatch` at the prior
16,384-token cap. One candidate passed visible examples but failed the
independent test verifier, so it was not admitted to training. An earlier
200,000-token attempt stopped at 14/20 with
`budget_exhausted` and is retained at
`artifacts/p3b-local-b20-09de5188fa8a`.

The verifier rebuilt zero training records from the retuned offset-0 run,
below the five-record minimum. `artifacts/p3b3-qlora-d18c14108561` records
`skipped_insufficient_data` with `trained=false`, `weights_updated=false`, and
the previous adapter preserved. The following additional disjoint B-only
windows were run with the 32,768-token prompt cap, thinking disabled, and the
same base model revision and 2M server token budget:

| Exploration tasks | B run | Result | Training gate |
| --- | --- | --- | --- |
| 20–39 | `artifacts/p3b-local-b20-792fca0afd82` | 20 scored, 0 errors, 40 candidates, pass@1/pass@2 0 | `artifacts/p3b3-qlora-ed4f5c8dd5e4`: 0 verified records, skipped |
| 40–59 | `artifacts/p3b-local-b20-offset40-1bda92fe5c54` | 20 scored, 0 errors, 40 candidates, pass@1/pass@2 0 | `artifacts/p3b3-qlora-92b9ea09f582`: 0 verified records, skipped |
| 60–79 | `artifacts/p3b-local-b20-offset60-d2f065950341` | 20 scored, 0 errors, 40 candidates, pass@1/pass@2 0 | `artifacts/p3b3-qlora-723b6b90e02e`: 0 verified records, skipped |

The offset-40 and offset-60 comparison gates also recorded
`skipped_no_verified_adapter`, `promotion=keep_previous`, and
`previous_adapter_preserved=true` (`artifacts/p3b3-compare-f24123bfabf6` and
`artifacts/p3b3-compare-e1be988e9567`). These are safe skip decisions, not
executed B/C comparisons. Later disjoint B-only windows used the same frozen
base revision, guarded server, and public Exploration split:

| Exploration tasks | B run | Result | Training gate | Comparison gate |
| --- | --- | --- | --- | --- |
| 80–99 | `artifacts/p3b-local-b20-offset80-6dc43be50fdc` | 20 scored, 0 errors, 40 candidates, pass@1/pass@2 0 | `artifacts/p3b3-qlora-012dfd0ed36d`: 0 verified, skipped | `artifacts/p3b3-compare-7725531d093a`: skipped |
| 100–119 | `artifacts/p3b-local-b20-offset100-bf405c0bd861` | 19 scored, 1 error, 40 candidates, pass@1/pass@2 0 | `artifacts/p3b3-qlora-d37420a0790e`: 0 verified, skipped | `artifacts/p3b3-compare-dd3880a6fb4c`: skipped |
| 120–139 | `artifacts/p3b-local-b20-offset120-compact-5b24ca037ef6` | 20 scored, 0 errors, 40 candidates, pass@1/pass@2 0 | `artifacts/p3b3-qlora-1c64771f4188`: 0 verified, skipped | Not run |
| 140–159 | `artifacts/p3b-local-b20-offset140-repair2-df15fbc8fc54` | 20 scored, 0 errors, 40 candidates, pass@1/pass@2 0 | `artifacts/p3b3-qlora-aafa14ec7db6`: 0 verified, skipped | `artifacts/p3b3-compare-a7fc0912cbfd`: skipped |
| 160–179 | `artifacts/p3b-local-b20-offset160-rowgrid-f355c9990362` | 20 scored, 0 errors, 40 candidates, pass@1/pass@2 0 | `artifacts/p3b3-qlora-9261ce9381fb`: 0 verified, skipped | `artifacts/p3b3-compare-4572c9177567`: skipped |
| 180–199 | `artifacts/p3b-local-b20-offset180-analysis-292249a135e7` | 20 scored, 0 errors, 40 candidates, pass@1/pass@2 0; 160 calls, 1,195,458 reserved tokens, 928.91 s | `artifacts/p3b3-qlora-07d2c9cd4d6d`: 0 verified, skipped | `artifacts/p3b3-compare-b0cf4cc2153c`: skipped |
| 200–219 | `artifacts/p3b-local-b20-offset200-detail-analysis-b50d83e8a209` | 19 scored, 1 request error, 40 candidates, pass@1/pass@2 0; 160 calls, 1,509,603 reserved tokens, 1,668.28 s | `artifacts/p3b3-qlora-180d641d641b`: 0 verified, skipped | `artifacts/p3b3-compare-ab063c2a432b`: skipped |

The offset-180 prompt added a concise rule-analysis call and represented grids
as text rows. All 40 candidates still exhausted the training-example repair
loop; independent re-verification found zero eligible solutions. The trainer
left the existing adapter untouched, and the comparison gate retained
`promotion=keep_previous`.

The offset-200 run expanded the analysis prompt to require explicit comparison
of dimensions, changed cells, colors, shapes, and positions across every
training pair. It scored 19 tasks; one candidate request had a prompt-limit
HTTP 400. All 39 returned candidate outputs failed the exact training-pair
check, and independent re-verification found zero eligible solutions. Its
trainer and comparison gates both skipped and preserved the existing adapter.

An attempted offset-200 window with Qwen's default thinking mode is retained at
`artifacts/p3b-local-b20-offset200-thinking-0b3ec8e79cb7` as an interrupted
diagnostic, not a scored ARC run. All eight recorded requests were analysis
calls; seven exhausted the 2,048-token completion cap with
`malformed_thinking`, and the final request was interrupted. The run has no
completed task results and was not sent to the trainer.

A second diagnostic, `artifacts/p3b-local-b20-offset220-thinking4096-37146aeaff87`,
raised both client and server completion caps to 4,096. Its first two analysis
requests each consumed all 4,096 completion tokens and still returned
`malformed_thinking`; the incomplete run was stopped before scoring and was
not sent to the trainer.

Those eleven completed Qwen3-4B windows over Exploration tasks 0-219 produced
no independently verified solutions. To see whether a larger model from the
same family could supply eligible data, two further B-only windows used the
pinned Qwen3-8B 4-bit base revision
`545dc4251c05440727734bcd94334791f6ab0192`, MLX-LM 0.30.7, the guarded
server, a 32,768-token prompt cap, thinking disabled, and the same 2M server
token allowance:

| Exploration tasks | B run | Result | Training gate | Comparison gate |
| --- | --- | --- | --- | --- |
| 240-259 | `artifacts/p3b-local-qwen8b-b20-7bfbd96a3e45` | 20 scored, 0 errors, 40 candidates, pass@1/pass@2 0; 160 calls, 1,447,936 client-reserved tokens, 2,402.45 s | `artifacts/p3b3-qlora-888124bc2ae8`: 0 verified records, skipped | `artifacts/p3b3-compare-901212fbf0b4`: skipped |
| 260-279 | `artifacts/p3b-local-qwen8b-b20-offset260-07182ba11e74` | 20 scored, 0 errors, 40 candidates, pass@1/pass@2 0.05; 157 calls, 1,483,380 client-reserved tokens, 1,054,587 server-reserved tokens, 2,306.14 s | Default five-record gate `artifacts/p3b3-qlora-832395342aae` skipped with 1 verified record; separate min-1 pilot trained from that record at `artifacts/p3b3-qlora-59d6e2f34fa7` | Initial gate `artifacts/p3b3-compare-13184f25b89e` skipped; matched B/C pilot `artifacts/p3b3-compare-9a26aaf6340c` completed and rolled back |

The one verified Qwen3-8B result is task `a416b8f3`; its submitted transform
repeated each row and passed independent re-execution against the held-out
output. The trainer rebuilt exactly one SFT record and confirmed test outputs
were absent from its message. The unchanged default five-record profile
correctly skipped training. Because the sample lower was identified as a pilot
value, a separate `configs/p3b.local.qlora.qwen8b.lowdata.yml` profile lowered
the floor to one without changing the model or QLoRA settings. It trained a
real ARC adapter for 20 steps from that single verified record: loss 5.319 to
0.013, 21.63 s, 4,955,799,552-byte peak child RSS. Qwen3-8B base revision
`545dc4251c05440727734bcd94334791f6ab0192` had identical safetensors SHA-256
before and after training (`f2d29621aab300336ad645567ff38c42aac755513006ef4e8a579cf7ef5256d8`);
the saved adapter SHA-256 is
`cb2b646ae6aa80b51792e8f1bec3e0fe4a25d3f02c34c695961f752475dba229`.

The matched comparison ran on the unused tasks `57edb29d`, `b0722778`,
`5582e5ca`, `7953d61e`, and `44d8ac46`. B and C used the same Qwen3-8B
revision, temperature, token limits, prompt cap, and thinking setting; their
normalized shared config hash is
`68a9a5a2c5c42b0f91ee289c0100dbc710b444c5e469fb2f6bcd961e7859d26d`.
The C guarded server bound the trained adapter SHA above. B scored all five
tasks with zero errors (pass@1 0, pass@2 0.2). C scored one task and had four
task errors; six C requests returned `malformed_thinking`. The comparison
therefore marked the result unverifiable and rolled back, leaving
`active_adapter_sha256=none`. This is an executed matched comparison and a real
Gen 0-to-Gen 1 training run, but it does not show an accuracy gain; the Gen 1
adapter remains sealed as an artifact rather than active.

Across 260 completed B-only tasks (220 with Qwen3-4B and 40 with Qwen3-8B),
only one solution passed independent held-out verification. The default five
sample floor remains in place for normal training; the separate one-sample
profile records this explicitly low-data pilot. Zero-data windows were safely
skipped, and the malformed/incomplete C evaluation exercised rollback. The
training error path was also exercised with an intentionally unavailable MLX
Python runtime in `artifacts/p3b3-qlora-failure-safety-check`; it recorded
`status=failed`, `trained=false`, `weights_updated=false`, and
`previous_adapter_preserved=true`. The saved Gen 1 adapter SHA remained
`cb2b646ae6aa80b51792e8f1bec3e0fe4a25d3f02c34c695961f752475dba229` after the
failure. The public Evaluation and protected Promotion Gate/Monitoring labels
are not available to collection or training; synthetic smoke results are not
ARC results.

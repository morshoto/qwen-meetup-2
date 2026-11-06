# LT Phase A: exploratory ARC comparison

Status as of 2026-11-07: **the frozen LT comparison is running; Qwen3-14B has completed 180 of 200 tasks, and Qwen3-32B has not started**. Scored checkpoints at 100, 120, 150, 160, and 180 tasks are in `artifacts/lt-phase-a-frozen-20261107-01/score-checkpoint-14b-{100,120,150,160,180}.json`. This remains separate from the original strict RSI experiment in issue #2. All checkpoints are partial and do not represent a completed 200-task comparison.

## Live Phase A checkpoint (2026-11-07)

- The frozen comparison manifest is `artifacts/lt-phase-a-frozen-20261107-01/manifest.json`, registered as `ac03cb7a0f9d2d56c8ed03586d474e0db5cc319acc7b6668f20221e7b6672a48`. It fixes 200 task IDs, two candidates per task, the direct-grid prompt and parser, thinking off, a 4,096-token candidate cap, 1,800-second task deadline, model revisions and file hashes, source hashes, scorer, MLX settings, and 10-task batches. Seed support remains unverified.
- The task allocation excludes the three development IDs, the frozen 100-task Phase B pool, and historical reserved IDs recorded in the allocation audit. The official 120 public evaluation tasks remain reserved for Phase C.
- At 100 tasks, task pass@1/pass@2 were **1/100**, with 169/200 valid candidates and 639,336 tokens. At 120 tasks they were **1/120**, with 200/240 valid candidates and 785,053 tokens. At 150 tasks they were **3/150**, with 249/300 valid candidates and 956,787 tokens. At 160 tasks they were **3/160**, with 264/320 valid candidates and 1,010,353 tokens. At 180 tasks they were **3/180**, with 300/360 valid candidates, 60 invalid candidates, zero timeouts, 1,118,865 tokens, 11,615.3 seconds summed runtime, 11.07 GB process-reported peak memory, and swap unavailable. Per-output pass counts use a different denominator; see checkpoint JSON files.
- All 180 completed task records report their owned worker group absent and worker leader reaped. This is local process lifecycle evidence; it does not certify macOS system services or complete system isolation. The production path uses the direct-grid solver and does not execute generated Python or start Docker containers.
- The 14B run continues in 10-task batches. Continue from the existing run directory and journal; do not change the frozen settings based on scores. Start 32B only after all 200 14B tasks are terminal and its workers are absent. Then run the same frozen 200 tasks sequentially.
- Original development, RSI, and comparison artifacts are preserved. Do not use the official public evaluation answers or begin Phase B training under this protocol.

## Local inventory (2026-11-06)

- The issue #2 worktree was clean at `03653fadf47739fa9ffa9c02a044539a875c58b6`;
  its changes have been merged as PR #7. Existing seals and artifacts were not
  modified. Its source data files remain local and are not copied into this branch.
- The local training challenge SHA-256 is
  `779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5`;
  the separate solution file SHA-256 is
  `9f07a38bd25af5e83aa5bf85c5cb1a1fefdb30f6a755256fa65429e697ca97f9`.
  These hashes identify local files; they do not independently authenticate origin.
- The earlier E9 record identifies 14B revision
  `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4` and 32B revision
  `bcaaf7f538adf166c1080a2befdb4f6019f66639`. Reverify the installed model
  files, runtime and template before any new run.
- Docker reports image `arc-sandbox:local` as
  `sha256:a3013c6339a68a667aa9a5f26377693a1f4357a7f7a12db6470a8f06732b91f1`.
  Image availability alone does not verify containment or cleanup.
- `research/rsi/bounded-exposure-proposal.json` and related reviews inventory
  prior project use, but explicitly leave historical and pretraining exposure
  uncertain. A separate 200-task proposal and 100-task training pool are now
  recorded under `configs/`; their run settings remain unfrozen.
  The copied challenge file contains 1,000 task IDs; the union of the bounded
  proposal and extended metadata review identifies 317 of them as exposed,
  leaving 683 outside that known union. This is a bounded inventory, not proof
  that the remaining tasks are unexposed. No Phase B training pool is frozen.
- The [development-only proposal](../configs/lt.phase-a.development.json)
  deterministically selects three known-exposed training IDs using seed
  `20261111` and excludes the original #2 provisional 200-task development
  partition. It binds the local challenge and exposure-registry hashes; no
  challenge contents or answers are stored in the proposal. These IDs must
  remain excluded from any Phase B training pool and the new 200-task
  comparison. The proposal does not authorize inference or establish
  independent source provenance.

## Local model-file recheck (2026-11-06 20:09 JST)

The locally cached 14B and 32B snapshot directories had exactly the 12 and
14 file names, respectively, listed in `research/rsi/audits/technical-model-14b.json`
and `technical-model-32b.json`. SHA-256 checks of every listed file, following
its local snapshot path, matched those historical inventories, including all
six weight shards. This is a point-in-time local consistency check, not
independent model provenance, a loaded-model identity, a complete runtime
closure, or verification of the live execution boundary. No model was loaded.

## Remaining work before a comparison

1. Complete the development-only real-model path and verify model, template,
   budget, sandbox and process cleanup on the actual host. The existing
   [execution-boundary review](../research/rsi/execution-boundary-migration-20261106.md)
   records unresolved cleanup and evaluator-isolation findings. Fixture tests do
   not resolve them. Stop and record an incomplete run if controls cannot be
   established.
2. Audit exposure and finalize development, Phase B training and 200 comparison
   IDs without overlap. Freeze task contents, source files, prompt, runtime,
   model revisions, seeds, candidate count and all resource budgets. The
   current helpers bind proposed selection, configuration and supplied input,
   model and runtime inventories. They do not authenticate provenance, prove
   those inventories are complete, or establish execution readiness.
3. Build a bounded live runner, model-specific manifests, and durable prediction
   and resource records. The `run_task` and `run_bounded_task` helpers accept
   injected components and are fixture seams, not safe live launchers. Tests
   check held-out answers do not affect agent prompts or repair feedback; this
   does not prove operating-system isolation.
4. Run models sequentially with equivalent settings only after those controls
   are established. Report actual attempts, failures, denominators and costs;
   do not represent an incomplete run as a completed comparison. Do not use
   the official 120-task evaluation set here.
5. Freeze Phase B's training design before inspecting the 200-task results if
   those tasks will be reused for its comparison; otherwise disclose reuse as
   exploratory. No training or Phase C evaluation is included here.

## Offline preparation and scoring

`python -m src.arc.lt_prepare_cli --proposal PROPOSAL.json --config CONFIG.json
--challenges CHALLENGES.json --solutions SOLUTIONS.json --sources SOURCES.json
--prompt PROMPT --registry REGISTRY.json --split SPLIT.json --development DEV.json
--training TRAINING.json --source-root ROOT --source-inventory INVENTORY.json
--part frozen` writes the comparison document to stdout; repeat
with `--part receipt` to obtain the input receipt. Both invocations validate
the same inputs. Preserve their outputs before scoring or any experiment; these
documents alone neither verify the loaded model nor permit live execution.
By default, preparation and scoring also require `--registry`, `--split`,
`--development`, `--training`, `--source-root` and `--source-inventory`.
They recheck the protected allocation and complete source inventory and bind
that audit into the receipt. The supplied training inventory remains
independently unverified. `--fixture-only` permits unchecked test inputs and
labels the receipt `allocation_unchecked`; it does not permit a live run.
The configuration also binds model names and full revisions, a loopback guarded
server URL, a digest-pinned sandbox image and its resource limits. Component
construction and guarded-server metadata checks only compare supplied settings;
they do not authenticate a server, prove model identity or verify cleanup.

`python -m src.arc.lt_score_cli --frozen FROZEN.json --receipt INPUTS.json
--challenges CHALLENGES.json --solutions SOLUTIONS.json --sources SOURCES.json
--prompt PROMPT --registry REGISTRY.json --split SPLIT.json --development DEV.json
--training TRAINING.json --source-root ROOT --source-inventory INVENTORY.json
--results RESULTS.jsonl` writes a JSON report to stdout only
after rechecking the frozen selection and input receipt. Add `--format jsonl`
for per-task records followed by one summary record. `SOURCES.json` maps
stable inventory names to local file paths. Each result row must include the
frozen `comparison_sha256`, model key (`14b` or `32b`), task ID, status,
candidates, tokens, runtime and peak memory/swap values; use JSON null for
unavailable measurements. The report hashes the input results and explicitly
marks run identity as independently unverified. Preserve the original result
JSONL separately because the scoring report does not embed predictions.

Task pass@2 requires one complete candidate to match every test output. For
output-level pass@2, each output counts if either candidate matches that output;
those matches can come from different candidates. Denominators reflect only
available candidate slots, with missing tasks and candidates reported explicitly.
A reported incomplete result must not be interpreted as a 200-task score.


The `lt_public_cli` command accepts `--frozen`, `--receipt`, `--challenges`,
`--registry`, `--split`, `--development` and `--training`, and writes only
selected public tasks to stdout. It rechecks the bound allocation audit,
verifies task hashes and rejects test-output fields. `--fixture-only` accepts
only an allocation-unchecked receipt and does not authorize execution. This export reduces data passed to the solver;
it is not evidence of filesystem, network, or process isolation. The
`run_fixture_model` journal writes owner-only started markers and per-task
results and refuses to reuse a started model run. It accepts only the built-in
synthetic `FixtureBackend` and `FakeSandbox`; other components are rejected
before dispatch. Model-specific terminal
records bind the task inventory, configuration identities and result-file hash;
a consistency audit remains explicitly unverified. Token totals are recorded
only when every reserved request reports consistent fixture usage. Unknown token and
memory values remain null. When a catchable keyboard interrupt occurs during a
fixture task, the journal records it as aborted with unknown token use before
propagating the interrupt; partial candidates are not preserved. An uncatchable
process death or a write failure may still leave only a started marker. Its input
components and results remain unverified fixture evidence; no live launcher or
cleanup gate is provided by this helper.

## Synthetic Docker probe (2026-11-06 JST)

On Docker server 24.0.6 and the image ID above, the existing `DockerSandbox`
returned `[[1]]` for a synthetic identity function and raised `TimeoutError`
for a synthetic infinite loop with a three-second timeout. A `docker ps -a`
query for each exact, randomly generated container name found neither name
before dispatch nor after return; the observations took about 0.3 and 3.1
seconds respectively. The names were `arc-0c2081cbb61c4d868829af51c0065aec`
and `arc-68ecac46a1ad45068aea39fd9ed7c52d`.

This is a single locally observed normal/timeout probe, not independent proof of
container containment, absence of escaped processes, daemon correctness, or
model-server cleanup. It does not change the live-run readiness decision. No
model-generated code, model inference, or ARC task was used in this probe.

## Source inventory observation

At commit `d889256`, `python -m src.arc.lt_source --root . --part inventory`
recorded 276 files under the existing source/config inventory rules and a
canonical code SHA-256 of
`f0ccbe1a180c55d7544a8c407514a76e112f2c4c2beb5760bfee874f96fb571f`.
The same command with `--part paths` supplies the path map consumed by the
offline preparation and scoring tools. Recapture and compare the inventory
before any future use: this local snapshot does not prove the loaded runtime,
external dependencies, or execution readiness.

At commit `d9c4d21`, the same observation contains 276 files and code SHA-256
`2f88415a970b78f491f167a36893b6b7c9ecdb8d60301d1f883bebfbc24da07f`.
The prior digest above is historical, not a current freeze. The preparation
and scoring commands accept `--source-root ROOT --source-inventory INVENTORY`
together to recheck the whole recorded tree, including newly added files. A
receipt produced without those options is labeled `caller_supplied_files`.
Neither scope authenticates external packages or proves model-server identity.

The shared Docker wrapper remains unchanged to preserve the historical source
seals. Its interrupt cleanup behavior is not verified for this protocol.

## Synthetic end-to-end Docker integration

At commit `64029a0`, an opt-in test used a synthetic loopback HTTP server,
the registered `OpenAICompatibleBackend`, the existing `code_agent`, the
pinned real Docker sandbox and the offline scorer. Both model configurations
sent matching synthetic prompts and candidate seeds; the real sandbox ran the
synthetic identity function and both synthetic candidate grids scored exactly.
The same test suite recorded two synthetic infinite-loop candidates as timeouts
and observed the exact named containers absent afterward. Run it locally with:

```sh
LT_SANDBOX_IMAGE=sha256:a3013c6339a68a667aa9a5f26377693a1f4357a7f7a12db6470a8f06732b91f1 \
  python -m unittest discover \
  -s tests/arc -p 'test_lt_live_boundary.py' -v
```

Both tests passed on the local host. Their server replies, grids and source
code are synthetic; no Qwen weights or ARC task were used. The container
observations are not independent cleanup or containment proof and do not
establish model-server identity or readiness for live inference.

## Synthetic macOS and Metal boundary probe

On 2026-11-06, a default-deny macOS profile initially aborted even `/bin/echo`.
The host sandbox log reported `file-read-data /`; allowing a read of the root
directory as a literal, while keeping child paths subject to the allowlist,
let arm64 Python start. Adding `iokit-open` let a synthetic MLX operation run on
`Device(gpu, 0)` and produce `[3, 4, 5]`; the same operation also passed
outside the profile. This used cached MLX 0.32.4 and MLX-LM 0.30.7 without
loading model weights or ARC data.

Synthetic probes found that the profile denied a secret file, an outside hard
link, a symlink to the secret, a new loopback TCP connection, and `fork`. A
hard link **inside** an allowed directory remained readable. An explicitly
inherited file descriptor and socket also remained usable; with descriptors
closed, those two accesses failed. The profile currently allows broad Mach and
IOKit operations and depends on caller-supplied allowed directories. These
observations do not exclude aliases, IPC services or escaped processes, nor do
they establish the identity or cleanup of a model server. The profile builder
is marked non-authorizing and is not wired to the runner.

A separate synthetic test registered a uniquely named local Mach service in the
test process. A child running under the proposed profile, with inherited file
descriptors closed, looked up that service and sent a message that the parent
received. The test destroys its own receive port afterward. This demonstrates
a concrete IPC path through the broad Mach allowance; it does not establish
what access any particular external service grants. The profile remains
unsuitable for isolating a live model from evaluator data.

A narrower **test-only** profile denies Mach service lookup except for
`com.apple.MTLCompilerService`. With that rule, the same controlled service
lookup failed, while the synthetic one-layer Qwen3-shaped GPU forward pass
succeeded. Denying all Mach lookups prevented the compiler connection. These
observations cover one service lookup and one small workload on this host;
other Mach operations and IOKit remain permitted, the compiler is external,
and alias, service behavior, ownership and cleanup are not verified. The
restricted profile is not connected to a live launcher.

The synthetic controls, including the opt-in Metal operation, can be repeated
from the worktree using cached dependencies:

```sh
LT_PROBE_MLX=1 uv run --offline --no-project --python /opt/homebrew/bin/python3.12 \
  --with mlx-lm==0.30.7 python -m unittest discover \
  -s tests/arc -p 'test_lt_macos_probe.py' -v
```

The base profile denied a local listener. In additional synthetic probes, an
explicit rule for one loopback port allowed that port to bind and exchange a
byte with a local client; a different bind and an outbound connection were
denied in the tested cases. The listener extension exists only in the tests,
not in the profile builder or a model launcher. The port is not authenticated,
and these controls do not rule out other local or Mach-service access.

All six controls passed locally. This narrows the earlier allowlist startup
failure but does **not** clear the live-run blocker.

A follow-up synthetic check imported MLX-LM and evaluated a randomly
initialized, one-layer Qwen3-shaped model with a 16-element hidden size; it
produced finite logits of shape `[1, 2, 32]` on the GPU. It used no downloaded
model weights or ARC inputs. The initial profile denied Metal compiler cache
writes. Setting `HOME`, `TMPDIR`, `XDG_CACHE_HOME` and compiler-cache environment
variables did not redirect the compiler. The host sandbox log then showed a
`file-issue-extension` denial followed by `MTLCompilerService` cache denials.
A test-only rule granting an extension and read/write access to the user's
`org.python.python/com.apple.metalfe` cache allowed the synthetic forward pass.
The check used an allowed temporary working directory because importing
MLX-LM from the repository working directory was denied by the profile.

This shows that denying child `fork` does not establish that no external
service participates: Metal compilation delegates to `MTLCompilerService`.
The test touches a shared system cache, and its extension, contents, aliases,
service identity and service lifetime are not independently controlled or
verified. The test-only extension and listener rules are not in the profile
builder or a live launcher. The user-data isolation and complete cleanup
requirements remain open.

## Synthetic Metal service lifecycle observation

A read-only process-list observation around a synthetic GPU forward pass saw a
previously unlisted `MTLCompilerService` PID 14845 in its own process group
with parent PID 1; it was absent from immediate and half-second later
snapshots. A subsequent run of the repeatable probe at commit `749f71d` took
20 samples, saw previously unlisted PID 15273 with parent PID 1 and its own
process group, and did not see that PID in the final snapshot. Both synthetic
child processes completed. Existing system compiler services were already
present before each observation.

```sh
uv run --offline --no-project --python /opt/homebrew/bin/python3.12 \
  --with mlx-lm==0.30.7 python -m src.cli.lt_metal_lifecycle_probe
```

These are temporal process-list observations: PID reuse, sampling gaps,
concurrent workloads, service ownership and delegated work are not resolved.
They do not prove that a newly observed service belongs to the probe, that no
helper remains, or that a server is cleaned up. The probe never marks cleanup,
ownership or execution readiness as verified and does not terminate shared
system services. No real weights or ARC inputs were used.

### Responsibility metadata limitation

A read-only synthetic run queried the undocumented macOS
`responsibility_get_pid_responsible_for_pid` interface while sampling the
compiler-service process list. In one run, synthetic child PID 15986 and newly
listed compiler-service PID 16002 both reported responsible PID 12160; PID
16002 had parent PID 1, its own process group, and was absent in the final
snapshot. A separate control started two unrelated owned sleep processes; both
also reported responsible PID 12160. Thus a matching responsible PID is shared
by unrelated work in this app session and cannot identify a particular run or
prove that a service is owned by it. Other service queries returned permission
errors, and `launchctl procinfo` and `launchctl blame` required privileges not
available to this process. No privilege escalation was attempted.

The lifecycle probe now records this API's response or an unavailable state,
with `ownership_verified`, `cleanup_verified` and `execution_ready` always
false. The API is undocumented, may be unavailable or race with PID exit and
reuse, and is not an independent receipt. No model weights or ARC data were
used; no shared service was signalled.

## Development-only public export

`python -m src.arc.lt_development_export` accepts the proposed development
manifest, local training challenges and bounded exposure registry. It checks
their hashes and selection, and emits either a development-only public task
export or its receipt. It does not open the solutions file. The loader rejects
extra tasks, changed grids and test-output fields. Both outputs remain
non-authorizing and do not establish filesystem or process isolation.

An in-memory check against the local training source reloaded all three
proposed IDs. Canonical JSON SHA-256 values were
`6eb4dedaee93fc1ea0da260a22d12278f8a04d3305e0f1050437ee885258161e`
for the development manifest,
`6ca84fe65af901605d24f3915411f7648b91e29b77d6ca7541cc2873e7c1bb0b`
for the public export, and
`3031f53e0ccf5951d1f1ed3021f07b675fd1b49cbd584bb8167a2e49e7c53756`
for its receipt. No export was persisted or sent to a model. These hashes
identify a local proposal, not independently verified data provenance or a
completed development run.

`python -m src.arc.lt_development_score` scores recorded development-only
predictions separately from comparison results. It rechecks the development
receipt and local source identities, accepts only selected task IDs, reads the
solutions only in the offline scorer, and marks its report ineligible for a
comparison claim. A local wiring check used the three proposed tasks with an
empty prediction file: both model summaries reported three selected, zero
attempted and zero pass@1/pass@2 denominators. No predictions were fabricated,
no accuracy was calculated, and the temporary report was not persisted.

## Protecting the original RSI reservations

The comparison proposal now has an explicit `protected_exclusions` set in
addition to development, training and known-exposure exclusions. The separate
allocation check recomputes the prior proposal and checks that the original
200-task confirmatory and 50-task exploratory partitions, the development
proposal, caller-supplied Phase B training IDs and known exposures are excluded.
It does not authenticate the Phase B inventory or authorize a run.

Recomputing the current local registries identified 250 original protected
IDs. None of the three proposed development IDs overlaps them. Of the 1,000
training IDs, 433 remain after excluding the 317 known exposures, the 250
protected IDs and the three development IDs, before any additional Phase B
training exclusions. The 433 are not proven unexposed. This is an allocation
count, not a selected or frozen comparison manifest.

For a future comparison proposal, `python -m src.arc.lt_allocation` requires
`--challenges`, `--registry`, `--split`, `--development`, `--training` (a JSON
array of Phase B task IDs), `--seed` and `--part proposal` or `--part audit`.
It derives known exposures and the prior 200/50 reservations from the bound
registries, verifies the development selection, and rejects training IDs in
those reservations or the development subset. The audit marks the supplied
training inventory unverified: it cannot establish that the list includes all
Phase B examples. This command has only been exercised with synthetic tasks;
a 200-task proposal has been recorded, but comparison settings are not frozen.

### Process start-time observations

The macOS `proc_pidinfo` BSD record exposes process start seconds and
microseconds. In a synthetic control, an owned sleep process returned the same
start time on repeated reads and became unavailable after it exited. The Metal
probe now samples those fields for the child and newly listed compiler-service
PIDs, retaining missing observations and conflicting start times rather than
assuming a PID is stable.

At commit `5d7a936`, one synthetic run observed child PID 22197 with start
`1793962620.029253` and newly listed compiler-service PID 22208 with start
`1793962620.934878`. The latter had parent PID 1, its own process group and
the same unverified responsible PID as the child; it was not listed in the
final snapshot. This is an observation of those samples only. Start times do
not identify unobserved helpers, prove which run requested a shared service,
or establish that all delegated work is complete. The probe continues to
report `ownership_verified: false`, `cleanup_verified: false` and
`execution_ready: false`.

## Development response-format investigation

The original 14B direct-grid run produced four complete JSON arrays containing
`output` objects. The original agent expected arrays of grids and rejected the
objects. Both `a64e4611` responses ended at 2,048 tokens with incomplete JSON.
The LT-only normalizer accepts exact output-object wrappers and validates grids;
it rejects duplicate or extra keys, malformed JSON, ambiguous structures and
invalid grid values. Re-scoring the original responses under this normalizer
produced four valid candidates out of six, with pass@1 and pass@2 both 0/3.
The original raw records and score remain unchanged.

A separate thinking-off direct run with a 4,096-token limit and a format example
produced two valid candidates out of six and scored 0/3 for both metrics. Four
responses copied the literal two-output format example; they were rejected
because these tasks each have one test input. The run logged 33,294 tokens,
169.0 seconds across tasks, 11.00 GB process-reported peak memory and unknown
swap. Its local record is `artifacts/lt-dev-20261106-direct-v2-off-01/`.

The separate `lt-dev-20261106-code-off-01` code-agent run produced zero
scoreable candidates out of six, with pass@1 and pass@2 both 0/3. All six
candidates exhausted the configured zero-repair allowance. Twenty uniquely
named Docker containers were absent on their post-execution checks; the owned
model process exited. The run recorded 70,175 tokens, 414.8 seconds across
tasks, 11.00 GB process-reported peak memory and unknown swap. These local
observations do not establish the original strict RSI isolation gates.

With the same format-example prompt and a 4,096-token ceiling, thinking-on
produced three valid candidates out of six and scored 0/3 for pass@1 and
pass@2. Three candidates hit the token ceiling. The run recorded 55,070
tokens, 1,027.6 seconds across tasks, 11.37 GB process-reported peak memory,
and unknown swap. It exited normally; local records are in
`artifacts/lt-dev-20261106-direct-v2-on-01/`.

The clarified single-output example with an explicit instruction not to copy it
was evaluated in `lt-dev-20261107-direct-v3-off-01`. All six candidates were
valid grid arrays; task-level pass@1 and pass@2 were both 0/3. The run logged
37,402 tokens, 347.9 seconds across tasks, 11.07 GB process-reported peak memory
and unknown swap. Two pairs of independently requested candidates were
identical, and seed support was not independently verified. The owned process
exited and its observed PIDs were absent. This direct run did not execute code;
the separate code-agent run used checked Docker containers but produced no
valid candidates. A single code-agent end-to-end run with valid predictions
and verified container cleanup has therefore not yet been demonstrated. No
200-task allocation or comparison configuration has been frozen.

## One-repair code-agent development

The six zero-repair failures were incorrect training-example outputs: each
candidate ran and returned grids but failed three or four training pairs. The
recorded histories contain no syntax, API, runtime or formatting exception.
The separate `lt-dev-20261107-code-repair-14b-01` run used one structured,
example-based repair per candidate. It again produced zero valid predictions
out of six, with pass@1 and pass@2 both 0/3; all 40 named containers were
absent on post-execution checks and the model process exited. The run recorded
149,432 tokens, 882.9 seconds across tasks, 13.64 GB process-reported peak
memory and unknown swap. The clarified direct-grid agent is the comparison
candidate; the 32B development check must finish before its settings can be
frozen.

The allocation files record 100 Phase B training IDs and a 200-ID Phase A
proposal, selected from the 433 eligible training IDs after excluding the
registered development, known-exposure and original RSI reserved IDs. They
have no task-ID overlap. The Phase B training recipe and the comparison run
settings remain unfrozen; historical and pretraining exposure remain unknown.

The sequential 32B direct-grid check (`lt-dev-20261107-direct-v3-32b-01`)
used the same clarified prompt, thinking-off setting, temperature 0.6, two
candidate requests and 4,096-token ceiling as 14B. It produced six valid grids
and scored 0/3 for both pass metrics, with 37,402 tokens, 794.6 seconds across
tasks, 22.61 GB process-reported peak memory, and unknown swap. Both observed
model PIDs were absent after normal process exit. All three 32B candidate pairs
differed; two of the three 14B pairs were identical. Seeds were requested, but
seed support and statistical independence were not independently verified.

## Supervised batch development validation

The `lt-batch-dev-20261107-01` validation used the native 14B worker on two
registered development tasks, stopping after one and resuming for the second.
Both task processes exited and their owned process groups were absent; all four
candidate grids were valid. The independent scorer recorded 0/2 for both
pass metrics, 33,292 tokens, 313.3 seconds of task inference and 11.07 GB
process-reported peak memory. Swap was not sampled. The validation used only
public task exports and did not execute generated code. Local worker and result
records are under `artifacts/lt-batch-dev-20261107-01/`.

## Frozen Phase A comparison

The direct-grid comparison registration is `ac03cb7a0f9d2d56c8ed03586d474e0db5cc319acc7b6668f20221e7b6672a48`,
with the reviewable registration in `configs/lt.phase-a.frozen.json` and local
public export and journal in `artifacts/lt-phase-a-frozen-20261107-01/`.
It fixes 200 task IDs, both model file inventories, code and scorer hashes,
thinking off, two candidate requests, temperature 0.6, 4,096 generated tokens
per request, a 1,800-second task deadline and an MLX working-memory limit of
48 GiB. Seeds are requested but independence is unverified; swap is not sampled.

Each `run` invocation processes up to ten tasks. The 32B run cannot begin until
all 14B tasks have terminal records. A task whose worker was interrupted stops
the journal; `reconcile --model MODEL --task TASK` checks the recorded process
group before recovering an already completed worker output or preserving a
partial attempt for retry. No automatic retry occurs if process evidence is
missing or the group still exists. The offline `lt_benchmark_score` command
reports actual attempted and candidate denominators without fabricating results.
These checks cover the owned task process group, not macOS system services.

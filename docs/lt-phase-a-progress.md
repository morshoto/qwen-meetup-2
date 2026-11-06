# LT Phase A: exploratory ARC comparison

Status: **incomplete; no model calls or ARC evaluation performed**. This is a
separate protocol from the strict RSI experiment in issue #2. No frozen task
manifest, model-specific run manifests, real predictions, or comparison score
exists yet. The comparison, attestation and scoring utilities are offline scaffolding,
not an execution gate or a complete live runner.

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
  uncertain. No 200 IDs have been selected for this protocol.
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
memory values remain null. Its input components and results remain unverified fixture
evidence; no live launcher or cleanup gate is provided by this helper.

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
no real 200-task proposal has been created or frozen.

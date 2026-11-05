# Issue 2 execution status

The [research design](rsi-research-protocol.md) and
[v2 target amendment](../research/rsi/design-v2.md) remain prospective. This
branch is an offline implementation in progress, not an executed experiment.

## Process monitoring

`src.arc.rsi_process_monitor.watch_owned_process` watches an already-created,
unreaped child that leads its own session. It checks resource samples and sends
termination signals only to that owned process group, retaining the leader PID
until cleanup. Missing, stale, or unavailable samples stop the child. On 64-bit
macOS it supports Python builds lacking `os.waitid` through the Darwin libc
interface. `src.arc.rsi_process_sampling` supplies host and process-tree samples.

Local verification used short-lived Python fixture children, including one that
ignored `SIGTERM`; no model or training process was started. In a local fixture
using the real host sampler, the leader exited with code 0 and three samples
were observed. macOS returned a permission uncertainty during group cleanup, so
`group_signals_confirmed` was false. The monitor does not prove that descendants
which escape its session are gone, that samples capture instantaneous peaks, or
that a model trained successfully. Training integration and independent process,
resource, artifact and adapter verification remain unfinished.

An incremental loss parser and optional bounded health sampler now stop an
owned fixture process after a non-finite MLX-style training loss or unavailable
health evidence. A live Python fixture emitted a `nan` loss line and was
terminated. The training runner has not been connected to this parser; loss
reports arrive only when the trainer emits them and do not prevent an earlier
non-finite update. The sampler and stream are caller supplied, and process
containment and adapter verification remain open. No model or training process
was started.

`watch_training_stream` now connects the loss parser to the owned-process
monitor through a bounded binary stdout reader. Fixture processes exercised
non-finite and malformed loss, finite loss and an escaped descendant keeping
the stream open. The last case is reported as an incomplete stream; the
supervisor cannot prove that escaped descendants have stopped. The caller must
drain or redirect stderr, and the actual MLX trainer is not connected. Loss
observations and clean fixture exit do not establish a successful preflight.

A local preflight observation review now checks the monitor's elapsed time,
resource samples, exit, stream completion and one finite step. Fixtures cover
missing loss, bad exit, resource limits and incomplete streams. A consistent
observation remains unverified and does not clear the independent adapter
reload, containment, provenance, registration or training authorization gates.

### Adapter structural precheck

A read-only precheck validates a proposed rank-8, two-layer Qwen3-14B adapter
against the expected 28 F32 LoRA tensor names and dimensions, contiguous
safetensors data, finite values and a caller-supplied digest. Synthetic fixtures
cover malformed files and non-finite values; the installed safetensors reader
also opened the 28-tensor fixture. The expected layout is derived from the
local model configuration and MLX-LM source, and has not been validated against
an actual trained adapter. A structural match does not prove weights changed,
that the adapter loads into the model, or that its origin is authentic.

A separate factor comparison rejects adapters without a nonzero A/B pair in
the same module and detects numerical changes from a supplied parent, treating
positive and negative zero as equal. It is tested with synthetic files only.
Nonzero factors do not prove their matrix product is nonzero, and changed file
values do not establish an effective model update or improved predictions.

A bounded product probe now examines up to eight nonzero rows and eight nonzero
columns per LoRA module and can report a sampled nonzero A/B product or
parent-relative product difference. Synthetic fixtures cover cancellation and
factor rescaling. No witness means the sampled entries were inconclusive; it
does not establish equality over the full matrices. A witness describes local
adapter files and does not prove model reload, execution provenance or a
prediction difference.

## Current gates

- The slot lifecycle consumes a claimed slot before fixture execution. A
  missing completion remains uncertain and cannot be replayed automatically.
  The fixture journal checks recorded byte hashes but is not an authenticated
  run seal and cannot establish training eligibility.
- The task split remains provisional and its retained-history inventory cannot
  prove the absence of missing prompt exposure or model pretraining exposure.
- Draft registration, replay, transport consistency and SFT prechecks do not
  establish an independently authenticated run or authorize training.
- A guarded live collector, independently audited run evidence, training gate,
  monitor integration and matched evaluation are not yet complete.
- Live model calls, training, external compute and independent scoring require
  their separately scoped authorizations and frozen budgets. None were run here.

### Scheduled call reservations

Owner-only, exclusive markers now consume each generation/slot/attempt before
fixture processing. The schedule permits at most 30 generation requests per
generation; a marker survives interruption and cannot be reused. The fixture
journal exercises this reservation but performs no model calls. These markers
are not an authenticated ledger: they do not account for calibration calls,
reserved tokens, sandbox starts or wall time, and a writer with access to the
run directory can alter them. Coordinated completion and dispatch, a shared
stage budget, independent provenance audit, and live authorization remain
required before any live collector can rely on them.

### Shared stage budget

A local, lock-serialized budget now reserves generation and canary calls against
one per-generation ceiling of 32 calls and 200,000 tokens, and reserves sandbox
starts against the 300-start ceiling. Reservations survive process interruption;
wall and monotonic clocks enforce the 3,600-second window and fail closed on
observed rollback. Adapters connect the budget to the capturing backend and
sandbox interface. The caller must supply the remaining-time callback and use
these adapters for every dispatch. These controls depend on cooperating local
processes and mutable local files; they do not authenticate a ledger, provide
hard termination of an already running sandbox, or authorize a live experiment.
No full collector or independent run audit is claimed by these components.

### Conditioned slot orchestration

`collect_slot` connects the task schedule, local claim and attempt reservations,
shared budget, public repair rules, metered sandbox, private verification, and
novelty precheck. It stores owner-only completion evidence and consumes an
uncertain slot after interruption. Its transport and sandbox are supplied by
the caller, the input digests are caller-supplied, and its local output is not
an independently authenticated run record or an eligible SFT record. A concrete
transport binding, execution registration, independent audit, training and
matched evaluation remain unfinished. Tests use scripted fake components only.

### Captured transport binding

The slot transport now constructs a guarded, local-only capturing backend for
each scheduled attempt, with the pinned model settings, scheduled seed, and a
per-request timeout bounded by the remaining stage budget. A fixture HTTP run
exercised a repair and replayed its captured request and response bytes through
the lineage precheck. This is local consistency evidence only: neither the
fixture server nor the caller-supplied sandbox establishes live provenance,
registered model weights, adapter identity, or independent audit. No live model
call was performed.

### Capture-to-slot replay

Captured transport request IDs are now recorded in each slot attempt. The replay
precheck joins request, response and ledger records by ID, verifies the supplied
slot evidence digest, reconstructs prompts and seeds, replays the verifier, and
checks the novelty result. Reordered fixture logs pass; missing, duplicate, and
tampered records fail. The digest, records, registration inputs and sandbox are
still supplied by the caller, so this does not establish an independently sealed
run or training eligibility.

### Matched task evaluation primitives

The evaluation helper requests two independent source samples with the same
label-free solve prompt, checks visible demonstrations, caps each sample at ten
sandbox starts, and retains at most two valid outputs per test input. It scores
through the isolated solution verifier, including cases where different samples
solve different test inputs. Infrastructure errors stop evaluation. These are
fixture-tested primitives, not a matched model comparison: model and adapter
identity, registered seeds, shared comparison budget, frozen outcomes and run
provenance must still be implemented and audited.

### Frozen paired comparison

A local comparison record can freeze 200 task outcomes for the parent and
candidate, with distinct per-sample seeds matched across arms and adapter IDs.
The summary reads a digest-checked frozen record before the paired bootstrap;
missing task coverage is reported without a score. The record contains
caller-supplied outcomes and asserted settings, so the resulting interval is
descriptive and cannot establish model improvement without independently
authenticated execution and scoring. No model comparison has been run.

### Matched comparison orchestration

A comparison-stage budget now caps evaluation at 800 model calls, 16 million
reserved tokens, 8,000 sandbox starts and 36,000 seconds. Setup canaries require
a separate budget. A local runner checks the task and seed schedule, reserves
before each model and sandbox call, records task results, and freezes complete
paired outcomes. An interruption records an incomplete result and consumes the
run marker. A 200-task fake run exercised both arms and the full 800-call count;
the supplied generators, sandbox, verifier and registration are not
authenticated. This is not evidence of a real model comparison or authorization
to run one.

### Comparison outcome replay

Task-arm records now retain bounded program sources. The replay precheck
re-executes all 400 records with supplied tasks, sandbox and verifier, compares
predictions and scores with the frozen outcomes, and checks task-content hashes
bound into the frozen record. A 200-task fixture replay passed; altered task
content, missing records and changed scores were rejected. The replay inputs,
model emissions and local digest remain unauthenticated, so this is not an
independent provenance audit or evidence of a model improvement.

### Recursive transition screening

An offline screening helper replays up to two frozen comparisons, requires the
first parent to be the base and the second parent to be the first selected
adapter, and checks matching development task content. It applies the proposed
positive point-estimate rule and stops after a nonpositive screen. Fixture
comparisons cover positive, nonpositive, wrong-parent and altered-record cases.
The selected adapter is provisional: these caller-supplied comparisons and
sandbox results are not independently authenticated and cannot start another
generation or establish improvement. The stronger confidence-interval criterion
remains separate from this development screening rule.

The captured screening path additionally joins and checks every recorded
request, response and ledger entry for each comparison before screening. A
two-generation local HTTP fixture exercised both positive screens and rejected
a missing second-generation capture. The server declarations and logs remain
locally supplied; they do not authenticate which weights or adapter a server
actually loaded. No live comparison was performed.

When the selected adapter is from generation two, a descriptive helper can
pair its frozen development outcomes with the original base outcomes. It
requires matching task content, settings and per-task sample seeds and reports
the fixed paired bootstrap; a mismatched seed or missing capture is rejected.
A two-generation HTTP fixture exercised the path. These are reused development
outcomes, not a separately registered final evaluation, independently verified
improvement or ARC-AGI-2 score.

### Final diagnostic proposal

A local proposal can freeze the provisionally selected adapter, captured
comparison digests, diagnostic task hashes, harness and settings hashes, an
explicit budget, and the v2 42.4% semi-private target. It checks task-ID and
exact prompt-content separation against caller-supplied generation and
development tasks; the budget
is constrained by the proposed comparison ceilings. A two-generation fixture
exercised the proposal and rejected overlap and an excessive budget. This is
an unwitnessed proposal, not an immutable execution registration, proof of
complete exposure history or nonidentical related tasks, an experiment-wide
once-only guarantee, or authorization. No public diagnostic or independent
submission was run.

An exclusive local journal can consume a diagnostic attempt within one
owner-only directory, bind completion evidence to its proposal, and retain an
uncertain state after interruption. Fixtures check concurrent claims and a
changed proposal. The journal depends on cooperating callers and local files;
another directory or a writer able to alter the journal can bypass it. It is
not an authenticated reservation, independent evidence or permission to run.

### Quarantined training preparation

Training preparation now checks the structural five-record threshold, canonical
label-free task prompts, grid validity, parent adapter, batch and task hashes,
and fixed proposed QLoRA settings. It records a separate one-step synthetic
preflight plan without placing the generation records in that preflight. The
existing MLX trainer expects a different provenance and record contract, and
the monitor does not prove containment of escaped descendants. Independent
batch audit, a registered canary, trainer data-format audit, process containment,
execution registration and explicit training authorization remain open gates.
No training or preflight was started.

### MLX configuration preview

`preview_mlx_configuration` translates the proposed preflight and generation
settings into a reviewable MLX configuration, including prompt masking, two
adapted layers, rank 8 and scale 2.0. It requires separate canary and
generation dataset paths and records the second-generation parent adapter as
an unverified claim. A local check against MLX-LM 0.30.7's configuration keys
accepted the 21 previewed keys for both phases. This checks the key names, not
trainer execution, adapter compatibility, path contents or resource limits.
The preview is not an execution registration or training authorization.

### Synthetic preflight canary dataset

A separate, owner-only canary artifact now contains five fixed synthetic chat
rows and rejects exact grid overlap with a supplied visible task inventory.
On 2026-11-05, the local 14B tokenizer and MLX-LM 0.30.7 format check accepted
the rows against the 1,000 locally supplied training tasks. The JSONL SHA-256
was `c7ca0ab04a51cebc44d0b4cf95de24519c3595448efbabd1abef2700bb26d19b`.
The check did not cover a complete evaluation or historical inventory. The
canary is human-authored infrastructure data, separate from generated research
records; it is unregistered and ineligible for training. No model weights,
preflight or trainer were run.

Canary readback now recomputes the rows and mask and checks the owner-only
dataset, audit and directory contents. The installed MLX-LM 0.30.7 local
dataset loader also read the five synthetic rows with matching masks and empty
validation and test splits in a 2026-11-05 fixture using the same 1,000-task
inventory. This is local loader evidence, not canary registration, full
exposure clearance or training feasibility.

The canary loader audit can now load the pinned local tokenizer and MLX-LM
0.30.7 components directly, checking the tokenizer and configuration hashes
before and after. A native local fixture passed with the five synthetic rows
and the same 1,000 supplied training tasks; it used `python -B` and loaded no
model weights. This does not authenticate the running process or establish a
complete forbidden-task inventory.

A preflight proposal can now bind a checked canary dataset, training
preparation, MLX configuration, fixed resource limits and supplied model,
runtime, environment, source and tokenizer hashes. Fixtures reject changed
inputs, overlapping paths and loader failures. The hashes and loader are
caller-supplied; the resulting owner-only proposal is not a witnessed
registration, an independent dependency audit or authorization to train.
The proposal now also requires the local synthetic preparation and dataset
replay markers, checks the generation dataset and binds its stage and dataset
hashes. A fixture exercises the path from generation replay through the
proposal. These markers are locally writable and do not establish the five
independently audited strict same-model Oracles required by the protocol.

The preflight identity checker now compares proposal-bound manifest bytes with
the local model, interpreter, virtual environment, source inventory and
tokenizer files. Its first fixture check found the previously recorded virtual
environment inventory had changed from 6,249 to 6,251 files. A recently
written `safetensors` bytecode file was observed; earlier local Python fixture
imports may have created bytecode, but the aggregate inventory cannot identify
every change. The earlier observation remains preserved. A [fresh local
inventory](../research/rsi/local-environment-observation-refresh.json) and a
fixture proposal then passed the local identity checks. The proposal used
synthetic data and caller-supplied manifests; this is not independent process
attestation, complete dependency closure or execution authorization.

The preflight launch bundle freezes an MLX YAML file, the existing
`src.cli.p3b_train` command with thinking disabled, interpreter and wrapper
hashes, working directory and offline environment overrides. It checks the
local proposal identities first and writes owner-only review material without
launching it. A fixture proposal using local identity observations produced a
bundle; the fixture used synthetic data and did not execute the command. This
does not clear strict-oracle provenance, containment, registration, independent
reload or the separate authorization required for preflight.
Bundle creation now replays the complete proposal and checks the synthetic
generation and canary datasets before and after local identity verification.
Fixtures reject changed datasets and a modified learning rate even with a
recomputed proposal digest. A synthetic fixture also produced the bundle after
local model and runtime identity checks; it did not execute the command.

### RSI chat format and mask precheck

A format precheck converts quarantined rows to MLX-style chat JSONL and checks
that the thinking-off inference prefix matches the masked token prefix, with a
nonempty completion and no sequence exceeding 2,048 tokens. Tests use supplied
fixture tokenizer and renderer objects. The existing trainer and SFT audit still
require their own provenance schema; this precheck does not adapt or bypass
that contract. Actual tokenizer/runtime identity, loss masking in the pinned
MLX environment, independent batch audit and training authorization remain
unverified. No tokenizer model, preflight or training was run.

### Quarantined dataset artifact and readback

The format-checked chat rows can be written once to an owner-only quarantine
directory with a hash-bound audit record. A separate readback recomputes the
rows from the supplied preparation, tasks, tokenizer and renderer, and rejects
changed data, audit records or unexpected files. Fixture tests exercise these
checks. The result stays `training_allowed: false`: these inputs and their
origins are not independently authenticated, and the existing MLX trainer
expects a different provenance contract. The artifact is not a registered
training dataset and no preflight or training was started.

### Local MLX dataset loader fixture

On 2026-11-05, `python -m tests.arc.rsi_mlx_dataset_fixture MODEL_DIRECTORY`
used the local 14B tokenizer and MLX-LM 0.30.7 loader to read five fabricated
rows from a temporary quarantine directory. The loaded rows and their masked
token sequences matched the checked artifact; validation and test splits were
empty. The dataset SHA-256 was
`97029d0e0ba4510041314d94499d61e7bac33c67f8d8bf0c68bad228f1acf501`.
The audit uses local, caller-supplied components and does not authenticate
their provenance or demonstrate training. It loaded no model weights and did
not run a preflight or training.

### Local 14B tokenizer fixture observation

On 2026-11-05, the read-only fixture procedure
`python -m tests.arc.rsi_tokenizer_fixture MODEL_DIRECTORY` ran against the
local 14B revision `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4` using native
arm64 MLX-LM 0.30.7 with offline mode enabled. Its [recorded output](../research/rsi/tokenizer-format-fixture.json)
shows five synthetic rows with 148 masked prefix tokens and 29 completion tokens
per row. The procedure computes its expected tokenizer hashes from the local
files, so it demonstrates local format behavior and drift checking during the
call, not independently authenticated tokenizer provenance. The rows are
fabricated test inputs; no inference, preflight or training occurred. The
existing MLX trainer's source-specific provenance contract remains unresolved.

### Captured comparison transport

The comparison runner can now retain unique request IDs alongside each sample.
A local guarded HTTP adapter routes parent and candidate calls to separate
capture logs with matched model settings and requested seeds. Fixture HTTP
responses passed the existing request/response byte precheck for both arms;
invalid routes and duplicate IDs stop execution. This does not verify the
server's loaded weights or adapter, bind all comparison captures into the
replay audit, or establish independent provenance. No live requests were made.

### Comparison capture replay

The comparison replay now joins all 800 sample request IDs to per-arm captured
request, response and ledger records, checks byte hashes, requested settings,
seeds and sample source, and replays the task outcomes. A complete fixture HTTP
comparison passed with reordered capture records; missing, duplicate and altered
records failed. The captured records and sandbox are still locally writable and
caller supplied, and server weight, adapter and runtime identity are not
independently established. No live comparison was performed.

### Guarded comparison server declarations

A wrapper now checks both loopback server declarations before each comparison
request using the existing guarded contract parser. It requires the expected
model, adapter claim, thinking canary, reviewed driver, fresh distinct ledgers,
and a combined server token ceiling no greater than the comparison budget. It
rejects changed or regressing ledger declarations. Local fixture servers tested
the actual parser and rejection of a changed adapter claim. These declarations
are self-reported and are not evidence of the server process, loaded weights or
adapter; the wrapper does not grant live execution authorization.

### Local 14B model file observation

On 2026-11-05, a read-only hash of the local 14B snapshot produced the
[model file observation](../research/rsi/local-model-observation.json). It
contains 12 file hashes, including both indexed weight shards, the tokenizer,
configuration and chat template. The index and proposed 4-bit, group-64
configuration were consistent at observation time. The checker also compares
adapter files with the existing driver's adapter digest; only fixture adapter
files were tested. This observation is not an execution registration, an
independent witness, proof of artifact authenticity, or proof of what any
server loaded. It did not run inference or training.

### Draft registration and local model check

A read-only fixture check on 2026-11-05 compared the local 14B snapshot with the
model file observation through `verify_registered_model`. It used the local
training challenges and solutions, the provisional split, synthetic verifier
probes, an empty reference pool, and synthetic runtime and sandbox manifests.
The model-manifest input hash was
`f77bd29a69a7d047ce17a3940178f40350af50c24efed4763bcc6ce8b4d319cf`;
the check returned `draft_model_files_consistent` and `execution_ready: false`.
This exercise does not validate the synthetic inputs, establish split
independence, authenticate the runtime or server, or authorize a model call.

### Captured slot to quarantined draft

A replayed, provisionally novel captured slot can now produce an SFT draft bound
to the final request ID, response digest, slot and slot-evidence digest. The
draft records its origin as `unverified_capture`, contains no private query
outputs, and remains ineligible for training. The current batch precheck rejects
that origin; no local replay or caller-supplied record is promoted to verified
live-model provenance. The integration was exercised with fixture HTTP and a
fake sandbox, not with Qwen.

### Invalid evaluation completions

The comparison transport now records truncated or malformed model completions as
failed samples while allowing the next independent sample to run. Transport,
identity, budget and infrastructure failures still stop the comparison. Capture
replay checks that the raw response and ledger justify the invalid outcome,
including the finish reason. Fixture HTTP exercised both truncated and
malformed responses; no live model call was made.

### Invalid collection completions

Truncated and malformed collection completions now consume the slot as
`invalid_model_output`; they are not accepted as oracle code or used as repair
feedback. Capture replay checks that the raw response and finish reason support
the terminal outcome, and draft conversion rejects it. Fixture HTTP exercised
these cases; infrastructure failures still leave the slot uncertain. No live
model calls were made.

### Collection generation and replay

A generation-stage runner now collects ten scheduled slots under one budget,
adds provisionally novel candidates to the reference pool before later slots,
and records completion or interruption. It binds the supplied task, probe,
parent and initial pool identities. A fixture HTTP run and replay checked all
ten slot journals, captures, pool evolution and counts; missing, duplicate or
altered evidence was rejected. The stage and replay use caller-supplied model,
sandbox and registration inputs, so their results remain unverified and cannot
authorize training. No Qwen call was made.

### Generation draft batch

A replayed generation can now derive quarantined SFT drafts in slot order using
the same evolving novelty pool. Fixture runs exercised both one provisional
draft and nine distinct provisional drafts; the count threshold is reported
separately from training eligibility. Every draft remains marked
`unverified_capture`, and the existing batch precheck rejects it for training.
The observed count is not the required independently audited strict-new count.
No Qwen calls or training occurred.

### Generated example retention and SFT design gap

Generation draft replay now retains each candidate's generated query
input/output pairs in the owner-only stage artifact, bound to the parent task,
program source, request ID and slot evidence. The pairs are generated labels,
not ground truth; the record and its origin remain unverified. They are kept
outside the chat messages and contain no private verifier probes.

The current SFT preparation still uses the original visible demonstrations as
the prompt and the generated program as the completion. It does not consume
the retained generated query outputs. Thus the current quarantined MLX dataset
does not implement training on those generated examples. A concrete synthetic
task-to-SFT mapping must be registered and audited before it can be evaluated
or used for training; the existing fixture format check does not resolve this
gap. No new model call or training occurred.

One proposed mapping is now exercised in generation replay: move the first
visible demonstration input to the synthetic task's query, retain other visible
demonstrations, and add generated query pairs as demonstrations. The completion
is the generating program. Repeated inputs are recorded as mapping rejections
to avoid revealing the held-out query in a demonstration. The resulting rows
are in the local stage artifact with `training_allowed: false`; they are not
registered as the experiment's method. The example and source bindings are locally supplied consistency checks,
not proof that a particular model generated the labels or that they are correct.

A separate synthetic preparation now recomputes those rows from a digest-checked
stage artifact, rejects conflicting identities and requires five mapped rows.
It freezes the original and synthetic task hashes and can be passed through the
existing format and mask precheck with the returned synthetic tasks. A fixture
generation exercised nine rows and rejected an altered row and duplicate
identity. The stage digest, tasks and generated labels are caller-supplied;
this preparation remains quarantined and cannot authorize training. Independent
audit, mapping registration, process containment and the other gates remain open.

The generation draft artifact can also be replayed from slot captures and the
supplied sandbox before and after synthetic preparation. Replay recomputes the
generated outputs and mapped prompts; a fixture with an altered generated
label, updated prompt and recomputed digest was rejected. A failed second replay
can leave an owner-only preparation file behind without the exclusive replay
completion marker. A successful replay writes a marker binding the stage and
preparation hashes; the marker remains local and does not authorize training.
Replay depends on locally
supplied captures, task data and sandbox behavior, and is not independent
attestation of model, execution or label provenance.

The replayed synthetic preparation can now be materialized into the existing
quarantined MLX JSONL format. The wrapper checks its preparation marker and
hashes, replays the generation before and after writing, reads the dataset
back, and writes a separate local completion marker. A fixture exercised nine
synthetic rows and an interrupted post-write replay; interruption left no
dataset completion marker. All inputs, including the sandbox and tokenizer,
remain caller supplied. The dataset is still ineligible for training.

### Local runtime identity observation

A read-only probe on 2026-11-05 recorded the local arm64 Python interpreter,
selected package versions, MLX-LM 0.30.7 module hash and reviewed
`generate_step` source hash in the [runtime observation](../research/rsi/local-runtime-observation.json).
A fixture draft registration using that observation, the provisional split and
synthetic probe and sandbox inputs returned `draft_runtime_consistent` with
runtime-manifest hash
`70f75b584986837b1e21afac6cf9e696fc601f68440ffe1d9f0be7bf1f03ed87`
and `execution_ready: false`. This samples local runtime identity; it does not
hash every transitive dependency, attest a server process, establish dataset
independence, or authorize inference or training.

### Local virtual-environment inventory

A read-only inventory on 2026-11-05 hashed 6,249 files and 959 directories in
the local MLX virtual environment, including unrecorded files and bytecode.
The [observation](../research/rsi/local-environment-observation.json) binds the
inventory to the interpreter hash. A fixture draft combining it with the runtime
identity returned `draft_runtime_consistent` and `draft_environment_consistent`;
the draft used a provisional split and synthetic verifier and sandbox inputs,
and remained `execution_ready: false`. The inventory excludes the base Python
standard library, operating-system libraries and server process identity. It
is a local observation, not a complete execution dependency freeze.

### Generation budget accounting

Generation reports now include before-and-after local budget counters and their
call, reserved-token and sandbox-start deltas, including interrupted stages.
For a completed captured stage, replay recomputes each token reservation from
the captured prompt and checks the call and token totals against the recorded
deltas. Sandbox counts remain local journal values and are not independently
verified process or container evidence. Concurrent unrelated reservations
cause reconciliation to fail rather than being attributed to the generation.

Captured generation replay can also summarize server-reported prompt,
completion and total tokens across all recorded attempts, including repairs.
It joins usage to raw responses and keeps partial or missing coverage explicit.
Fixture runs covered complete and absent usage and rejected an altered ledger.
These counts are not independently metered, monetary cost remains unknown, and
the audit's sandbox replay is separate from the recorded collection budget.
The report also retains budget counters before collection and after the stage.
If setup calls were reserved before the slot loop, their reported usage is not
in the slot capture and the full-stage reported token total remains unknown.
The fixture reserved a setup call without issuing it; it did not run a canary.

### Comparison budget accounting

Completed comparison records now bind before-and-after shared budget counters;
the descriptive summary includes their deltas. Capture replay recomputes all
800 token reservations from the request messages and reconciles them with the
frozen call and token totals. Incomplete runs retain their local budget deltas.
Fixture replay rejected altered reservation data. Sandbox counts remain local
journal values, and neither the budget nor captured records independently
prove process identity or actual compute cost.

### Comparison reported tokens

A captured comparison can now report server-supplied prompt, completion and
total token counts separately for each arm, after checking each ledger usage
against the captured response and its reservation. Missing or partial reports
include coverage counts and do not produce a complete token total. Fixtures
cover complete reports, absent reports and altered ledger values. These counts
are server-reported rather than independently metered, and monetary cost
remains unknown; reserved token ceilings are reported separately.

### Guarded collection server declaration

A collection transport wrapper checks the declared model, generation parent
adapter, thinking canary, ledger identity and token ceiling before dispatch. It
also requires one exact local call and token reservation for each invocation and
rejects ledger drift or a server reservation greater than the local budget.
Fixture tests exercised both generation seed ranges and the existing contract
parser. The declaration remains self-reported, and the counters depend on
cooperating local components; this is not server attestation or authorization.
No live model call was made.

### Comparison reservation binding

The guarded comparison transport now shares the comparison budget and requires
one exact call and token reservation before checking either server or sending a
request. It rejects a reused or mismatched reservation and server-declared
reserved tokens above the local total. A fixture runner exercised the binding
and preserved the final reservation after interruption. Server declarations and
local counters still require independent provenance; this change does not
authorize live evaluation.

### Duplicate-only historical references

A reference-pool builder now recomputes source, AST and sampled-behavior
fingerprints on the selected private probe set through a bounded supplied
sandbox. It checks supplied source hashes, requires nonempty ancestry confined
to the generation partition, rejects development overlap, and stops on a
reference execution failure. Fixture tests connect its output to generation
screening. The source origins and ancestry are caller claims, retained history
is not proven complete, and the builder does not admit historical records into
training. No historical source was executed during this observation.

### Exact duplicate feedback

Collection now allows a bounded repair after an exact source or AST duplicate,
using only the candidate's own source and a generic duplicate category. A
sampled behavior match remains terminal, so private-probe-derived feedback is
not sent to the model. Replay recomputes duplicate decisions against the same
reference pool for each attempt. Generation reports now count intermediate
attempt outcomes as well as final slot outcomes; uncertain interrupted attempts
remain visible through budget usage. These checks used fixtures only.

## Experiment outcome

The concise [results and gate report](rsi-experiment-results.md) links the
protocol to the current stopping point and identifies the evidence needed before
continuation. It does not report a live RSI result.

### Retained exact-prompt review

A separate [bounded review](../research/rsi/exact-prompt-exposure-review.json)
scanned JSON and JSONL string values in 6,762 retained artifact files for the
current `solve_prompt` text of each of the 200 provisional development tasks.
It found no exact matches. The inventory, source dataset, split and relevant
code hashes are recorded in the review. A match would identify retained text,
not prove that a model received it. A non-match does not rule out other prompt
formats, omitted files, other encodings or pretraining exposure; the split
remains provisional. The scanner's nested-string, malformed-file and changing
inventory behavior is covered by offline tests.

### Historical prompt hash review

The [hash review](../research/rsi/historical-prompt-hash-review.json) checked
reconstructions of direct and analysis prompts, including both system-prompt
modes and their serialized request messages, for 200 provisional development
tasks. It found no matching hashes in 6,762 retained JSON/JSONL files. As a
separate exploratory positive control, applying these reconstructions to all
1,000 source tasks matched 191 entries in 211 request-log files. That count is
not a count of distinct tasks or independently verified dispatches. Historical
format changes, missing records and pretraining remain unresolved, so these
checks do not establish split independence.

### Historical code task text review

The [task-text review](../research/rsi/historical-task-text-review.json) found
no exact legacy code-agent task blocks for the 200 provisional development tasks
in retained JSON/JSONL strings. An exploratory scan of all 1,000 source tasks
also found no matches, so there was no positive control showing that this text
format is present in the retained files. Many request logs store hashes; this
check cannot detect hash-only code prompts. Missing records, different formats
and pretraining exposure remain unresolved.

### Historical candidate reservations

A [candidate journal review](../research/rsi/candidate-journal-review.json)
parsed 700 reservations in 39 retained budget journals, including interrupted
candidates and fixture journals. They refer to 287 distinct source task IDs;
none overlap the provisional development split, and all 287 were already in
the extended metadata review. A reservation does not establish model dispatch.
Missing journals, other record formats and pretraining remain outside this audit,
so the split is still provisional.

### Interrupted requests in retained journals

Five retained `budget.json` files contain non-null pending request digests. The
current P3-A runner records the candidate scope before dispatch and hashes the
scope, request kind, candidate number and prompt into the pending digest. A
read-only reconstruction using each journal's candidate keys and the current
direct and analysis prompt formats matched one pending digest to an analysis
request for a non-development task. Four did not match these formats; code
requests include a generated hypothesis and may have additional repair history,
and historical code may differ. The current runner's ordering is not proof of
the implementation used by every old run. The candidate review found no
development IDs in any of these journals, but it cannot identify the four
pending requests or establish whether the server received any of them. Do not
resume or replay them automatically.

A further read-only check found the `cache` object empty in all five journals,
so there are no cached analysis responses there from which to reconstruct an
initial code request. Three associated `identity.json` files record an
`src/arc/agent.py` digest different from the current file; the other two lack
that implementation digest. Reconstructions using current code therefore do
not establish the historical prompt format. The one digest match above remains
only a consistency observation, and the other four requests remain unidentified.

### Version 3 bounded inventory proposal

`src.arc.rsi_bounded_inventory.propose_bounded_inventory` recomputes the legacy
registry, retained-history inventory, extended metadata review and candidate
journal review, unions their known dataset task IDs, and deterministically
constructs a development candidate. It hashes the task source, registry and
review code and repeats the reviews to detect local changes. The [local
proposal](../research/rsi/bounded-exposure-proposal.json) records 317 known
exposed source tasks and selects the same 200 development IDs as the earlier
provisional split. The separately recorded prompt reviews and location search
must also be independently reviewed and reconciled before any freeze. This
proposal is not independent review, a final manifest, execution authorization,
or evidence of no prior exposure.

### Prompt review reconciliation

A [local reconciliation](../research/rsi/prompt-reconciliation.json) replayed
all three saved prompt checks over the proposed 200 development tasks and the
same 6,762 retained JSON/JSONL files. The dataset, provisional split, recorded
review code and result inventories matched, with no saved or replayed matches.
The reconciliation binds the proposal and review file hashes; it is not an
independent audit, and missing history, unrecognized formats and pretraining
exposure remain unknown. The legacy task-text scan had no positive control in
the retained files, as noted above.

### Retained SQLite ledger review

A read-only [schema and inventory review](../research/rsi/ledger-schema-review.json)
covered 220 retained SQLite files containing 2,271 request rows. The request
tables have token-sequence hashes, token counts and request IDs but no task ID
column. No write-ahead log was present, and database hashes matched before and
after the review. This does not associate the hashes with source tasks, rule out
missing ledgers, or independently establish what a model received. The bounded
inventory's independent review must account for this limitation.

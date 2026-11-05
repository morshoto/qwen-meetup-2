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

### RSI chat format and mask precheck

A format precheck converts quarantined rows to MLX-style chat JSONL and checks
that the thinking-off inference prefix matches the masked token prefix, with a
nonempty completion and no sequence exceeding 2,048 tokens. Tests use supplied
fixture tokenizer and renderer objects. The existing trainer and SFT audit still
require their own provenance schema; this precheck does not adapt or bypass
that contract. Actual tokenizer/runtime identity, loss masking in the pinned
MLX environment, independent batch audit and training authorization remain
unverified. No tokenizer model, preflight or training was run.

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

# Phase B inventory and implementation plan

Status: **preparation only; not a frozen protocol or permission to execute a run.**
No Phase B inference, canary, adapter training, or evaluation has been performed.
Issue #5 describes a separate exploratory experiment and does not change strict
RSI issue #2 or its gates.

## Inputs and allocation

The worktree starts at merged Phase A `020f427aa3c6471126e36db744ebe733bad8bf9c`.
The Phase A frozen manifest and 100-ID proposed training pool are tracked.
The Phase A raw journal exists locally in worktree 4 under
`artifacts/lt-phase-a-frozen-20261107-01`; it is not assumed to be on GitHub.
The original checkout contains the ignored ARC training challenges and solutions.
The local competition archive contains 120 official evaluation **challenge IDs**;
its evaluation answers were not read. Comparison by exact ID found no overlaps
between the 100-ID pool and Phase A's 200 IDs, three LT development IDs, the
prior strict-RSI 200-task and 50-task evaluation allocations, the bounded known
exposure registry, or the official 120 IDs. This is an identity check, not proof
of semantic non-duplication or complete historical and pretraining exposure.

The prior strict-RSI design also schedules 20 generation task IDs. The Phase B
pool overlaps five: `009d5c81`, `00d62c1b`, `03560426`, `045e512c`, and
`0962bcdd`. The Phase A allocation routine did not check those slots. Do not
freeze or dispatch this pool while the allocation decision is unresolved. A
50-task distinct comparison pool must also exclude earlier reservations, Phase A,
Phase B, known exposure, and official evaluation IDs. The currently observed
133 candidates before excluding the scheduled slots are only a capacity check,
not a selection or proof of non-exposure.

E8/E9 report no 14B adapter training and do not satisfy the original five-Oracle
threshold. E8's Full provenance gap and E9's strict collection history remain
separate from this real-task experiment. No old seal or result was modified.

## Local runtime preflight

The pinned 14B snapshot directory exists, but its file hashes must be recomputed
and checked immediately before a frozen run. Native arm64 MLX-LM 0.30.7 is
available through the existing offline `uv` environment. `mlx_lm.lora --help`
shows `--mask-prompt`, `--num-layers`, `--iters`, `--seed`, `--max-seq-length`,
`--adapter-path`, and `--save-every`; this is CLI inspection, not a training
canary. The current trainer `ChatDataset` masks the user prompt using the chat
template, but the Qwen template's empty thinking block is included in its
assistant target whereas the inference backend sends it in the thinking-off
prompt. A local tokenizer-only probe confirmed this distinction. Before freezing
training, verify the exact tokenized prefix and target, sequence truncation,
trainer settings and model template; report any unavoidable masking difference.

## Remaining test-driven steps

1. Resolve the scheduled-slot conflict, independently recompute all exclusions,
   choose and register a disjoint comparison set, bind input hashes and source
   closure, and document the exposure limits before any inference or training.
2. Test public-only handoff, supervised worker lifecycle and crash recovery,
   provenance binding, offline verification, and task-lineage-disjoint JSONL
   creation including empty-yield behavior. Do not admit fixture data as live data.
3. Preregister the canary and actual training settings and limits; run a real
   one-step synthetic canary only with resource monitoring, base and adapter
   tensor evidence, reload in a new process, and owned-process cleanup.
4. Collect the bounded real-model samples and follow the issue's yield gate;
   train only if the canary and actual yield permit it. Lock matched evaluation
   before scoring and preserve negative, null, failed, or not-run outcomes.
5. Run relevant unit and fake end-to-end tests, document coverage and omissions,
   and deliver the result and an appropriately scoped PR. Official evaluation
   answers remain reserved for Phase C.

### Paired scoring test plan

Use the unchanged Phase A task scorer on recorded base and adapter rows, with
one explicit adapter state per row. Reject unknown states, duplicate rows and
unselected IDs. Report complete pairs separately for the first candidate and the
two recorded attempts, including improvement, regression, both-right and
both-wrong counts. Missing or failed rows are not ties. Preserve each state's
candidate validity, usage and latency accounting from the existing scorer and
report identical candidate pairs without inferring independent sampling. These
checks do not establish that actual inference settings or adapter provenance match.

### Verified dataset test plan

Build trainer chat messages only from recorded 14B candidate predictions after
full-task verification, never from the verifier's answers. Use the Phase A
public-task payload and direct-grid instruction as the user message. Reject
unknown or repeated task rows, comparison-task overlap, malformed provenance,
and non-14B or unclean worker results. Journal both candidates and rejection
reasons; select at most one correct candidate per task lineage. Zero yield skips
training, one or two tasks are labeled overfitting-only, and at least three use
a deterministic task-disjoint split with one or more validation tasks. Schema
and mask behavior still require a real installed-trainer preflight before use.

For paired uncertainty, report the observed improvement-minus-regression rate
and a conservative two-sided 95% Hoeffding interval over paired per-task changes
in [-1, 1]. The interval is conditional on independent task observations and
is not evidence that the fixed, selected ARC tasks are a random population
sample. Report no interval for an empty paired denominator.

### Adapter backend test plan

Keep Phase A's frozen backend unchanged and inherit its generation method for
both adapter states. Before loading, require an exact local model manifest and,
for adapter-on only, a caller-supplied digest and explicit rank-8, eight-layer
adapter configuration. Check saved LoRA names, shapes, dtype, finite values,
and equality to the loaded model tensors; recheck file identities after load.
Missing or mismatched inputs, extra or missing tensors, and adapter arguments
in the off state fail closed. Tests use synthetic loaders and tiny arrays only;
real-model reload, inference, process teardown, and runtime attestation remain
separate evidence that must be obtained later.

### Failed collection handling

Preserve a terminal worker timeout or error as a task-level journal record with
no invented candidate or correctness result. Require the same model identity and
owned-process cleanup evidence as successful tasks. A fully attempted collection
with any worker failures is `collection_failed`, not a zero-yield result; a
partially attempted collection remains incomplete. Neither status produces
training or validation examples. Do not retry an interrupted journal entry
without the existing recovery/reconciliation checks.

### Visible-content split check

Before dataset creation, compare both task IDs and canonical visible task
content across the full training and comparison manifests, including tasks that
produced no eligible candidate. Detect exact repeated examples even if train or
test pair order differs; reject repeated content within either manifest as well
as across them. This is not a semantic ancestry audit or proof that differently
encoded but related tasks are independent, so the exposure ledger must retain
that uncertainty.

A local, preliminary check of the 100 proposed training tasks and 200 Phase A
tasks found no exact visible-content duplicates within or across the groups
when train and test example order is ignored. It read only training-challenge
content, not test answers. SHA-256 inputs: challenge file
`779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5`,
training-pool file `06412cefcf3714e410a8b5a35d7dd90c3bfd4e31ee3543f3d460252222604157`,
and Phase A frozen config file
`469bae58cf5771e21aea3d22ecf37dc945c3c5a74792dda6a8cbe1d4cbb70820`.
This does not resolve the five scheduled-slot overlaps or prove semantic
independence. Re-run on the actual frozen manifests before use.

### Bounded canary stream test plan

Observe an already-owned child with the existing process resource monitor and
loss parser. Capture a bounded raw log while parsing it; stop on non-finite or
malformed loss, missing samples, timeout, or log overflow. A finite loss or a
zero exit is only an observation: adapter changes, base identity, independent
reload, and owned-process cleanup remain separate checks. Synthetic subprocess
tests must not be reported as a model canary or proof of escaped-child cleanup.

### Distinct comparison proposal test plan

Propose 50 IDs without reading outcome or answer files, using a fixed hash-based
ordering of the eligible training-challenge IDs. Exclude the whole original
100-ID Phase B pool, the 200 Phase A IDs, three development IDs, both strict
RSI evaluation partitions, all 20 scheduled strict generation slots, the
bounded known-exposure IDs, and official public-evaluation IDs. Any replacement
training IDs must also exclude the resulting proposal. Check the source and
input hashes separately; this selection is a proposal, not a frozen protocol or
proof of historical non-exposure.

### Monitored collection worker plan

Reuse the existing macOS host/process sampling, resource decision, owned-session
watcher and bounded stream capture for a Phase B worker. Check the baseline
before spawning; start a fresh owned session, record its PID, continuously
sample while it runs, and stop on timeout, pressure, missing samples or log
failure. Keep the process observation and samples; do not commit a worker result
unless termination and group cleanup are observed. Escaped descendants and
system-owned Metal services remain explicitly unproven. Tests use synthetic
children and controlled samples, not Qwen inference. The per-task 300-second
ceiling is a conservative preflight proposal and is not yet a frozen run cap.

### Monitored collection journal plan

Run one task at a time through the existing durable `BatchJournal` and
`supervised_batch`. Persist each worker's resource and cleanup observation before
committing its task result. Stop the batch after the first worker error, timeout,
preflight refusal, invalid evidence, or unverified group cleanup. An interrupted
or partially recorded task remains marked started and requires reconciliation;
never automatically rerun it. This orchestration is not a substitute for
verifying the frozen manifest, model/runtime and public-export identities.

A synthetic worker on this macOS host showed the existing pre-reap observer
returning the zombie leader with `group_cleanup_verified: false`; the attempted
group signals were also not confirmed. A separate signal-zero lookup immediately
after the leader was reaped observed that the original group ID was absent.
The new supervisor records that snapshot without changing the earlier observer's
fields. It is not independent proof against PID reuse, escaped descendants, or
system-owned services. A run with an extant or inaccessible post-reap group is
refused. Any actual canary or collection report must preserve both observations
and their limitations.

At source commit `b57f80e`, a one-second synthetic Python child was run on
2026-11-07 with the actual local macOS resource sampler. The supervisor reported
`process_exited`, five samples, a reaped leader and an absent process group in
the post-reap signal-zero lookup. `escaped_processes_excluded` remained false.
Only this summary was retained; it is not model, canary, independent cleanup,
or peak-resource evidence. The synthetic check used no ARC data or model.

### Public worker handoff plan

Validate the entire Phase A-style public export and its per-task hashes before
constructing a backend. Bind a selected task to the supplied run identity,
adapter state and source digest, then use the existing two-candidate worker and
frozen direct-grid prompt. Record a hash of the actual serialized public user
payload used by the direct-grid backend. Test that held-out test outputs are
rejected before backend construction and never appear in the backend prompt.
The caller must separately verify that run/source identities are the frozen
registration and that model, runtime and adapter identities are genuine; this
handoff is not an OS isolation boundary or authorization to run.

### Dataset provenance binding

Bind each successful collection row to the declared run digest, source digest,
public task hash, prompt hash and adapter-off state before verification. Reject
mismatches and do not promote a self-reported 32B, adapted or differently
prompted result. Failed rows remain task-level observations without fabricated
worker metadata; their journal identity must be independently verified by the
caller. These comparisons establish consistency only, not independent model or
process attestation.

### Adapter state identity at worker boundary

The public handoff must check the constructed backend's adapter state and digest
against the requested state before generation. Adapter-off must have no digest;
adapter-on must match the registered digest. This catches accidental local
misconfiguration, not a dishonest runtime or a mismatch of other inference
settings. The paired scorer still needs independent run and settings binding.

### Paired identity test plan

Require a supplied comparison registration with run, source, adapter and
per-task hashes. Reject completed rows whose identity differs, whose adapter-on
hash does not match, or whose adapter-off row has an adapter. Compare prompt
hashes for each complete pair. Failed rows may carry no candidates and remain
unpaired; they cannot contribute a pass. These checks bind locally recorded
claims to one another; verifying the registration and the loaded runtime is a
separate prerequisite for real evaluation.

### Registered collection worker plan

Connect the public-only handoff to the real Phase B backend through a dedicated
worker. Before loading a model, require a canonical frozen collection descriptor,
its expected digest, exactly 100 selected public task hashes, matching receipt,
fixed Phase A inference settings, a bound local source inventory and the pinned
MLX runtime versions. Verify the source inventory again after generation and
write results exclusively through the existing journal file creator. Model files
are checked by the backend. The descriptor and source inventory are local
consistency claims: the parent still must validate the full allocation, exposure
ledger and preregistration before dispatch. No descriptor is frozen by this PR
until the allocation conflict is resolved.

### Allocation receipt gate plan

Before any dispatch, recompute a receipt from the source challenge bytes, the
strict-RSI split and scheduled slots, bounded exposure, official evaluation IDs,
the Phase A frozen set, the training selection and the distinct comparison
proposal. Require exactly 100 permitted training IDs, 50 comparison IDs, no
exact ID or visible-content overlaps with protected groups, and a hash binding
the result to the worker descriptor. The existing preallocated pool must fail
this check while the five scheduled-slot overlaps remain. A synthetic or
alternative set used in tests does not amend the pool or authorize execution.

### Runtime identity test plan

The local `/tmp/qwen-p3a-mlx-venv/bin/python` probe reported arm64 Python 3.12.9,
MLX-LM 0.30.7 and MLX 0.32.4. Its venv-only inventory contained 6,251 files,
368,149,344 bytes, and SHA-256
`f864e7d3a30f5d945998a459f00963f902ce7410dacac7488184429fcfc6b9ab`.
This is a point-in-time inventory, not a frozen run or complete native-library
closure. Bind the worker descriptor to the existing runtime and environment
manifests, verify both against the executing interpreter before and after model
generation, and reject mismatches before backend construction. Continue to
report independent provenance and process isolation as unverified.

### Collection dispatcher plan

Read only the frozen public descriptor and handoff files from an owner-only
`artifacts/lt-phase-b-<run-id>` directory. Hold an exclusive local run lock,
recompute the allocation, model, runtime, source and public-export preflight,
and only then dispatch one registered worker at a time with the verified Python
and bounded supervisor. Use isolated Python mode to ignore `PYTHONPATH`; never
pass solution files or official evaluation data to the worker. Durable journal
markers prevent automatic replay of interrupted attempts. Fixture tests must
show that failed preflight and concurrent ownership never launch a child. The
local lock and consistency checks do not independently attest the runtime,
prove complete descendant cleanup, or replace the outstanding allocation choice.

The implemented collection worker currently accepts only a descriptor matching
the Phase A direct-grid inference settings and the proposed 300-second per-task
monitor ceiling. Its resource policy specifies at least 20 GiB free disk and
20% free memory, at most 512 MiB swap growth and 32 GiB sampled process-tree
RSS, with MLX's 48 GiB memory limit. These are provisional limits, not measured
peaks or a frozen registration. The runtime/environment and model-file
inventories are rechecked; the local files and process remain unauthenticated.
The dispatcher passes no solution or official evaluation path to the child and
uses `-I -B` with the selected source root. No Phase B model task has run.

A subsequent tokenizer-only preflight used the installed MLX-LM
`TokenizerWrapper` with the existing `install_training_template_policy('off')`.
For a synthetic chat example, the native `ChatDataset` prompt-mask offset
matched the full thinking-off inference prefix exactly, and the target contained
the completion. This resolves the earlier observed mask difference for that
fixture when the policy is installed; it is not evidence of a training step or
that every real example fits the sequence limit. Register and test the policy
and truncation for actual datasets before training.

### Canary tensor evidence plan

Before and after a single optimizer step, record deterministic hashes, shapes
and dtypes of every loaded base tensor and every expected trainable LoRA tensor.
Require the trainable names to equal the preregistered eight-layer rank-8 LoRA
set, require finite LoRA values, reject changed base tensors or metadata, and
require at least one LoRA tensor to change. Capture initial and final adapter
hashes separately from the saved artifact identity. Tests use small synthetic
arrays and cannot attest a live model or process. A real canary also needs a
finite loss, independent saved-adapter reload/inference, bounded resource and
cleanup evidence; no synthetic canary adapter may be promoted as Gen 1.

The pinned 14B model files contain BF16 and U32 tensors. A local native-MLX
fixture showed that direct NumPy conversion of MLX BF16 fails; the tensor
snapshot code now hashes a lossless U16 bit view and preserves the logical BF16
dtype. A synthetic MLX-array test detects both an adapter change and a base
change. No 14B model was loaded for this check.

### Synthetic canary configuration plan

Define a fixed non-ARC chat example and an explicit MLX-LM 0.30.7 CLI
configuration for exactly one optimizer step, rank 8, eight layers, batch size
1, 512 tokens, learning rate 1e-4, seed 20261111, thinking off and prompt
masking. Bind the paths and configuration before launching, keep dataset, model
and new adapter output disjoint, and never reuse the canary as Gen 1. The
configuration builder and tokenizer checks are preparation only; a real canary
still needs registered source/runtime identities, a monitored owned process,
base and adapter tensor observations, and a new-process reload/inference.

### Instrumented trainer plan

Wrap the installed MLX-LM trainer's one-step call in the tensor observer after
its own LoRA setup. Require the fixed synthetic dataset, exact CLI config,
thinking-off template policy, a single call with one optimizer step, and no
validation dataset. Persist before/after tensor evidence exclusively in a
private directory and restore the trainer hook even on failure. The parent must
still register and verify the model, runtime, source, resource limits, saved
adapter and independent reload before treating a canary as completed. Tests use
synthetic model arrays and a fake trainer; they do not execute the 14B model.

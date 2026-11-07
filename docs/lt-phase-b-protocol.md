# LT Phase B exploratory protocol — draft

**Not frozen; no Phase B execution has begun.** The machine-readable
[draft](../configs/lt.phase-b.protocol.json) is a planning record, not a run
descriptor. Its allocation, source closure, runtime inventory, model-file
verification, actual training configuration and evaluation manifest must be
bound before inference, canary training, real training or evaluation. The
original strict RSI experiment in issue #2 and its five-Oracle gate are separate
and unchanged.

## Allocation and exposure

The proposed training pool contains 100 ARC training IDs. It is disjoint by ID
from Phase A's 200 tasks, the three LT development tasks, the earlier strict-RSI
200- and 50-task evaluation allocations, the bounded known-exposure registry and
the 120 official public evaluation IDs. Five IDs also belong to the earlier
strict-RSI design's 20 scheduled **generation** slots. The preallocated pool
and protection of those slots cannot both be retained unchanged; the allocation
decision is pending. Do not expose these tasks to a model until resolved and
re-audited. The separate [50-task comparison proposal](../configs/lt.phase-b.comparison-proposal.json)
excludes the original 100 IDs and all 20 slots; it remains unfrozen.

Exact visible-content checks also reject duplicates within or across the
selected task sets, ignoring the order of examples. They do not establish
semantic independence or complete historical and pretraining non-exposure.
The official 120 public evaluation answers are reserved for Phase C. No
Phase A results may be used to choose examples, prompts, hyperparameters or a
checkpoint. Any later reuse of Phase A's 200 IDs is explicitly a reused
exploratory holdout, not untouched independent validation.

## Candidate collection and dataset

Use the pinned `mlx-community/Qwen3-14B-4bit` revision
`a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`, with the Phase A direct-grid
prompt, tokenizer, parser and scorer. The proposed inference settings are
thinking off, temperature 0.6, seed base 20261111, two recorded candidates,
and at most 4,096 tokens per candidate. Seed independence is unverified.
Only public training pairs and test inputs go to the worker. The offline
verifier sees the permitted training-task test answers afterward; only a full
task exact match is eligible. Different candidates for the same task count as
one task lineage. Persist predictions, model revision, task and prompt hashes,
source and run identities, correctness, usage and rejection reasons locally.
Do not inject human-written solutions or substitute 32B outputs or synthetic
Oracles.

Build MLX-LM chat JSONL from the verified predictions and the same public user
payload. Install the thinking-off template policy and use prompt masking; check
actual tokenized lengths and reject examples that would be truncated. With zero
verified tasks, skip real training with `insufficient_verified_self_solutions`.
One or two tasks permit only a clearly labeled overfitting demonstration. For
three or more, reserve the first `max(1, floor(n/5))` sorted eligible task IDs
for validation and keep both task IDs and source lineages disjoint. The current
observed yield is **unknown**, not zero.

## Canary and conditional training

First run a separate synthetic one-step technical canary on the pinned 14B
base, with the locally checked MLX-LM 0.30.7 runtime. The proposed configuration
is LoRA rank 8, scale 2, eight layers, batch size 1, sequence length 512,
learning rate 1e-4 and seed 20261111. Record finite loss, all base and trainable
tensor hashes before and after, adapter files and hashes, resource observations,
and a new-process reload and unscored synthetic inference sample. A synthetic
canary adapter is never Gen 1 or evidence of ARC improvement.

If the canary and observed yield support a run, use the same proposed LoRA
settings for at most 20 optimizer steps and four hours, with a task-disjoint
validation set when yield permits. Freeze the exact run-specific resource caps,
source closure and training settings before training; the real-data training
runner is not yet complete. Use the final step rather than selecting checkpoints
by validation score. Stop on non-finite loss, truncation, resource or identity
failure, or unverified required cleanup. Preserve failed attempts and report
why any step was not run.

## Matched comparison and reporting

When a real adapter is trainable and passes the applicable gates, compare
adapter-off and adapter-on runs on the same preregistered tasks with the same
base revision, prompt, parser, tokenizer, seed policy, inference settings,
candidate budget and scorer. Report paired improvements, regressions,
unchanged-right and unchanged-wrong, first-candidate and two-recorded-attempt
counts, format validity, tokens, elapsed time, raw denominators and uncertainty.
The proposed 95% Hoeffding interval assumes independent paired task
observations; the fixed selection is not a random population sample. Null or
negative changes are valid. Adapter tensor change establishes neither
held-out generalization nor strict RSI success.

The current code and local evidence are summarized in the
[inventory](lt-phase-b-inventory.md). Raw protected data, datasets, predictions
and model artifacts stay in private local `artifacts/lt-phase-b-<run-id>/`;
the tracked report must disclose the actual outcome and any not-run reasons
without publishing protected answers.

Before any real-data trainer invocation, independently audit every emitted
example against its verified journal entry, task ID and solution lineage. The
native trainer's tokenized prompt offset must equal the thinking-off inference
prefix, and every full example must fit the registered sequence ceiling. Hash
the exact private JSONL bytes; a missing validation split when required, mismatched lineage,
prefix, or truncated example must stop the run rather than be silently dropped.

Materialization must recheck full-task correctness with the offline verifier,
then create new owner-only files under the private run directory. Preserve the
exact JSONL hashes and task/lineage split in a non-answer manifest. For the
one- or two-task overfitting case, omit `valid.jsonl`: the installed trainer
cannot load an existing empty JSONL file as a validation dataset. Failed or
incomplete collections must not create training files.

## Real-data trainer configuration preparation

Before any real-data training, verify the owner-only SFT manifest and exact
JSONL file hashes and reject missing files, changed files, duplicate task IDs or
lineage, and unsupported yield states. Propose one fixed MLX-LM 0.30.7
configuration: 20 steps at most, rank 8, eight layers, batch size 1, sequence
length 512, learning rate 1e-4, seed 20261111, prompt masking and thinking off.
For a disjoint split the trainer may report validation loss, but no checkpoint
is chosen from it; for an overfitting-only yield the validation file is absent.
This configuration proposal does not itself pass the real 14B canary, establish
model or source provenance, or authorize a training launch.

### Real-data trainer instrumentation preparation

Before a supervised real-data attempt, bind the private SFT manifest and file hashes to the exact proposed trainer configuration. Instrument the installed trainer once, require the planned step count, batch size, masking-compatible dataset lengths and final adapter path, and capture base and adapter tensor hashes immediately before and after the trainer callback. Recheck private input and configuration hashes after training. A partial or failed callback retains evidence and cannot be automatically retried or promoted. This local wrapper does not register a run, supervise resources, establish independent provenance, verify reload, or authorize execution.

### Canary reload handoff

Before using a completed canary reload as input to a later training gate, replay its retained start and process markers, supervised observation, sample output, training tensor evidence and saved adapter identity. Reject any changed or missing evidence, including a mismatched saved summary. This replay remains a local consistency check and does not establish independent attestation or escaped-process cleanup.

### Supervised training completion

The supervised worker must distinguish the synthetic one-step canary from the registered 20-step real-data configuration. Preserve the one-step default; for a real-data run, require an explicit 20-step expectation and accept completion only when its retained finite loss stream ends at step 20. Both modes keep the existing 300-second worker ceiling and resource stop rules until a different limit is separately implemented and preregistered. This monitor setting does not itself authorize a run or prove the trainer executed the claimed steps.

### Registered real-training worker preparation

Bind a proposed frozen real-training descriptor to the collection, source inventory, canary descriptor and replayed canary reload, private SFT manifest, exact trainer configuration, selected training task IDs, and existing resource settings. The worker must verify runtime, environment, source and 14B model identity on both sides of the instrumented trainer callback. This descriptor-level consistency check is not authorization: the supervising dispatcher must independently replay the complete collection and offline eligibility evidence, enforce resource and time limits, and preserve one-shot failure records before launching the worker.

### Collection-to-dataset replay

Before real training, replay every registered collection task from its private journal: require matching identity, start and process markers, complete worker output and matching committed row, in-limit resource observations, and observed child/group termination. Reconstruct eligibility with the independent training-task verifier, reproduce the exact SFT examples and split, and compare their private hashes and lineage metadata to the materialized manifest. Reject missing, failed, interrupted, changed, or extra records. This is local evidence replay, not independent process attestation or a substitute for allocation and run registration checks.

### Training-answer identity

The frozen real-training descriptor must bind the exact ARC training-solutions file hash before eligibility is replayed. Recompute the collection preflight from the public challenge inventories, verify the solutions bytes against that registered hash, and construct the verifier only from selected training-task answers. Comparison-task public inputs are used only for disjointness checks. Neither answers nor reconstructed examples are included in the preflight result or passed to the trainer worker.

### One-shot real-data dispatch

Under the private run lock, replay the registered collection and dataset, validate the canary identity and reload evidence, and launch the registered trainer in a fresh isolated Python child with a 20-step monitored expectation. Persist start, PID, resource/loss observation and worker output without overwriting. Recompute the finite loss stream, tensor delta and saved adapter identity before recording an unverified, reload-pending summary. Preserve failed or interrupted attempts for reconciliation; never retry or promote automatically. Recheck the collection/data preflight after the child exits. This dispatcher does not by itself establish independent runtime attestation or escaped-descendant cleanup.

### Real-data adapter reload worker

A separate supervised worker must replay the retained real-training and synthetic-canary evidence, bind the registered model and adapter hashes, load the real-data adapter into the same 14B backend, and perform the fixed synthetic unscored reload sample. Recheck source, model, adapter and training evidence after the sample. Record local reload observations without treating them as ARC evaluation, independent attestation, or permission to promote Gen 1.

### Real-adapter reload supervision

Replay the registered data, synthetic canary and completed training evidence before starting one isolated reload child. Record its PID, resource samples, owned-process termination and output in private files. Verify the sample identity, adapter hash, token and memory bounds, recheck the inputs and previous evidence, and retain a non-promoting local reload summary. Reject altered or interrupted attempts and require reconciliation rather than automatic retries.

### Comparison reload evidence handoff

Before a matched comparison uses a real adapter, replay the saved real-reload start and process markers, monitored output and sample, prior training and canary evidence, and the saved summary. Reject missing or altered records. This handoff only reports local consistency; it does not promote the adapter or establish independent cleanup or evaluation validity.

### Matched public comparison worker

A frozen comparison descriptor must bind the distinct 50 public task identities, the existing direct-grid prompt and generation settings, source/runtime/model manifests, the real training and canary identities, and the saved real adapter hash. For each task, use the same descriptor and backend settings for adapter-off and adapter-on, changing only adapter state and its hash. The worker receives no evaluation answers. Its local checks and output are not independent attestation or scored evidence; a supervised journal and offline paired scorer remain necessary.

### Matched comparison journal

Keep separate private journals for adapter-off and adapter-on under one comparison identity. Finish all registered off-state tasks before starting on-state tasks, with one supervised worker at a time. Validate the public task, prompt, source, run and adapter identities before committing each completed row; retain timeouts and errors and stop rather than retry. Before resuming, replay existing start, process, resource, worker and result records and reject incomplete or changed evidence. The journal contains predictions but no held-out answers; scoring remains offline after collection.

### Matched comparison dispatch

Under the private run lock, verify the registered collection, training-data replay, canary, real-adapter reload and comparison descriptor. Reconstruct the comparison public export from the fixed challenge inventory and derive the per-task prompt hashes. Run at most the requested number of sequential, resource-monitored isolated children through the two comparison journals, with no solution data in their arguments. Recheck inputs and evidence after the batch; failed or interrupted work remains recorded for reconciliation. This dispatcher does not score results or establish independent attestation.

### Offline paired scoring handoff

Before exposing comparison answers to the offline scorer, verify that both state journals are complete and replay every retained worker, identity and resource record against the fixed comparison registration. Only then pass the recorded predictions and selected held-out answers to the existing paired scorer. Missing, failed, interrupted or changed work must not be scored as a complete comparison; no answers are written into the report or sent to generation workers. Local replay does not establish independent attestation.

### Offline scoring entry point

Under the same private run lock, revalidate the frozen comparison inputs, prior training and reload evidence, and every completed batch marker. Require both journals to be complete and replay their records before selecting comparison answers from the registered solutions file. Write the paired per-task score and cost report once inside the private run directory; do not dispatch workers from the scoring path or include correct answers. Recheck inputs and evidence before persisting the report.

### Collection-to-SFT preparation handoff

Reconstruct the dataset directly from a fully replayed collection journal and the selected-task verifier before private SFT materialization. Keep reconstructed records and predictions private, preserve the zero-yield status without making synthetic substitutions, and require the existing tokenizer and independent-verification audit before creating training JSONL. The source, allocation and answer-file identities remain separate preflight requirements.

### Private SFT preparation

Use a separately frozen data descriptor binding the collection and training-solutions file identity to reconstruct the private eligibility journal from completed collection records. Verify selected training answers offline, then materialize only independently verified examples with the native tokenizer and masking audit. Persist the eligibility journal owner-only, retain all rejection reasons, and record zero yield without creating synthetic training examples. Existing or partial artifacts require reconciliation.

### Real-training operator commands

Expose the existing one-shot real-training and real-adapter reload dispatchers through command-line entry points with explicit registered hashes and local paths. Commands must pass the paths to the dispatchers without printing private data; dispatcher preflight, locks, resource checks and reconciliation behavior remain authoritative.

### SFT preparation entry point

Read the registered data and collection descriptors from the private run directory and the explicitly named local challenge, solution and official challenge inventory files. Verify that the preparing Python is the registered interpreter and run collection preflight before loading the pinned tokenizer. Use the installed MLX-LM chat dataset and thinking-off policy to prepare private eligibility and SFT artifacts; print only the count/status summary. This entry point does not load model weights or authorize training.

### Collection registration proposal

Build a draft collection descriptor only after recomputing the 100-task allocation against public inventories, then capture local source, 14B model, interpreter and environment identities and construct the public-only task export. Validate the assembled shape against the existing collection contract. The builder writes no run artifacts and returns a draft, non-authorizing proposal; it neither resolves the current allocation conflict nor freezes or starts an experiment. Recompute the full preflight when a future run is actually registered.

### Draft registration integration check

A local, opt-in integration check may assemble a draft descriptor using an ephemeral eligible fixture pool and the installed runtime and model-file inventory. It must keep the pool in memory only, load no model weights or solution answers, write no run artifacts, and assert that the result remains draft and non-authorizing. This fixture is not a decision about the five overlapping IDs or a real experiment registration.

### Draft registration operator handoff

An operator command may save the draft proposal once in a private `lt-phase-b-<run-id>` artifact directory, using an explicitly supplied pool file and public challenge inventories. It must refuse existing output, print only status, counts and hashes, and leave the proposed descriptor marked draft. The command must not choose replacement IDs, freeze a run, expose answers, or launch a model.

### Synthetic canary registration proposal

Given a valid registered collection descriptor and an explicit private synthetic-data location, construct a draft canary descriptor and fixed trainer configuration using the existing one-step contract. Bind the collection, source, model, runtime, exact synthetic JSONL and config bytes. Verify that any existing synthetic-data file is exactly the fixed example, and refuse an existing adapter. The proposal is not frozen, starts no trainer, and cannot by itself authorize the canary.

### Real-training registration proposal

After a complete collection and observed canary reload, construct a draft real-training descriptor from the verified private dataset, the predeclared fixed trainer configuration, registered answer-file hash and resource settings. Replay collection eligibility and canary evidence before returning it, and recheck both afterward. Keep the proposal non-authorizing; it must not start training or choose a checkpoint based on validation or held-out scores.

### Matched comparison registration proposal

After real training and its separately observed reload, build a draft comparison descriptor from the already allocated public comparison tasks, the unchanged collection generation settings and the exact reloaded adapter hash. Recheck the collection/data preflight and saved reload evidence before and after building the public-only export. The proposal does not examine comparison answers or scores, freeze evaluation, or authorize inference.

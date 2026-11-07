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
the exact private JSONL bytes; a missing validation split, mismatched lineage,
prefix, or truncated example must stop the run rather than be silently dropped.

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

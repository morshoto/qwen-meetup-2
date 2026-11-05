# Qwen RSI research protocol, v1

Status: **design only, not an execution registration**. No model calls, training,
spending, or submission are authorized by this document. Issue #1 defines the
design; issue #2 covers implementation and separately authorized execution.
`configs/rsi-research-protocol.json` records the design's key fixed boundaries;
it is not consumed by an experiment runner. An execution must first freeze a
complete dependency-closed registration and its budgets. Do not overwrite this
version after a run begins.

## Question and evidence

Can Qwen generate verified, diverse training data that improves the same Qwen
model on unseen ARC-AGI-2 tasks, and can the resulting system reach a competitive
independently verified score? Improvement and competitiveness are separate claims.
Training, a generated Oracle, or a successful runtime do not establish either.

The corrected E8 comparison produced 0, 1, and 1 strict new Oracles for 8B,
14B, and 32B in ten slots each; E9's audited extra collections produced zero new
Oracles and left 14B and 32B at one eligible example each. No E9 training or
adapter comparison ran. See [E8](e8-full-rerun-20261104.md) and
[E9](e9-dual-model-rsi.md). E9's initial 14B batch was excluded from the strict
gate. These observations motivate measuring data yield before training.

[SOAR](https://openreview.net/attachment?id=z4IG090qt2&name=pdf) alternates
model-generated program search and training on generated examples, including
relabeling program outputs; its reported ARC-test results are ARC-AGI-1, not a
verified ARC-AGI-2 result. It uses far more search and compute than this local
proposal. [Akyurek et al.](https://proceedings.mlr.press/v267/akyurek25a.html)
study test-time training, which is distinct from our persistent generation-to-
generation weight update. Neither establishes that this proposed loop works.

## Target and comparison

Fix a **26.0% ARC-AGI-2 semi-private score** as an ambitious numerical reference
before evaluation: ARC Prize's [community leaderboard](https://arcprize.org/leaderboard/community)
lists Efficient Evolutionary Program Synthesis at 26.0% and $476, verified on
that split. It is an open-source system reference, not a matched-compute or
same-model comparator; its inference method and resource accounting differ.
Record actual cost and hardware for both when available. Meeting 26% alone does
not establish equal efficiency or parity with current frontier systems. The
reference is fixed for this design and must not be chosen after observing results.
A directly comparable verified open-weight system with published compute has not
yet been established: document one before making a matched-compute claim.

The primary causal comparison is candidate versus untrained base: same Qwen
revision, task IDs, harness, seeds, sampling settings, attempts, and per-task
inference budget. Report paired task outcomes, exact solved counts, score difference
in percentage points, and a paired bootstrap 95% interval over tasks. Improvement
is established only if the candidate's point estimate is positive and the lower
interval bound exceeds zero. A non-positive or inconclusive result is not success.
One generation can show improvement; a recursive-improvement claim requires at
least two successive independently audited generation transitions, each compared
with its immediate parent under the same criterion. Freeze both comparisons
before final evaluation; development screening may decide whether generation 2
exists, but final results cannot influence either model or the harness. Never
select a candidate using the final evaluation results. If independent scoring is
available for only one model, report the other comparisons as local evidence.

Use exact output matching with at most two predictions per test input; report the
organizer's actual score and the task-level metric separately if the applicable
scorer differs. Public evaluation is a final, once-only diagnostic, not a tuning
set. The primary competitive claim requires a score independently verified on
semi-private ARC-AGI-2. The [testing policy](https://arcprize.org/policy) says
community submissions are normally self-reported and independent verification is
selective; availability is not guaranteed. If unavailable, report local results
only and leave the competitive claim unverified.

## Data and loop

1. Use the existing strict E9 provenance model: pin Qwen3-14B revision and
   quantization, chat template, environment, sandbox, verifier and complete
   transitive execution dependencies. Generation 0 is the untrained base.
   Independently audit any legacy same-model records before including them;
   exclude the E9 initial 14B run and unpinned E7 records.
2. Split the public ARC-AGI-2 training set by task before generating anything.
   The earlier proposal of the last 200 sorted IDs for development overlaps 50
   IDs in the checked-in P3-C prior-use registry; do not use that split. Audit
   all later E2/E3 and other retained task-use records as well as the registry,
   then choose the last 200 sorted IDs outside the resulting complete exposure
   set for development; assign the remaining 800 to generation and record hashes
   of both manifests. Stop if the exposure audit is incomplete or leaves fewer
   than 200 eligible tasks. The registry alone is not a complete exposure audit.
   Do not use public
   evaluation or hidden evaluation tasks for generation, tuning, or selection.
   Check source dataset hashes and keep generated descendants with their parent
   task in the same partition. Before including historical records, audit their
   complete ancestry against the development manifest; exclude any record whose
   parent overlaps it or whose ancestry cannot be established. Independence of
   the proposed development split remains unestablished until the history audit
   and manifest freeze are complete. Model pretraining exposure is
   unknown and must be disclosed; this split cannot prove absence of contamination.
   A [provisional split](../research/rsi/provisional-split.json) records an
   offline scan of the retained local artifacts: 317 of the 1,000 source tasks
   appeared in the union of that scan and the older registry. It records source
   hashes and uses `src.arc.rsi_split.make_split` to choose 200 development
   tasks outside those known IDs. The snapshot inventories 373 retained input
   files and configurations, including E5 generation records, requires the named E2/E3 manifests, fails on
   malformed inputs, and checks for changes while scanning. The historical
   scanner alone silently skips unreadable files, so use the wrapper in
   `src.arc.rsi_history.provisional_snapshot`, which validates the inventory
   before invoking it. This is not the final manifest: the inventory cannot
   establish that every historical source was retained or recognized.
   Independently audit missing history before use.
3. For each generation select ten successive task IDs in sorted generation
   order: positions 0–9 for the first generation and 10–19 for the second.
   For each selected task use the existing `solve_prompt` format: expose its
   demonstration input/output pairs and test inputs, but never its test outputs.
   Qwen writes a `transform(grid)` program; allow up to two model-authored
   repairs using only sanitized execution and visible-demonstration feedback.
   Run the accepted program on the source task's demonstration inputs and test
   inputs to produce synthetic input/output examples. These generated outputs
   are not ground-truth labels for the source task. The E9 generator currently
   uses a generic prompt with no task input, so this task-conditioned generation
   and its provenance audit must be implemented and tested in issue #2; E9
   collection results cannot be treated as results from this design.
   Execute in the existing network-isolated sandbox with bounded resources. A generic
   verifier checks format, determinism, safety, output validity and independent
   probes; model-authored program output may be used as a synthetic label but is
   labeled **generator distillation**, not a verified real ARC solution. Do not
   supply human-authored ARC transformations, operator search, solution code, or
   task-specific heuristic curricula. General infrastructure and validity checks
   are permitted.
4. Improve diversity through model-authored variations and feedback on generic
   execution errors and duplicate fingerprints. Exclude duplicate ASTs and
   sampled behavioral matches using the preregistered reference pool; finite
   probes detect possible duplicates, not semantic equivalence. Measure attempted,
   valid, strict-new, duplicate, and rejected candidates separately; retain all
   reasons and raw provenance. Never feed private probe outputs back to Qwen.
5. Require at least five distinct strict same-model Oracles with sealed source
   and data provenance before any training. Include only newly generated records
   from the current model generation in this gate; historical records are
   duplicate references only. Five is an existing feasibility gate,
   not evidence that the dataset is sufficient for generalization. Freeze SFT
   input hashes, parent model identity, split, loss mask, seed, hyperparameters,
   adapter configuration and training budget before training. First perform a
   separately authorized one-step synthetic-canary preflight; stop on OOM,
   sustained swap growth, non-finite loss, invalid reload or resource cap.
6. Train one adapter, reload it independently and compare it with its immediate
   parent on the development split under the matched protocol: the parent is the
   untrained base for generation 1 and the selected generation-1 adapter for
   generation 2. Generation 2 continues the selected parent adapter, using only
   its new generation's eligible records and no replay. Select only a candidate
   with a positive paired point estimate and no invalid runs; this is a screening
   rule, not proof of improvement. Repeat once at most with the chosen adapter as
   generator, using fresh seeds and no evaluation feedback. Freeze final model,
   harness, data and budget before the once-only public diagnostic and any
   independent submission. A generation that misses a gate ends the loop.

## Resources, limits, and registration

The inspected host is an Apple M4 Pro with 64 GiB unified memory. E8/E9 recorded
successful 14B and 32B inference without observed swap growth, but those samples
do not establish QLoRA training feasibility. Start with 14B only; do not infer
32B training feasibility. Before any live run independently record free disk,
RAM pressure, runtime versions, model hashes and Docker image. Preserve all
existing sealed artifacts.

Design caps for each proposed collection stage are at most 32 calls, 200,000
reserved tokens, 300 sandbox starts and 3,600 seconds, matching the E8 per-model
ceiling; exact slots, seeds, prompt and dependency closure must be registered
before the run. Each generation uses ten slots, temperature 0.2, at most 512
new tokens per request, Thinking off, and at most two repairs per slot; canaries
count toward the 32-call stage cap. Request seed is `12000 + 100 * generation
+ 3 * slot + attempt` for zero-based generation, slot, and attempt indices;
requested seeds do not guarantee deterministic inference. Hash the exact prompt,
model revision, task manifest, verifier, reference pool, and dependency closure
in the execution registration. The existing E9 runner does not implement this
task-conditioned schedule.

For each candidate-versus-parent development comparison, propose at most 200
tasks × two model arms × two independent program samples, or 800 model calls;
cap total reserved tokens at 16 million, sandbox starts at 8,000, and wall time
at 36,000 seconds. These are proposed ceilings, not measured resource needs or
authorization. Setup canaries require their own separately registered cap. Use
the same `solve_prompt` format and generation settings for both arms, run each
program on visible demonstrations and test inputs; retain candidates that pass
all visible demonstrations and submit at most two valid outputs per test input
in candidate order. Invalid model outputs and rejected programs count as failed
attempts. Permit at most ten sandbox starts per sample; abort and report an
incomplete comparison if a task exceeds that cap or infrastructure fails,
rather than excluding the task. Do not expose test outputs to the model.
Freeze task-level results before computing a 10,000-resample paired bootstrap
with seed 42; report all outcomes, including incomplete runs. These limits are
per comparison and cannot be transferred to another stage.

The proposed training configuration follows the existing 8B
pilot, without assuming 14B will fit: batch size 1, rank 8, alpha 16, learning
rate 0.0001, maximum sequence length 2048, two adapted layers, and at most 20
steps and 1,800 seconds per generation. A separately authorized preflight uses
one step. The preflight and full run each need numeric memory, disk, and
preflight wall-time caps based on a measured resource assessment before
registration; these design limits are ceilings, not permission to execute.
Budget exhaustion, provenance
mismatch, incomplete audit, inadequate strict data, failed preflight or a failed
comparison stops the dependent stage. Do not transfer unused budgets or silently
retry. All future live inference, training, external compute and submission need
separate authorization.

Before execution, create a new immutable registration specifying manifests and
hashes, exact algorithm and prompts, run seeds, sampling parameters, safety and
verification code, training hyperparameters, candidate selection, per-stage caps,
scorer version and artifact paths. Resolve feasibility and obtain explicit
resource authorization first. If any field is unresolved, the design is not an
executable preregistration. Check current organizer submission rules before
submission; do not assume a competition remains open or independent scoring is
available.

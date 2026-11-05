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
least two successive independently audited generation transitions meeting the
same criterion. Never select a candidate using the final evaluation results.

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
2. Split the public ARC-AGI-2 training set by task before generating anything:
   sort task IDs and assign the first 800 to generation and the remaining 200
   to development, recording hashes of both manifests. Do not use public
   evaluation or hidden evaluation tasks for generation, tuning, or selection.
   Check source dataset hashes and keep generated descendants with their parent
   task in the same partition. Model pretraining exposure is unknown and must be
   disclosed; this split cannot prove absence of contamination.
3. On generation tasks, Qwen writes and repairs candidate programs. Execute in
   the existing network-isolated sandbox with bounded resources. A generic
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
   and data provenance before any training. Five is an existing feasibility gate,
   not evidence that the dataset is sufficient for generalization. Freeze SFT
   input hashes, parent model identity, split, loss mask, seed, hyperparameters,
   adapter configuration and training budget before training. First perform a
   separately authorized one-step synthetic-canary preflight; stop on OOM,
   sustained swap growth, non-finite loss, invalid reload or resource cap.
6. Train one adapter, reload it independently and compare it with the base on
   the development split under the matched protocol. Select only a candidate
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
before the run. A training preflight and training run each need separate numeric
step, wall-time, memory and disk caps based on a measured resource assessment;
no such caps or spending are authorized here. Budget exhaustion, provenance
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

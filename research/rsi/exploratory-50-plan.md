# Exploratory 50-task matched comparison

The user requested an exploratory comparison after training the Qwen3-14B RSI
adapter, followed by a separate 200-task evaluation if the results justify it.
The current experiment has no trained adapter from this protocol. This plan and
the [notebook](../../nb/13_rsi_exploratory_50.ipynb) do not run or authorize a
model, training, external compute, or benchmark submission.

## Task separation

The [local split proposal](exploratory-split-proposal.json) takes the latest
bounded-history proposal, reserves the last 50 sorted generation-partition task
IDs after excluding known historical exposures and the first 20 scheduled
generation tasks, and keeps them out of generation. It leaves 750 generation,
50 exploratory and the original 200 development task IDs. The first 20
scheduled generation IDs do not change. All three lists remain provisional:
independent history review, source hashes, content-overlap checks, and a new
execution registration must be completed before use. Existing components that
assume 800 generation tasks must be updated and audited for the 750-task
manifest. Unknown historical and pretraining exposure must be reported.

The 50 tasks must never be used for model generation or training, and their
outcomes must not be mixed into the later 200-task score. Record any decisions
made after seeing the 50-task results before running the 200 tasks. The 200-task
comparison is still a development evaluation, not an independently verified
ARC-AGI-2 leaderboard score.

## Matched comparison and notebook

Compare the same base model and revision, with the trained adapter as the only
intended model difference. Freeze the task and seed manifests, prompts, model
settings, execution harness, limits and independently reviewed budgets before
either arm runs. Record incomplete requests and infrastructure errors instead
of assigning fabricated scores. Task-type labels are reviewer annotations, with
their method recorded, rather than an assumed official ARC taxonomy.

The notebook reads caller-supplied task records, displays correct counts, task
type, failure reasons, elapsed time and server-reported prompt/completion token
usage, and preserves missing measurements as unknown. Its consistency checks do
not attest model identity, task provenance, the timing measurement or token
usage. This 50-task check cannot establish RSI improvement or benchmark
competitiveness; retain the 200-task and independent scoring gates for those
claims.

A local check of the proposed tasks found 50 distinct prompt-content hashes and
no exact prompt-content overlap with the proposed 200-task set or the first 20
scheduled generation tasks. This is not an independently sealed manifest and
does not establish semantic independence or exclude later training ancestry.

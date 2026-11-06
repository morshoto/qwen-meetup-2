# RSI resume: separate-context review

Reviewer: `/root/independent_review`, a separate read-only Codex model context,
2026-11-06. This is not a human or independent organizational approval. Scope:
the provisional split, current source packet, retained candidate identity and
visible containment/cleanup evidence. It is not a full replay of historical
seals or a review of all private artifacts. No model call, training or
evaluation was performed.

## Findings

1. A standard-library implementation, separate from the allocation review
   code, found 750 generation, 50 exploratory and 200 later-evaluation IDs,
   pairwise disjoint and covering all 1,000 dataset keys. Intersection counts
   with the recorded known-exposure set are 317, 0 and 0. Canonical visible
   content hashes were unique within and across the three groups. This does
   not establish complete history, semantic separation or pretraining novelty.
2. The current provisional `protocol.lock` SHA-256
   `6a547adc1644cec3f5fbe493b7e48c0c2e2541b7cdc8f2dd2ceda2aad5b56b1b`
   and its budget, 265-file source inventory, nine input hashes and plan hash
   matched current files. The packet refers to stale v22 canary registrations
   and generation drafts. It does not bind an actual export, complete runtime
   and model-weight closure, a reviewed launch configuration, verified cleanup
   or enforced data isolation. It is not a launch registration.
3. Following the retained-source identity procedure in
   `oracle-probe-review-guide.md`, the reviewer found each E8 sanitized record
   matches its retained E9 record and the declared model/revision, source and
   AST hashes. The references explicitly do not confer training eligibility.
   This is not a complete historical seal, request provenance or sandbox audit.
4. The fork/network sandbox permits default operations besides its denials;
   neither the export/loader nor the profile proves evaluation-file isolation.
   The group-signal `EPERM` and cleanup evidence remain unresolved.

**Verdict:** local allocation arithmetic and retained-record consistency are
supported within this scope. The evidence reviewed does not approve a real
model canary or live Oracle collection. Required independent review of the
complete launch closure and safety boundaries remains open.

## Inputs and reproduction

- Challenge dataset SHA-256
  `779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5`.
- Bounded proposal SHA-256
  `67b421802ffd521b18e8170b7f79b11f03ba2822879a49a17898a0fc67706512`.
- Exploratory selection SHA-256
  `68ef1fc9c0c081d8f90e08506a4fcbdd795715315e01736da277314c216892de`.
- 14B and 32B historical candidate-index SHA-256 values:
  `a7d46234ec80fc1c1997d5bd64ce7bc21d303dc1ee7c196ac45b38c26aeec7a5`,
  `84fcc3aefe8203994ffa2feb139d62a8c7bd78ec45b2dbf90e9f3a7e1367000f`.
- 14B and 32B static-reference SHA-256 values:
  `3344002bfd31d77566db11a96079b3e36919a1bb21f43d6beef185561f7d7c4d`,
  `df2c358c7bfcc328521013bf87e4e6e2d2920a51f6644dcd4ad9bf069cb73fd8`.

For the separate split check, parse the three input JSON files with Python's
standard library, compare each list length with its set length, check the
partition intersections and union, and intersect each set with the recorded
known-exposure IDs. For every task hash canonical JSON containing its `train`
array and `test` array restricted to `input` fields using sorted keys and
compact separators; compare the digest sets without printing task content.
The candidate check follows the explicit read-only command in the probe guide.

## Follow-up review of v6

The same separate reviewer context read the current tracker, process monitor,
training/reload streams and tiny fixture. At review time v6's 266 source files
matched its recorded inventory. The reviewer identified that the tiny
pipeline could start reload even when training cleanup was unverified, and
that its `finally` paths call `poll()` before deciding on group cleanup,
potentially reaping an exited leader. A timed-out daemon sampler can also
continue running and mutate its tracker after the watchdog has stopped. The
reviewer reiterated that sampled process lists, second-resolution start time,
inherited IPC and file aliases cannot establish full cleanup or isolation.
Neither model was approved for execution. This is not human or external
organizational approval.

After this review, the tiny pipeline was changed to stop before reload when
cleanup is unverified, so the reviewed v6 source identity is now stale. The
`poll()` cleanup path and timed-out sampler mutation remain open. The prior
tiny adapter reload observation is historical and must not be treated as
passing the newly enforced cleanup boundary.

## Follow-up review of v9

The same separate reviewer context confirmed that the tiny fixture now stops
before reload when cleanup is unverified and that a timed-out raw process-row
sampler cannot later mutate the fixture tracker. Generic sampler callbacks may
still have side effects, and the daemon worker cannot be cancelled. The v9
source inventory matched 266 files at review time. The reviewer identified
that if group observation fails after the owned child has exited, the fixture
could leave that child unreaped. Neither model was approved for launch; this
remains a separate model context, not human or organizational approval.

After the review, the fixture was changed to reap the already-confirmed
exited owned child if group observation fails, without signalling the
unverified group. Descendants may remain and cleanup is still unverified.
That source change makes the reviewed v9 identity stale.

## Follow-up review of v10

The separate reviewer checked v10 protocol SHA-256
`db924fb27e05c0edfb80b6e325eef1412601d6d71a37bef3606189874ab0a4a8`,
its 266-file source inventory, recorded input hashes and budget; all matched at
review time. The exited-child fallback reaps the confirmed owned child without
signalling an unverified group, addressing the observed zombie case. The test
uses a child without descendants: surviving same-group or escaped processes
remain possible, and reaping releases the leader identity for future PID-based
group signalling. If `waitid` fails, ownership/termination remains unknown.
The actual generation isolation boundary is also unestablished. Verdict:
neither real canary nor live Oracle collection is approved. This review is a
separate AI context, not human or organizational approval.

<!--
Thanks for the PR. A short, structured description makes review faster
and gives a paper trail when behavior changes show up later in `git blame`.
-->

## What changed

<!-- One or two sentences. What does this PR actually do? -->

## Why

<!-- The user-facing or technical reason. Link to issues this addresses. -->
Closes #

## How I tested

<!--
Specific commands or scenarios you ran. For new endpoints, paste a working
curl. For visual changes, attach before/after screenshots. For perf work,
include before/after numbers.
-->

- [ ] `pytest` passes locally
- [ ] `ruff check` clean
- [ ] `pyright` clean (or no new errors)
- [ ] Tested against `modal serve` with a real image
- [ ] If new endpoint: added a happy-path test in `tests/`
- [ ] If bug fix: added a regression test that fails without the fix

## Trade-offs / things to watch

<!--
Anything you're unsure about, anything that adds latency, memory, or
new dependencies. If a model swap, include the before/after benchmarks
or sample images.
-->

## Notes for reviewers

<!-- Anything tricky in the diff worth flagging up front. -->

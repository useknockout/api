# Contributing to useknockout

Thanks for being here. This is a young project — ~50 GitHub stars and ~3k SDK downloads at the time of writing — and feedback from real users genuinely shapes what gets built next. Issues, PRs, model suggestions, doc fixes, typo fixes: all welcome.

## Project shape

useknockout is a single FastAPI service running on Modal that bundles SOTA vision models behind a clean REST API. Most of the API lives in [`main.py`](./main.py) — model loading inside `@modal.enter`, tiled inference helpers, matting refinement, and per-endpoint handlers all live there. As the surface grows we'll extract; for now, single-file is the feature.

The hosted endpoint at `useknockout--api.modal.run` runs the same code in this repo with weights baked into the Docker image for fast cold starts.

## Running locally

You'll need [Modal](https://modal.com) (free tier works fine) and Python 3.11+.

```bash
git clone https://github.com/useknockout/api
cd api
pip install -e ".[dev]"

# auth modal once
modal token new

# hot-reloading dev endpoint — every save re-deploys
modal serve main.py
```

`modal serve` connects to your Modal workspace and gives you a temporary URL. Logs stream to your terminal. For a one-off invocation:

```bash
modal run main.py
```

## Filing issues

Please use the templates: **bug report**, **feature request**, **model suggestion**. They're structured to save us both triage time. If your input genuinely doesn't fit a template, the templates are a starting point, not a constraint — open a "feature request" with whatever framing works.

**Don't paste API tokens in issues.** Use `$TOKEN` in command examples. Public-beta tokens are public on purpose, but keys you mint for yourself shouldn't be.

## Pull requests

1. **Open a draft PR early.** Easier to redirect at 50 lines than at 500.
2. **One concern per PR.** "Add /colorize endpoint" + "fix /upscale tile size" is two PRs.
3. **Tests with bug fixes.** A regression test that fails without your fix is a stronger argument than a paragraph in the PR description.
4. **Tests with new endpoints.** Add a happy-path test in `tests/` that hits the endpoint with a real image fixture and asserts shape + content-type.
5. **Commits: imperative, present tense.** "add /colorize endpoint" not "added /colorize endpoint" or "this PR adds /colorize".
6. **CI must pass.** `pytest`, `ruff check`, and `pyright` run on every push.

## In scope

- New endpoints that extend the surface naturally (`/colorize`, `/inpaint`, `/relight`, `/depth`)
- Model swaps that hold the `~200ms warm` baseline within ~2× — i.e. anything that'd push P50 past ~500ms warm needs a strong reason
- New SDKs (Go, Rust, PHP, Ruby — please open an issue first so I can pre-create the npm/PyPI scope)
- Documentation, examples, demo assets — yes, always
- Bug fixes with regression tests
- Performance work (tile sizing, fp16 paths, model quantization) with before/after numbers

## Out of scope (for now)

- A full web app — the playground at [useknockout.com/playground](https://useknockout.com/playground) is intentionally narrow
- A different cloud backend — Modal is the deploy target. Self-hosting on any GPU box via Docker works and is documented
- Massive model pivots without prior discussion (e.g., swapping BiRefNet for a 30-second diffusion model as the default — open a `[Model]` issue first)
- Auth providers beyond bearer tokens. Simplicity is the feature; OAuth/SSO belongs in your gateway, not in this API

## Be patient

This is a side project run by one person. Most issues get a real reply within 48 hours but it's not always next-day. If something is genuinely blocking your work, say so explicitly in the issue and I'll prioritize.

## License

[MIT](./LICENSE), all the way down. PRs are accepted under the same MIT license. By submitting a PR you confirm you have the right to do so.

Thanks for being here. 🥊

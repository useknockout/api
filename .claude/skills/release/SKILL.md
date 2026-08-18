---
name: release
description: Run the full useknockout release train in order - deploy to Modal, verify health and endpoints, smoke test, GitHub release, SDK propagation checklist, publish verification, and cross-repo commit audit. Use when the user says "release", "deploy", "ship vX.Y.Z", "deploy to modal", or after any endpoint change in main.py that needs to reach production and the SDKs.
---

# useknockout Release Train

This codifies the release process that was previously re-derived by hand for every feature (/colorize, /silhouette, /inpaint, /collage). Run the stages IN ORDER and report a stage-by-stage pass/fail summary at the end. A paying customer depends on this API - never skip verification stages.

## Repo map

| Repo | Path | Publishes to |
| :--- | :--- | :--- |
| API | C:\Users\Troy\projects\useknockout-api | Modal (main.py) |
| Node SDK | C:\Users\Troy\projects\useknockout-node | npm |
| Python SDK | C:\Users\Troy\projects\useknockout-python | PyPI |
| CLI | C:\Users\Troy\projects\useknockout-cli | npm |
| React SDK | C:\Users\Troy\projects\useknockout-react | npm |
| Frontend | C:\Users\Troy\projects\useknockout-landing-page | Vercel CLI ONLY (git push does NOT deploy it) - owned by a separate session, coordinate via /handoff |

Prod URL: `https://useknockout--api.modal.run`. Test key: read from `.env` - NEVER echo a key into chat or hardcode it in commands that get logged.

## Stage 1: Pre-flight

```
cd /c/Users/Troy/projects/useknockout-api
python -m py_compile main.py && echo "COMPILE OK"
git status --porcelain
```
If there are unrelated uncommitted changes, list them and confirm scope with Troy before committing anything (his standing rule: commit only the files worked on).

## Stage 2: Deploy

The UTF-8 prefix is REQUIRED on Windows or the deploy crashes with a charmap error:
```
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal deploy main.py 2>&1 | tail -n 5
```
Run as a background task; deploys take minutes.

If the change touches REQUEST PARAMETERS or routing (new Form field, new param
handling), force-drain old containers immediately after the deploy:
```
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal app stop api --yes
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal deploy main.py 2>&1 | tail -n 3
```
Old warm containers keep serving during drain and SILENTLY IGNORE unknown form
fields, so callers see the new parameter "not working" nondeterministically.
This has caused four incidents (twice on dev, engine=product-v1 on prod
2026-08-18 reported by the partner as a bug). Cost: ~2 min of cold starts.

## Stage 3: Verify live

Cold starts run 120-180s, so warm first with a long timeout, then smoke:
```
curl -s --max-time 180 "$BASE/health"
curl -s "$BASE/" | grep -o '"endpoints":[0-9]*'     # endpoint count must match expectation
```
Then POST one real image to the changed endpoint(s) with the key from .env and confirm a valid image comes back (check output file size is plausible, not a 395-byte error body - that exact failure shipped once).

## Stage 4: Release + SDK propagation

Note: SDK versions are INDEPENDENT of the API version (SDKs on their own line, e.g. 0.7.0, while the API is at 0.11.0). Do not assume they share a number.

1. `gh release create vX.Y.Z` on the API repo with change notes (API version).
2. For each SDK repo the change affects (new endpoint or new param means ALL FOUR: node, python, react, cli): add the method, run `npx tsc --noEmit` (node/cli/react) or `py -m build` sanity for python, bump the SDK's own version, commit, tag, push.
3. PUBLISH IS MANUAL - Claude CANNOT run it (npm/PyPI require 2FA). Do NOT call `npm publish`/`twine` yourself. Instead, once all 4 are committed+tagged+pushed, hand Troy this exact block to run in order (node → python → react → cli):
   ```
   cd C:\Users\Troy\projects\useknockout-node ; npm publish
   cd C:\Users\Troy\projects\useknockout-python ; py -m build ; py -m twine upload dist/*
   cd C:\Users\Troy\projects\useknockout-react ; npm publish
   cd C:\Users\Troy\projects\useknockout-cli ; npm publish
   ```
   After he confirms, THEN verify the registries actually updated (`npm view <pkg> version`, `pip index versions <pkg>`) - "command ran" is not "published".
4. Update stale endpoint counts: `grep -rn "endpoint" README.md APIREFERENCE.md POSTS.md` and fix any "N endpoints" strings - these drifted three separate times before.

## Stage 5: Cross-repo commit audit

```
for r in useknockout-api useknockout-node useknockout-python useknockout-cli useknockout-react; do
  echo "=== $r ==="; git -C /c/Users/Troy/projects/$r status --porcelain; git -C /c/Users/Troy/projects/$r log origin/main..HEAD --oneline
done
```
Anything uncommitted or unpushed fails the release. (Frontend collaborators once noticed the API branch had no commits for a week before Troy did.)

## Stage 6: Frontend handoff

If the change affects docs, pricing, or the playground, write the brief for the frontend session with `/handoff` (it lives in useknockout-landing-page). Do not edit that repo from this session.

## Final report

Output a table: stage, pass/fail, evidence (deploy tail, health code, endpoint count, registry versions, audit result). If any stage failed, say plainly which and stop - no "mostly done".

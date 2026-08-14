# Spec: prepaid credits + `bg` param

Status: DRAFT, not implemented. Written 2026-08-13.
Two independent pieces. The `bg` param is small and can ship alone; credits is a
cross-repo project.

---

# Part 1 — `bg` param on `/remove` (small, ship anytime)

## Problem

Callers who want a white background make two decisions today: `/remove` for
transparent, `/replace-bg` for a colour. Competitors expose one control with
four choices (transparent / white / black / custom) and it is the first thing a
user reaches for in a playground.

## Design

Add to `POST /remove` and `POST /remove-url`:

| Field | Type | Default | Values |
| :--- | :--- | :--- | :--- |
| `bg` | str | `transparent` | `transparent`, `white`, `black`, or `#RRGGBB` / `#RGB` |

- `transparent` = today's behaviour exactly. Default unchanged, so every
  existing caller is byte-identical.
- Anything else composites onto that colour and returns an opaque image.
- Invalid value -> 400 `"bg must be 'transparent', 'white', 'black', or a hex colour"`.

## Implementation

`_parse_color` (main.py:844) already parses hex. `_composite_on_bg` already
takes a colour tuple. So:

```python
def _check_bg(self, bg: str):
    """Returns None for transparent, else an RGB tuple."""
    b = (bg or "transparent").strip().lower()
    if b == "transparent":
        return None
    if b == "white":
        return (255, 255, 255)
    if b == "black":
        return (0, 0, 0)
    if b.startswith("#"):
        return self._parse_color(b)
    raise HTTPException(400, "bg must be 'transparent', 'white', 'black', or a hex colour")
```

In the handler, after the mask is acquired:

```python
bg_rgb = self._check_bg(bg)
if bg_rgb is None:
    result = self._remove(image_obj, despill=despill, edge=edge,
                          detect=detect, decontaminate=decontaminate)
else:
    result = self._composite_on_bg(image_obj, bg_rgb, despill=despill,
                                   detect=detect, decontaminate=decontaminate)
```

Note `jpg` output already implies opacity; with `bg=transparent` and
`format=jpg` the existing behaviour (flatten on white) is unchanged.

## Tier note

`/remove` is in `FREE_TIER_ENDPOINTS`; `/replace-bg` is paid-only. Adding `bg`
to `/remove` therefore gives free-tier users solid-colour compositing they
previously had to pay for. That is a deliberate, small giveaway (it is one
composite op, no extra GPU) and it removes a confusing paywall on something
every competitor gives away. Flag to Troy: if that is unwanted, gate non-
`transparent` values behind `_require_pro`.

---

# Part 2 — Prepaid credits

## Why

Three problems solved at once:

1. **AI backgrounds cost $0.02-0.19 per call** against a $0.02 list price.
   Post-pay metering loses money on every call. Prepaid means the customer has
   already paid before any provider cost is incurred.
2. **Cash flow.** Money arrives up front instead of at invoice close.
3. **Breakage.** Competitor withoutbg expires credits at 30-360 days depending
   on pack size; unused credits are pure margin and are why they can advertise
   "50% off" at volume.

## Competitive anchor (verified 2026-08-13, withoutbg.com/pricing)

| Pack | Price | Per image |
| :--- | :--- | :--- |
| 100 | EUR 10 | 0.10 |
| 1,000 | EUR 70 | 0.07 |
| 10,000 | EUR 500 | 0.05 |

Free tier: 50 credits, expires in 30 days.

We are currently at $0.02/image metered, i.e. 2.5-5x cheaper. Credits should
not raise the effective price of plain background removal — the point is to
make expensive operations viable, not to reprice the core product.

## Model: one currency, weighted actions

A credit is an abstract unit, not an image. Different endpoints consume
different amounts, so a single balance covers cheap and expensive work.

| Action | Credits |
| :--- | :--- |
| `/remove`, `/remove-url`, `/mask`, `/smart-crop`, `/replace-bg`, `/sticker`, `/outline`, `/silhouette`, `/shadow`, `/compare`, `/collage`, `/studio-shot`, `/headshot` | 1 |
| `/upscale`, `/face-restore`, `/colorize`, `/inpaint` | 2 |
| `/psd` | 5 |
| `/replace-bg-ai` | 10 |
| `/video/remove` | 50 per output second |

Weights live in one dict in `main.py` so they are auditable and changeable in
one place. The existing `units` argument to `_end()` is NOT reused for this —
`units` is Stripe meter weight and must keep its current meaning.

## Packs

| Pack | Price | $/credit | Expiry |
| :--- | :--- | :--- | :--- |
| 100 | $5 | 0.050 | 90 days |
| 500 | $20 | 0.040 | 180 days |
| 2,000 | $60 | 0.030 | 365 days |
| 10,000 | $250 | 0.025 | 365 days |

At the 2,000 pack, `/replace-bg-ai` costs the customer $0.30 against a worst
case provider cost of $0.19 — positive margin even before the backdrop cache,
which makes batch runs of one prompt nearly free on our side.

Plain removal at $0.025-0.05/credit is more than the current $0.02 metered
rate, so credits must be OPTIONAL, not a replacement. See coexistence below.

## Coexistence with metered billing (important)

Existing paying customers are on Stripe metered subscriptions at $0.02 (and two
grandfathered accounts at $0.005). Migrating them to credits would be a price
rise and a breaking change. So:

- If `credits_balance > 0`, decrement credits and **skip the Stripe meter**.
- Else fall through to today's metered path unchanged.
- Endpoints with no metered equivalent (`/replace-bg-ai`) require credits and
  402 when the balance is short.

This means credits are additive: nobody's bill changes unless they buy a pack.

## Schema (Supabase)

```sql
alter table public.users
  add column credits_balance integer not null default 0;

create table public.credit_ledger (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid not null references public.users(id),
  delta        integer not null,          -- +purchase, -consumption, -expiry
  reason       text not null,             -- 'purchase' | 'usage' | 'expiry' | 'grant'
  endpoint     text,                      -- set when reason='usage'
  stripe_ref   text,                      -- checkout session / payment intent
  expires_at   timestamptz,               -- set on purchase rows
  created_at   timestamptz not null default now()
);
create index on public.credit_ledger (user_id, created_at desc);
```

`credits_balance` is the fast read for the hot path; `credit_ledger` is the
audit trail and the source of truth for reconciliation and expiry. Never mutate
balance without writing a ledger row.

## Consumption (backend)

Decrement AFTER the work succeeds, in `_end()`, alongside the existing usage
row — a failed request must not consume credits.

```python
cost = CREDIT_COSTS.get(endpoint, 1)
if ctx.get("credits_balance", 0) >= cost:
    self._consume_credits(ctx, endpoint, cost)   # ledger row + balance decrement
    skip_meter = True
```

Race conditions: use a single SQL statement that decrements and returns the new
value, not read-then-write. Postgres:

```sql
update public.users
   set credits_balance = credits_balance - $2
 where id = $1 and credits_balance >= $2
returning credits_balance;
```

Zero rows returned = insufficient balance.

## Expiry

Daily job (frontend Vercel cron already exists for spend alerts): find purchase
ledger rows past `expires_at` whose credits were not consumed, write a negative
`reason='expiry'` row, and reduce the balance. FIFO — oldest purchase expires
first, which is what customers expect and what makes the ledger reconcilable.

## Stripe

One-time prices, not subscriptions. Four `price_...` ids, one per pack.
Checkout Session in payment mode; on `checkout.session.completed` webhook,
credit the user. Idempotent on `stripe_ref` so a webhook retry cannot
double-credit.

## Open decisions for Troy

1. **Free-tier grant.** Competitors give 50 credits expiring in 30 days. We give
   10 images/month with no expiry. Keep, replace, or run both?
2. **Do credits ever apply to `/video/remove`?** 50/sec at the 2,000 pack is
   $1.50/sec vs the current $0.10/sec metered price — wildly higher. Either
   drop video from the credit table or reprice it. Currently the table's
   weakest row.
3. **Grandfathered accounts** ($0.005/image): credits would cost them 5-10x. The
   coexistence rule above protects them, but confirm they should never be
   nudged toward packs.
4. Whether `bg` non-transparent values stay free-tier (Part 1 tier note).

## Sequencing

`bg` param is independent and small — ship it whenever.
Credits should wait until the `/replace-bg-ai` experiment reports whether the
partner actually wants AI backgrounds. Building a billing system for a feature
nobody asked twice for is the wrong order.

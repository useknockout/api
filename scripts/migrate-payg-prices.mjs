#!/usr/bin/env node
/**
 * Migrate existing pay-as-you-go subscriptions from the old $0.005/image price
 * to the new $0.05/image price AT EACH SUB'S PERIOD BOUNDARY (never mid-cycle).
 *
 * It does this with a Stripe Subscription Schedule per sub:
 *   phase 0: current (old) price, ends at the sub's current_period_end
 *   phase 1: new price, ongoing (1 iteration then schedule releases, leaving
 *            the subscription on the new price permanently)
 * proration_behavior=none, so the current period's already-metered usage stays
 * billed at the OLD price. Only future periods bill $0.05.
 *
 * USAGE (Git Bash / PowerShell / cmd, Node 18+):
 *   # 1. Dry run (default) - reads subs, prints the plan, writes NOTHING:
 *   STRIPE_KEY=rk_live_xxx node scripts/migrate-payg-prices.mjs
 *
 *   # 2. Execute for real:
 *   STRIPE_KEY=rk_live_xxx RUN=1 node scripts/migrate-payg-prices.mjs
 *
 * KEY: use a restricted key (Developers -> API keys -> Restricted keys) with
 *      "Subscriptions: Write" + "Subscription schedules: Write". A full sk_live_
 *      also works but a restricted key is safer.
 */

const KEY = process.env.STRIPE_KEY;
const RUN = process.env.RUN === "1";
const NEW_PRICE = "price_1Tj6tlF69w1SVxJl39Bh0kqc"; // payg $0.05/image

// The existing payg subscriptions to migrate. (Troy's own account is excluded.)
const SUBS = [
  { id: "sub_1TcBw0F69w1SVxJljY5XxKxW", who: "tom@kravento.com" },
  { id: "sub_1TcVkdF69w1SVxJl2FK8OgPt", who: "ajextechtv@gmail.com" },
  { id: "sub_1TgdOgF69w1SVxJld8mFYnK6", who: "mugunghwa24601@gmail.com" },
  { id: "sub_1TXMGEF69w1SVxJlBiPcVeAN", who: "pedrobeatmaker@gmail.com" },
  { id: "sub_1TiiTKF69w1SVxJlPbz71pTM", who: "joshua.raboteau@gmail.com" },
];

if (!KEY) {
  console.error("ERROR: set STRIPE_KEY (rk_live_... with subscription + subscription schedule write).");
  process.exit(1);
}

const form = (obj) => new URLSearchParams(obj).toString();

async function api(path, method = "GET", body) {
  const res = await fetch("https://api.stripe.com/v1/" + path, {
    method,
    headers: {
      Authorization: "Bearer " + KEY,
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: body ? form(body) : undefined,
  });
  const json = await res.json();
  if (!res.ok) {
    throw new Error(`${res.status} ${json.error?.code || ""}: ${json.error?.message || JSON.stringify(json)}`);
  }
  return json;
}

const fmt = (unix) => new Date(unix * 1000).toISOString().slice(0, 10);

async function main() {
  console.log(RUN ? "=== EXECUTING (RUN=1) ===" : "=== DRY RUN (set RUN=1 to execute) ===");
  console.log("New price:", NEW_PRICE, "\n");

  for (const { id, who } of SUBS) {
    try {
      const sub = await api(`subscriptions/${id}`);
      if (sub.status !== "active") {
        console.log(`SKIP  ${who} (${id}) - status=${sub.status}`);
        continue;
      }
      const item = sub.items.data[0];
      const curPrice = item.price.id;
      const periodEnd = item.current_period_end ?? sub.current_period_end;
      const periodStart = item.current_period_start ?? sub.current_period_start;

      if (curPrice === NEW_PRICE) {
        console.log(`SKIP  ${who} - already on new price`);
        continue;
      }

      console.log(`PLAN  ${who} (${id})`);
      console.log(`        phase0 ${curPrice} ($0.005)  ->  ends ${fmt(periodEnd)}`);
      console.log(`        phase1 ${NEW_PRICE} ($0.05)   ->  from ${fmt(periodEnd)}`);

      if (!RUN) continue;

      // Reuse a schedule if a prior run already created one (a sub can only have
      // one), otherwise create one mirroring the current subscription.
      let schedId = typeof sub.schedule === "string" ? sub.schedule : sub.schedule?.id;
      if (!schedId) {
        const sched = await api("subscription_schedules", "POST", { from_subscription: id });
        schedId = sched.id;
      }

      // phase0 = old price until period end; phase1 = new price, OPEN-ENDED
      // (no iterations / no end_date) so the sub stays on $0.05 indefinitely.
      const updated = await api(`subscription_schedules/${schedId}`, "POST", {
        end_behavior: "release",
        proration_behavior: "none",
        "phases[0][items][0][price]": curPrice,
        "phases[0][start_date]": String(periodStart),
        "phases[0][end_date]": String(periodEnd),
        "phases[1][items][0][price]": NEW_PRICE,
      });

      const ph = updated.phases
        .map((p) => `${fmt(p.start_date)}->${p.end_date ? fmt(p.end_date) : "open"}:${p.items[0].price.slice(-8)}`)
        .join("  ");
      console.log(`        OK schedule=${updated.id} status=${updated.status}`);
      console.log(`        phases: ${ph}`);
    } catch (err) {
      console.error(`FAIL  ${who} (${id}): ${err.message}`);
    }
  }

  console.log("\nDone.", RUN ? "Schedules created." : "Dry run only - rerun with RUN=1 to apply.");
}

main();

/**
 * Tests for the money maths in economics.js.
 *
 * These exist because Josh spotted a building reading "$1,000 yearly, $12,000
 * lifetime, -$8,600 net loss" on 13 MWh/yr of generation, and nothing in the
 * codebase could check whether that was right. Running the same model on the
 * same inputs gives $2,407 / $55,083 / +$34,563 -- a factor of 2.4 that flips
 * the sign of the answer.
 *
 * The economics were untestable until they were pulled out of preview.html,
 * because every function was a closure inside a 4,000-line page. That is the
 * whole reason a 2.4x error could survive: no assertion could reach it.
 *
 * The most useful test here is the FLOOR one. Self-consumed electricity is
 * always worth more per kWh than exported electricity, so the annual value can
 * never fall below "everything exported at the buyback rate". That single
 * inequality catches the reported figure: 13,000 kWh x 14c = $1,820, and the
 * map said $1,000.
 *
 * Run:  node tests/test_economics.mjs
 */

import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const E = require("../economics.js");

let pass = 0;
const failures = [];
function check(name, fn) {
  try { fn(); console.log(`  pass  ${name}`); pass++; }
  catch (e) { console.log(`  FAIL  ${name}: ${e.message}`); failures.push(name); }
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }
function close(a, b, tol, msg) {
  if (Math.abs(a - b) > tol) throw new Error(`${msg}: ${a} vs ${b}`);
}

// A representative house: 11.4 kW, 13 MWh/yr, 234 m2 of roof. This is the
// building Josh was looking at (105 Arrowtown-Lake Hayes Road).
const HOUSE = [11.4, 13000, 234.4];

check("a house is not classified as a business", () => {
  const e = E.economicsFor(...HOUSE);
  assert(e.biz === false, "classified as business");
  assert(e.buy === 30 && e.sellNow === 14, `rates ${e.buy}/${e.sellNow}`);
});

check("annual value is never below everything-exported", () => {
  // THE ONE THAT CATCHES THE REPORTED BUG. Self-consumption is worth more per
  // kWh than export, so all-export is a hard floor on the annual figure.
  const e = E.economicsFor(...HOUSE);
  const floor = 13000 * (e.sellNow / 100);
  assert(e.annual >= floor,
         `annual $${e.annual.toFixed(0)} below the all-export floor $${floor.toFixed(0)}`);
});

check("annual value is exactly self x retail + exported x buyback", () => {
  const e = E.economicsFor(...HOUSE);
  const self = e.selfKwh, exported = 13000 - self;
  const expected = (self * e.buy + exported * e.sellNow) / 100;
  close(e.annual, expected, 1, "annual does not equal its own parts");
});

check("self-consumption is capped by the daytime load, not the roof", () => {
  const small = E.economicsFor(3, 4000, 100);
  const big = E.economicsFor(20, 26000, 300);
  assert(big.selfKwh <= big.ceiling + 1, "exceeded its ceiling");
  assert(big.selfKwh === big.ceiling, "a big array should hit the ceiling");
  assert(small.selfKwh < big.selfKwh || small.selfKwh === small.ceiling,
         "a small array should be generation-limited");
});

check("raising household consumption raises self-consumption", () => {
  // Josh, 1 Sep: changing 7,000 -> 10,000 kWh used to change nothing at all,
  // because the kW ceiling always bound first.
  const base = E.economicsFor(...HOUSE, { useKwh: 7000 });
  const more = E.economicsFor(...HOUSE, { useKwh: 10000 });
  assert(more.selfKwh > base.selfKwh,
         `self-consumption did not move: ${base.selfKwh} -> ${more.selfKwh}`);
  const ratio = more.selfKwh / base.selfKwh;
  close(ratio, 10000 / 7000, 0.02, "self-consumption did not scale proportionally");
});

check("more generation is never worth less money", () => {
  let prev = 0;
  for (const kwh of [2000, 5000, 9000, 13000, 20000, 30000]) {
    const e = E.economicsFor(11.4, kwh, 234.4);
    assert(e.annual >= prev, `annual fell from ${prev} to ${e.annual} at ${kwh} kWh`);
    prev = e.annual;
  }
});

check("lifetime is consistent with the annual figure", () => {
  // 30 years of a roughly flat real value, discounted, cannot be under 8x or
  // over 30x the first year. A 12x reported against a $1,000 first year was
  // plausible; the first year itself was what was wrong.
  const e = E.economicsFor(...HOUSE);
  const ratio = (e.lifetime + e.inverterCost) / e.annual;
  assert(ratio > 8 && ratio < 30, `lifetime/annual ratio ${ratio.toFixed(1)} implausible`);
});

check("payback under the system life implies a net gain", () => {
  // The contradiction on screen: a 5% yearly ROI reported alongside a lifetime
  // NET LOSS. If it pays back inside its life, the lifetime figure must exceed
  // the cost.
  const e = E.economicsFor(...HOUSE);
  if (e.payback !== null && e.payback < 30) {
    assert(e.lifetime > e.cost,
           `pays back in ${e.payback.toFixed(1)} yrs but lifetime $${e.lifetime.toFixed(0)}`
           + ` < cost $${e.cost.toFixed(0)}`);
  }
});

check("cost follows the size bands", () => {
  close(E.costPerKw(2), 3000, 0, "under 3 kW");
  close(E.costPerKw(5), 2000, 0, "3-8 kW");
  close(E.costPerKw(11.4), 1800, 0, "8-25 kW");
  close(E.costPerKw(30), 1500, 0, "25-50 kW");
  close(E.costPerKw(200), 1200, 0, "over 50 kW");
  // No gap: every size must price.
  for (let kw = 0.5; kw < 300; kw += 0.5) {
    assert(E.costPerKw(kw) > 0, `no rate at ${kw} kW`);
  }
});

check("a big roof or a big array is treated as a business", () => {
  assert(E.economicsFor(11.4, 13000, 500).biz === true, "500 m2 roof should be business");
  assert(E.economicsFor(120, 150000, 100).biz === true, "120 kW should be business");
  assert(E.economicsFor(11.4, 13000, 234).biz === false, "house misclassified");
});

check("the export rate declines and never goes negative", () => {
  const now = E.sellRate(new Date().getFullYear(), 14);
  const later = E.sellRate(new Date().getFullYear() + 30, 14);
  close(now, 14, 1e-9, "today's rate should be the starting rate");
  assert(later < now && later > 0, `30-year rate ${later}`);
  close(later, 14 * Math.pow(0.98, 30), 0.01, "decline schedule");
});

check("zero or negative inputs return nothing rather than nonsense", () => {
  assert(E.economicsFor(0, 13000, 200) === null, "0 kW should return null");
  assert(E.economicsFor(11.4, 0, 200) === null, "0 kWh should return null");
});

console.log(`\n${pass}/${pass + failures.length} passed`);
if (failures.length) {
  console.log("failed: " + failures.join(", "));
  process.exit(1);
}

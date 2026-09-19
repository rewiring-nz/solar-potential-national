// Solar economics: cost, self-consumption, savings, payback.
//
// Pulled out of preview.html on 1 Sep so the money maths is ONE implementation
// that both the map and a test runner can call. Josh: "We need to be able to
// check the economics calculations" -- which was impossible while every
// function was a closure inside a 4,000-line page, and is how a 2.4x error in
// the yearly figure survived long enough for him to spot it on the map.
//
// Loaded as a plain script by preview.html (so it shares the page's scope, no
// build step) and required by tests/test_economics.mjs under Node.
//
// Everything here is pure: numbers in, numbers out, no DOM and no map. That is
// deliberate -- the moment this reaches for the page state it stops being
// testable, which is the condition it was just rescued from.

const ECON_DEFAULTS = {
  cost_tiers: [
    { max_kw: 3,     rate: 3000, label: "under 3 kW" },
    { max_kw: 8,     rate: 2000, label: "3 – 8 kW" },
    { max_kw: 25,    rate: 1800, label: "8 – 25 kW" },
    { max_kw: 50,    rate: 1500, label: "25 – 50 kW" },
    { max_kw: Infinity, rate: 1200, label: "over 50 kW" },
  ],
  // Homes and businesses buy electricity at very different prices, so a
  // single retail rate would flatter one and penalise the other. Export is
  // the same for both today.
  home_buy_c: 30, home_sell_c: 14,
  biz_buy_c: 17,  biz_sell_c: 14,
  // Which rate set a building gets.
  //
  // ROOF AREA is the real signal and does the work. System size is not an
  // independent second signal at all -- it is roof area times the coverage
  // setting -- so using it as a threshold meant a house FLIPPED to business
  // pricing when the coverage slider went up, which is nonsense: a building
  // does not change what it is because we modelled more panels on it. That
  // is the bug Josh hit, "a home that is being treated as a business".
  //
  // The kW test is kept only as a backstop for the case roof area would miss
  // (a genuinely industrial system on a modest footprint), at Josh's 100kW.
  // NZ households are 3-12kW, and even a 300 m2 house fully covered lands
  // near 48kW, so nothing residential reaches it.
  //
  // Better signals exist if this proves too blunt: address_count is already
  // on every building, and a footprint carrying many addresses is a complex
  // rather than a house. LINZ outlines carry no building-use attribute.
  biz_min_roof_m2: 400,
  biz_min_kw: 100,
  // Export earns less over time. Anchors are Josh's; this interpolates
  // between them and holds flat past the last.
  // Export falls at a steady rate rather than via dated anchors (Josh:
  // "just starting at 14 cents and dropping by some percentage each year").
  //
  // 2%/yr from 14c: 12.9c in 2030, 11.7c in 2035, 10.5c in 2040, 8.6c in
  // 2050. Above the 12c/9c Josh first sketched, and deliberately so -- at 3%
  // a compound decline reaches 5.6c by the end of a 30-year system, and
  // unlike the old anchored schedule it never levels off. 2% keeps the tail
  // from doing more work than a buyback forecast can honestly carry.
  export_decline_pct: 2.0,
  // Self-consumption is a LOAD, not a share of output (Josh, 26 Aug). A
  // house drawing 1.2kW through the day soaks up the same ~8kWh whether the
  // array is 5kW or 40kW, so the kWh self-consumed is flat and the SHARE
  // falls as the system grows. The old fixed-percentage model did the
  // opposite -- every extra panel earned the retail price -- which is why
  // big roofs on small houses looked implausibly good.
  home_daytime_kw: 1.5,   // average daytime draw of a house (Josh's figure)
  biz_daytime_kw: 6.0,    // weekday-daytime business, ~49% of biz_use_kwh
  profile: "home_typ",
  // Hours a day the daytime load actually meets useful sun. 1.5kW x 6.7h is
  // ~10kWh/day. Not 12: the load only offsets generation while there is sun
  // on the roof, so the hours are daylight hours minus the shoulders.
  daytime_hours: 6.7,
  // Annual electricity use on site. This is not decoration: self-consumption
  // is capped by it. A house cannot use 30% of a modelled 48kW system's
  // output if that exceeds everything it burns in a year -- without the cap,
  // oversized systems on small buildings show savings nobody could realise.
  home_use_kwh: 7000,     // typical NZ household
  biz_use_kwh: 30000,     // placeholder, varies enormously -- editable
  life_years: 30,
  degradation_pct: 0.5, // per year
  // Retail electricity rises; the export rate does NOT (Josh). Buyback is
  // already on a declining schedule, and inflating it as well would have the
  // two assumptions fighting each other.
  elec_inflation_pct: 4.0,   // Josh
  // An inverter does not last the life of the panels. Standard practice
  // includes replacing it; leaving it out flatters every system equally.
  inverter_replace_year: 15,
  inverter_cost_pct: 18,
  // Money later is worth less than money now. Without this the model sums 30
  // years of nominal dollars and calls it savings, which overstates the
  // result badly -- and adding 5% inflation without it would overstate it
  // further still. Editable; set to 0 for an undiscounted view.
  // 3% (Josh). Lower than the ~5% a private householder would use, which is a
  // defensible public-good framing -- and it is on screen and adjustable, so
  // the choice is visible rather than buried.
  discount_rate_pct: 3.0,
};
// Named daytime-load profiles. The kW figure drives the maths; the names
// exist so the choice is a recognisable situation rather than a number
// pulled out of the air. A battery does not raise daytime demand, it moves
// evening demand into the daylight -- which is the same thing to this model,
// so it is carried as a larger effective daytime load.
const SELF_PROFILES = [
  { id: "home_out",   label: "Home — out on weekdays",       kw: 0.8 },
  { id: "home_typ",   label: "Home — typical mix",           kw: 1.5 },
  { id: "home_in",    label: "Home — someone there daytime", kw: 2.0 },
  { id: "home_batt",  label: "Home — with a battery",        kw: 3.5 },
  { id: "business",   label: "Business — weekday daytime",   kw: 6.0 },
  { id: "custom",     label: "Custom",                       kw: null },
];
const THIS_YEAR = new Date().getFullYear();
let econ = JSON.parse(JSON.stringify(ECON_DEFAULTS));
econ.cost_tiers[4].max_kw = Infinity;   // JSON round-trip turns Infinity into null

function costPerKw(kwp) {
  for (const t of econ.cost_tiers) if (kwp < t.max_kw) return t.rate;
  return econ.cost_tiers[econ.cost_tiers.length - 1].rate;
}
function systemCost(kwp) { return kwp * costPerKw(kwp); }

// First-year value of the energy: what is used on site displaces electricity
// at the retail price, what is exported earns the buyback rate. Lifetime
// applies linear panel degradation over the system life. Deliberately no
// discount rate or price inflation -- two assumptions that mostly cancel and
// that nobody can check, and their absence is easier to explain than a
// number picked to make payback look good.
// Business or home? Either signal is enough -- see biz_min_* above.
function isBusiness(kwp, roofM2) {
  return (roofM2 || 0) >= econ.biz_min_roof_m2 || (kwp || 0) >= econ.biz_min_kw;
}
// Export rate in a calendar year: today's rate declining at a steady
// percentage. Takes the starting rate as an argument rather than reading a
// schedule, so the home and business figures each decline from their own
// starting point without needing the rescaling the anchored version did.
function sellRate(year, startCents) {
  const yrs = Math.max(0, year - THIS_YEAR);
  return startCents * Math.pow(1 - econ.export_decline_pct / 100, yrs);
}

function economicsFor(kwp, kwhYear, roofM2, override) {
  if (!(kwp > 0) || !(kwhYear > 0)) return null;
  const biz = isBusiness(kwp, roofM2);
  const buy = biz ? econ.biz_buy_c : econ.home_buy_c;
  const sellNow = biz ? econ.biz_sell_c : econ.home_sell_c;
  // The decline schedule is anchored on today's rate, so editing the
  // home/business export price moves the whole curve with it instead of
  // snapping back to the schedule's own first value.
  // The old anchored schedule needed rescaling to today's edited rate; a
  // decline from the rate itself does not.
  const daytimeKw = (override && override.daytimeKw != null)
    ? override.daytimeKw : (biz ? econ.biz_daytime_kw : econ.home_daytime_kw);
  const useKwh = (override && override.useKwh != null)
    ? override.useKwh : (biz ? econ.biz_use_kwh : econ.home_use_kwh);
  const cost = systemCost(kwp);
  const d = econ.degradation_pct / 100;
  // The site can absorb at most daytime_kw for daytime_hours a day, so that
  // is the self-consumption ceiling however large the array gets. Annual use
  // is a second ceiling, for the case where someone enters a very low yearly
  // figure -- a site cannot self-consume more than it uses. Whichever binds
  // first, binds; generation past it is exported.
  // Josh, 1 Sep: "if you change the consumption of a home from 7000 to
  // 10,000 kWh, then the self consumption kW should go up higher, not stay
  // the same". He is right, and the old code could not do that: the ceiling
  // was min(daytime kW load, annual use), and for a typical home the kW side
  // always bound -- 1.5 kW x 6.7 h x 365 = 3,668 kWh against 7,000 used --
  // so editing annual use moved nothing at all until it dropped below 3,668.
  //
  // A house that uses more electricity uses more of it in daylight too, so
  // the daytime load scales with annual consumption rather than sitting
  // beside it as an independent number. The profile's kW is now read as
  // "this load AT THE DEFAULT annual use", and moves proportionally from
  // there. Both ceilings then respond together instead of one silently
  // dominating.
  const defaultUse = biz ? ECON_DEFAULTS.biz_use_kwh : ECON_DEFAULTS.home_use_kwh;
  const scaledDaytimeKw = daytimeKw * (useKwh / defaultUse);
  const loadCeiling = scaledDaytimeKw * econ.daytime_hours * 365;
  const ceiling = Math.min(loadCeiling, useKwh);
  const selfKwhFor = gen => Math.min(gen, ceiling);
  const valueInYear = y => {
    const gen = kwhYear * Math.pow(1 - d, y);
    const selfKwh = selfKwhFor(gen);
    return (selfKwh * buy + (gen - selfKwh) * sellRate(THIS_YEAR + y, sellNow)) / 100;
  };
  // Retail inflates, export does not, and everything is discounted back to
  // today. Applied in valueInYear's caller rather than inside it so payback
  // below uses the same figures.
  const infl = 1 + econ.elec_inflation_pct / 100;
  const disc = 1 + econ.discount_rate_pct / 100;
  const realValueInYear = y => {
    const gen = kwhYear * Math.pow(1 - d, y);
    const selfKwh = selfKwhFor(gen);
    const retail = selfKwh * buy * Math.pow(infl, y);
    const exported = (gen - selfKwh) * sellRate(THIS_YEAR + y, sellNow);
    return (retail + exported) / 100 / Math.pow(disc, y);
  };
  const annual = realValueInYear(0);
  let lifetime = 0;
  for (let y = 0; y < econ.life_years; y++) lifetime += realValueInYear(y);
  // The inverter replacement is a cost in a future year, so it is discounted
  // the same way and subtracted from the lifetime figure.
  const inverterCost = econ.inverter_replace_year < econ.life_years
    ? cost * (econ.inverter_cost_pct / 100) / Math.pow(disc, econ.inverter_replace_year)
    : 0;
  lifetime -= inverterCost;
  let cum = 0, payback = null;
  for (let y = 0; y < 60; y++) {
    const v = realValueInYear(y);
    cum += v;
    if (cum >= cost) { payback = y + 1 - (cum - cost) / v; break; }
  }
  // Lifetime cents earned by ONE kWh/yr of extra generation, if all of it is
  // exported. This is the marginal panel's world: self-consumption is capped
  // by the daytime load and already met by the panels that came before it,
  // so an added panel earns the buyback rate, not the retail one.
  let exportLifetimePerKwhYr = 0, retailLifetimePerKwhYr = 0;
  for (let y = 0; y < econ.life_years; y++) {
    const deg = Math.pow(1 - d, y), dsc = Math.pow(disc, y);
    exportLifetimePerKwhYr += deg * sellRate(THIS_YEAR + y, sellNow) / 100 / dsc;
    retailLifetimePerKwhYr += deg * buy * Math.pow(infl, y) / 100 / dsc;
  }
  return { cost, annual, lifetime, payback, rate: costPerKw(kwp), biz, buy, sellNow,
           exportLifetimePerKwhYr, retailLifetimePerKwhYr,
           sellLater: sellRate(THIS_YEAR + econ.life_years, sellNow),
           inverterCost,
           daytimeKw: scaledDaytimeKw, ceiling, useKwh, selfKwh: selfKwhFor(kwhYear),
           // Share is now an OUTPUT of the model, not an input to it.
           selfPct: 100 * selfKwhFor(kwhYear) / kwhYear,
           capped: kwhYear > ceiling, capByUse: loadCeiling > useKwh };
}
// Node (tests) and browser (preview.html) both, without a build step.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { ECON_DEFAULTS, SELF_PROFILES, costPerKw, systemCost,
                     isBusiness, sellRate, economicsFor, setEcon: e => { econ = e; },
                     getEcon: () => econ };
}

// ============================================================ HOURLY ENGINE
//
// WHY AN HOUR IS THE UNIT NOW. Everything above works on annual kWh with
// self-consumption capped by an average daytime load. That is enough to
// answer "is solar worth it", and it cannot answer either of the two
// questions Josh asked for on 19 Sep:
//
//   * a RETAIL PLAN only differs from another if it prices electricity
//     differently at different times -- day/night, peak/off-peak, a free
//     hour. Annually they are all just "a number of cents".
//   * a BATTERY is nothing but a time-shifting device. It moves midday
//     surplus into the evening. At annual resolution it does not exist;
//     the old model faked one as a bigger daytime load, which credits it
//     for energy it never stored.
//
// The resolution is a REPRESENTATIVE DAY PER SEASON -- four days of 24
// hours, weighted by the days in each season -- not 8,760 hours. Two
// reasons. The generation side is exactly what data/seasonal_curves.json
// already holds, so the money and the chart on screen can never disagree
// about what the roof makes. And an 8,760-hour run would need a genuine
// half-hourly load trace, which we do not have; pretending to that
// resolution with a repeated daily shape would be precision theatre.
//
// Everything here stays pure: numbers in, numbers out.

// Household and business demand through the day, as a share of the day's
// total. Shapes, not magnitudes: the annual use figure sets the scale.
// The home shape is the familiar double peak (breakfast, then a larger
// evening one after work) with a midday trough -- which is exactly why
// self-consumption is limited and why a battery has something to do.
const LOAD_SHAPES = {
  home: [
    0.020, 0.018, 0.017, 0.017, 0.018, 0.024, 0.038, 0.055,   // 0-7
    0.052, 0.042, 0.036, 0.034, 0.034, 0.033, 0.033, 0.036,   // 8-15
    0.045, 0.062, 0.085, 0.082, 0.068, 0.054, 0.040, 0.027,   // 16-23
  ],
  business: [
    0.012, 0.011, 0.011, 0.011, 0.012, 0.016, 0.026, 0.042,   // 0-7
    0.062, 0.073, 0.077, 0.078, 0.074, 0.076, 0.075, 0.070,   // 8-15
    0.058, 0.044, 0.030, 0.022, 0.018, 0.015, 0.014, 0.013,   // 16-23
  ],
};

// A plan prices each hour. `periods` maps an hour (0-23) to a rate in
// cents; `sell_c` is that plan's buyback. These are ILLUSTRATIVE SHAPES OF
// REAL PLAN TYPES sold in New Zealand, not offers from any retailer, and
// they are all editable -- which is why none of them carries a company
// name. The point is to let someone see how much the plan structure
// matters, which for a solar household is a lot: a plan that pays little
// for export but charges little at night suits a battery, and the reverse
// suits a bare array.
function flatRates(c) { return new Array(24).fill(c); }
function touRates(spec, base) {
  const r = new Array(24).fill(base);
  for (const [from, to, c] of spec) {
    for (let h = from; h !== to; h = (h + 1) % 24) r[h] = c;
  }
  return r;
}
const RETAIL_PLANS = [
  {
    id: "flat", label: "Flat rate", biz: false,
    note: "One price all day. The simplest plan and the default here.",
    buy: () => flatRates(30), sell_c: 14, daily_c: 250,
  },
  {
    id: "day_night", label: "Day / night", biz: false,
    note: "Cheaper overnight, dearer through the day. Suits a battery that " +
          "can fill up on cheap night power; suits bare solar less.",
    buy: () => touRates([[23, 7, 18]], 33), sell_c: 12, daily_c: 250,
  },
  {
    id: "peak_offpeak", label: "Peak / off-peak / night", biz: false,
    note: "Morning and evening peaks priced highest. Solar rarely covers " +
          "either peak on its own, so this is where a battery earns most.",
    buy: () => touRates([[23, 7, 17], [7, 9, 45], [17, 21, 45]], 28),
    sell_c: 12, daily_c: 250,
  },
  {
    id: "high_buyback", label: "High buyback, dearer power", biz: false,
    note: "Pays well for export and charges more for what you draw. Suits " +
          "a big array on a house that is empty during the day.",
    buy: () => flatRates(35), sell_c: 20, daily_c: 250,
  },
  {
    id: "free_evening", label: "One free hour a day", biz: false,
    note: "A free hour in the evening, paid for with a higher rate the rest " +
          "of the time. Worth checking against your own usage.",
    buy: () => touRates([[21, 22, 0]], 33), sell_c: 12, daily_c: 250,
  },
  {
    id: "biz_flat", label: "Business — flat rate", biz: true,
    note: "Commercial rates are lower per kWh and the load is daytime, " +
          "which is when the roof generates.",
    buy: () => flatRates(17), sell_c: 14, daily_c: 800,
  },
  {
    id: "biz_tou", label: "Business — time of use", biz: true,
    note: "Daytime commercial rate with a cheaper overnight block.",
    buy: () => touRates([[22, 7, 12]], 19), sell_c: 14, daily_c: 800,
  },
];

// A battery is defined by what it can hold, how fast it moves energy, and
// what it loses doing so. `reserve_pct` is the floor a real installation
// keeps for backup and cycle life -- ignoring it would credit the battery
// with capacity no installer lets you use.
const BATTERY_DEFAULTS = {
  enabled: false,
  kwh: 10,            // usable-before-reserve capacity
  kw: 5,              // charge/discharge power limit
  round_trip_pct: 90, // in-and-out efficiency
  reserve_pct: 10,    // never discharged below this
  cost_per_kwh: 1000, // installed, NZ, indicative
  life_years: 15,     // replaced once inside a 30-year system life
};

// One representative day. Generation and load are 24 kW values; the battery
// charges only from surplus (never from the grid, which is a separate
// product decision and a different plan calculation) and discharges only
// into a deficit.
//
// Returns the day's kWh split three ways, and the hour each one happened in,
// because a time-of-use plan prices them differently.
function simulateDay(genKw, loadKw, batt, socStart) {
  const cap = batt && batt.enabled ? Math.max(0, batt.kwh) : 0;
  const floor = cap * (batt ? batt.reserve_pct : 0) / 100;
  const pmax = batt ? Math.max(0, batt.kw) : 0;
  // Round-trip loss is charged on the way in, so a kWh that comes back out
  // is a kWh the house actually uses.
  const eff = Math.sqrt(Math.max(0.01, (batt ? batt.round_trip_pct : 100) / 100));
  let soc = Math.min(Math.max(socStart, floor), cap);
  const selfH = new Array(24).fill(0);
  const expH = new Array(24).fill(0);
  const impH = new Array(24).fill(0);
  for (let h = 0; h < 24; h++) {
    const g = genKw[h], l = loadKw[h];
    const direct = Math.min(g, l);
    selfH[h] += direct;
    let surplus = g - direct;
    let deficit = l - direct;
    if (cap > 0 && surplus > 0) {
      const room = (cap - soc) / eff;              // grid-side kWh to fill
      const take = Math.min(surplus, pmax, room);
      soc += take * eff;
      surplus -= take;
    }
    if (cap > 0 && deficit > 0) {
      const avail = (soc - floor) * eff;           // house-side kWh available
      const give = Math.min(deficit, pmax, Math.max(0, avail));
      soc -= give / eff;
      selfH[h] += give;                            // stored sun, used later
      deficit -= give;
    }
    expH[h] = surplus;
    impH[h] = deficit;
  }
  return { selfH, expH, impH, socEnd: soc };
}

// A year of representative days. `genByHour[s][h]` is kW for season s, and
// `seasonDays[s]` how many days that season stands for.
//
// The battery's state of charge is carried across two passes of each season
// so a day starts where the previous one ended rather than empty -- an
// empty start every morning would understate a battery that habitually
// holds charge overnight.
function simulateYear(genByHour, seasonDays, dailyKwh, shape, batt) {
  const out = { selfKwh: 0, exportKwh: 0, importKwh: 0,
                selfByHour: new Array(24).fill(0),
                exportByHour: new Array(24).fill(0),
                importByHour: new Array(24).fill(0) };
  const loadKw = shape.map(f => f * dailyKwh);     // kWh in an hour == kW
  let soc = 0;
  for (let pass = 0; pass < 2; pass++) {
    for (let s = 0; s < genByHour.length; s++) {
      const d = simulateDay(genByHour[s], loadKw, batt, soc);
      soc = d.socEnd;
      if (pass === 0) continue;                    // first pass only warms soc
      const days = seasonDays[s];
      for (let h = 0; h < 24; h++) {
        out.selfByHour[h] += d.selfH[h] * days;
        out.exportByHour[h] += d.expH[h] * days;
        out.importByHour[h] += d.impH[h] * days;
      }
    }
  }
  for (let h = 0; h < 24; h++) {
    out.selfKwh += out.selfByHour[h];
    out.exportKwh += out.exportByHour[h];
    out.importKwh += out.importByHour[h];
  }
  return out;
}

// The money, hour by hour. Same shape of answer as economicsFor above --
// cost, annual, lifetime, payback -- so the page can swap one for the other,
// but every kWh is now priced at the hour it happened in and a battery is
// simulated rather than approximated.
//
// genByHour[s][h] is kW per hour for a representative day in season s, at
// TODAY's output. Degradation, retail inflation and the export decline are
// applied per year exactly as in the annual model, and the year is
// re-simulated each time: as generation fades the battery fills less, which
// a single scaling factor could not express.
function economicsHourlyFor(kwp, genByHour, seasonDays, roofM2, override) {
  if (!(kwp > 0) || !genByHour || !genByHour.length) return null;
  const o = override || {};
  const biz = (o.biz != null) ? o.biz : isBusiness(kwp, roofM2);
  const plan = o.plan || RETAIL_PLANS.find(p => !!p.biz === !!biz) || RETAIL_PLANS[0];
  const buyRates = o.buyRates || plan.buy();
  const sellStart = (o.sell_c != null) ? o.sell_c : plan.sell_c;
  const batt = Object.assign({}, BATTERY_DEFAULTS, o.battery || {});
  const useKwh = (o.useKwh != null) ? o.useKwh
    : (biz ? econ.biz_use_kwh : econ.home_use_kwh);
  const shape = o.shape || (biz ? LOAD_SHAPES.business : LOAD_SHAPES.home);
  const days = seasonDays || new Array(genByHour.length)
    .fill(365 / genByHour.length);
  const dailyKwh = useKwh / 365;

  const battCost = batt.enabled ? batt.kwh * batt.cost_per_kwh : 0;
  const cost = systemCost(kwp) + battCost;
  const d = econ.degradation_pct / 100;
  const infl = 1 + econ.elec_inflation_pct / 100;
  const disc = 1 + econ.discount_rate_pct / 100;

  const yearOf = y => {
    const deg = Math.pow(1 - d, y);
    const gen = genByHour.map(row => row.map(v => v * deg));
    const sim = simulateYear(gen, days, dailyKwh, shape, batt);
    let retail = 0;
    for (let h = 0; h < 24; h++) retail += sim.selfByHour[h] * buyRates[h];
    const exported = sim.exportKwh * sellRate(THIS_YEAR + y, sellStart);
    return { sim, value: (retail * Math.pow(infl, y) + exported) / 100 };
  };

  const y0 = yearOf(0);
  const annual = y0.value / Math.pow(disc, 0);
  let lifetime = 0;
  for (let y = 0; y < econ.life_years; y++) {
    lifetime += yearOf(y).value / Math.pow(disc, y);
  }
  const inverterCost = econ.inverter_replace_year < econ.life_years
    ? systemCost(kwp) * (econ.inverter_cost_pct / 100)
      / Math.pow(disc, econ.inverter_replace_year)
    : 0;
  // A battery does not last a 30-year system either, and leaving its
  // replacement out is the single easiest way to make one look good.
  const battReplace = (batt.enabled && batt.life_years < econ.life_years)
    ? battCost / Math.pow(disc, batt.life_years) : 0;
  lifetime -= inverterCost + battReplace;

  let cum = 0, payback = null;
  for (let y = 0; y < 60; y++) {
    const v = yearOf(Math.min(y, econ.life_years - 1)).value / Math.pow(disc, y);
    cum += v;
    if (cum >= cost) { payback = y + 1 - (cum - cost) / v; break; }
  }
  const genKwh = genByHour.reduce(
    (t, row, s) => t + row.reduce((a, b) => a + b, 0) * days[s], 0);
  return {
    cost, annual, lifetime, payback, biz, plan, batteryCost: battCost,
    inverterCost, batteryReplaceCost: battReplace,
    rate: costPerKw(kwp), sellNow: sellStart,
    genKwh, useKwh,
    selfKwh: y0.sim.selfKwh, exportKwh: y0.sim.exportKwh,
    importKwh: y0.sim.importKwh,
    selfPct: genKwh > 0 ? 100 * y0.sim.selfKwh / genKwh : 0,
    selfByHour: y0.sim.selfByHour, exportByHour: y0.sim.exportByHour,
    importByHour: y0.sim.importByHour,
    buyRates, hourly: true,
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = Object.assign(module.exports || {}, {
    LOAD_SHAPES, RETAIL_PLANS, BATTERY_DEFAULTS,
    simulateDay, simulateYear, economicsHourlyFor,
  });
}

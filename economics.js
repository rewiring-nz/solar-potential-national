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

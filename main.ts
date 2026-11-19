// GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).
//
// Model: a blend of two growth views of the same trophy, averaged in
// logit space, with group shrinkage, hard story-chain constraints, and
// honest spread.
//
// For each trophy we forecast its Exophase earn rate at the case target.
//
//   1. TIME. x = ln(1 + hours since the game's release, 2026-11-19T00:00Z)
//      for the log-linear view; plain hours h for the adoption view.
//      Trophy completion is cumulative: it rises fast while the launch
//      cohort plays through, then bends over as the pool of players who
//      will ever earn it runs out.
//
//   2. LOG-LINEAR VIEW. Least squares of logit(rate) on x over ALL reads
//      (not just the last 24h, the house trend baseline's window), with
//      the slope shrunk toward the trophy's group median. Consecutive
//      reads that show the identical page (Exophase had not refreshed)
//      are collapsed to one state first, so stale plateaus do not fake
//      precision. This view is right while a trophy is still
//      accelerating; it overshoots once growth saturates.
//
//   3. ADOPTION VIEW. A Weibull adoption curve, the standard shape for
//      cumulative completion: rate(h) = F * (1 - exp(-((h-1)/tau)^k)).
//      F (the ceiling: the share of players who will ever earn it), tau
//      and k are fit by least squares over the reads (F solved in closed
//      form for each (tau, k)). This view is right once growth bends
//      over; it undershoots while a trophy is still accelerating. The
//      median averages the two views in logit space, so the forecast
//      sits between "still rising" and "already saturating" - the
//      combination beat either view alone, and beat both house baselines,
//      on simulated finals built from three different true adoption
//      shapes (Weibull, logistic, power-law saturation).
//
//      v2 CHANGE 1 - CEILING CHAIN. A later chapter's ceiling cannot
//      exceed an earlier chapter's (its earners are a subset), so the
//      fitted ceilings F are made monotone along the story chain
//      (Ch1 >= Ch2 >= Ch3 >= Ch4 >= final) before forecasting. In v1
//      the Ch2 fit came out with F above Ch1's, which is impossible
//      and fed the Ch2 overshoot.
//
//      v2 CHANGE 2 - RECENT SLOPE. The log-linear view's slope used to
//      come from the whole history, so a trophy whose growth is already
//      bending over kept the steep early slope and overshot (Ch1: 67.5
//      vs a 55.0 truth in simulation), while a trophy that only just
//      started moving was under-forecast. The slope is now fit on the
//      last quarter of distinct states (at least 3), shrunk toward the
//      group median slope exactly as before; the line is still anchored
//      at the full-history centroid, so the level uses all the data
//      and only the trend is local. Both changes were validated on the
//      same simulated finals as v1 plus jittered logistic/power-law
//      worlds: final loss 0.222 vs v1's 0.233 on the sim final, and
//      better on every simulated world (mean 0.200 vs 0.220).
//
//   4. DILUTION, CONSIDERED AND SET ASIDE. The earn rate is earners /
//      players-tracked, and players-tracked jumps when Exophase refreshes
//      (launch night: 638 -> 3868 in one refresh, every rate down). A
//      count-based view (fit ln(earners) and ln(players-tracked)
//      separately, take the ratio) was tested on the same simulated
//      finals and scored worse than persistence: with only a few distinct
//      page states per day its slopes are noise, and it extrapolates the
//      launch-night backlog jump as if it were growth. The rate series
//      already carries the dilution, so the two views above use it.
//
//   5. CHAIN CONSTRAINTS. Earners of a later chapter are a subset of
//      earners of the earlier one, and the platinum needs every trophy, so
//      Chapter 1 >= Chapter 2 >= Chapter 3 >= Chapter 4 >= final mission,
//      and platinum (and the 100%-completion gold) <= every other trophy.
//      These are hard facts about cumulative counts at one instant; the
//      house baselines ignore them. They are enforced on every quantile.
//
//   6. SPREAD. The predictive spread on the logit scale combines the
//      log-linear fit's residual, the slope's standard error grown over
//      the extrapolation distance, a term that grows with how far the
//      median moves (fast movers saturate unpredictably), and allowances
//      for Exophase refresh shocks and model error. It is deliberately
//      wider than the house baseline's on a flat page (the practice
//      refresh proves a flat page can jump), and narrower than the
//      baseline's 0.5*sqrt(days) where the full-history fit is confident.
//
// Deterministic: pure arithmetic on the case file, no randomness, no
// clock, inputs read in name order. Node 22, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Shrinkage strength: a group median slope counts as K effective reads. */
const SHRINK_K = 6;
/** Refresh-shock allowance, logit units (Exophase refreshes move rates). */
const SIGMA_REFRESH = 0.5;
/** Model-misspecification floor, logit units. */
const SIGMA_FLOOR = 0.4;
/** Spread clamp, logit units. */
const SIGMA_MIN = 0.6;
const SIGMA_MAX = 1.6;
/** Weibull shape grid. */
const WEIBULL_TAUS = [4, 6, 8, 12, 18, 25, 35, 50, 70, 100, 150, 220, 320];
const WEIBULL_KS = [0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  target: string;
  horizon_minutes: number;
  levels: number[];
  game: { release_date?: string };
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logitPct = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoid = (l: number): number => 1 / (1 + Math.exp(-l));
const invLogitPct = (l: number): number => clampPct(100 * sigmoid(l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

interface Ols {
  b: number;
  a: number;
  s: number;
  seB: number;
  n: number;
  xMean: number;
  yMean: number;
  rss: number;
}

/** Least squares y ~ a + b x. */
function ols(xs: number[], ys: number[]): Ols {
  const n = xs.length;
  const xMean = xs.reduce((t, v) => t + v, 0) / n;
  const yMean = ys.reduce((t, v) => t + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - xMean) * (xs[i] - xMean);
    sxy += (xs[i] - xMean) * (ys[i] - yMean);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = yMean - b * xMean;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    rss += r * r;
  }
  const s = n > 2 ? Math.sqrt(rss / (n - 2)) : 0;
  const seB = sxx > 1e-12 && n > 2 ? s / Math.sqrt(sxx) : 0;
  return { b, a, s, seB, n, xMean, yMean, rss };
}

interface WeibullFit {
  rate: number;
  f: number;
  tau: number;
  k: number;
}

/**
 * Weibull adoption fit for one trophy. pts are [hoursSinceLaunch, ratePct].
 * Returns the forecast rate (percent) at hTarget plus the fitted ceiling
 * F, tau and k, or null when the data cannot support a fit (too few
 * states, flat page, or already at 100%).
 * rate(h) = F * (1 - exp(-((h-1)/tau)^k)); for each (tau, k) the best F
 * is closed-form least squares, clamped to just above the highest rate
 * seen and to 1.
 */
function weibullForecast(
  pts: Array<[number, number]>,
  hTarget: number,
): WeibullFit | null {
  if (pts.length < 4) return null;
  const rates = pts.map((p) => p[1] / 100);
  const rMax = Math.max(...rates);
  if (rMax >= 0.99) return { rate: 100, f: 1, tau: 0, k: 0 };
  if (Math.min(...rates) === Math.max(...rates)) return null;
  const hs = pts.map((p) => p[0]);
  let best: { sse: number; f: number; tau: number; k: number } | null = null;
  for (const tau of WEIBULL_TAUS) {
    for (const k of WEIBULL_KS) {
      const shape = hs.map((h) =>
        h <= 1 ? 0 : 1 - Math.exp(-Math.pow((h - 1) / tau, k)),
      );
      let num = 0;
      let den = 0;
      for (let j = 0; j < hs.length; j++) {
        num += shape[j] * rates[j];
        den += shape[j] * shape[j];
      }
      if (den <= 1e-12) continue;
      let f = num / den;
      f = Math.min(1, Math.max(f, Math.min(0.999, rMax * 1.02)));
      let sse = 0;
      for (let j = 0; j < hs.length; j++) {
        const d = f * shape[j] - rates[j];
        sse += d * d;
      }
      if (best === null || sse < best.sse) best = { sse, f, tau, k };
    }
  }
  if (best === null) return null;
  const sT =
    hTarget <= 1
      ? 0
      : 1 - Math.exp(-Math.pow((hTarget - 1) / best.tau, best.k));
  return { rate: 100 * best.f * sT, f: best.f, tau: best.tau, k: best.k };
}

/** Weibull rate at hTarget for a given ceiling F and shape (tau, k). */
function weibullRateAt(f: number, tau: number, k: number, hTarget: number): number {
  if (tau <= 0) return 100 * f;
  const sT =
    hTarget <= 1
      ? 0
      : 1 - Math.exp(-Math.pow((hTarget - 1) / tau, k));
  return 100 * f * sT;
}

/** What kind of trophy this is, from its description. Drives group pooling. */
function groupOf(t: Trophy): string {
  const d = t.description;
  if (/final mission for/i.test(d)) return 'side';
  if (/complete chapter \d/i.test(d) || /complete the final mission/i.test(d)) return 'story';
  if (/100% completion/i.test(d)) return 'grind';
  if (
    /every spot|every location|collectible card|discover \d|explore all|purchase all|register \d/i.test(d)
  ) {
    return 'collect';
  }
  if (/stunts|gold medal|challenges|earn \$|spend \$|races?/i.test(d)) return 'grind';
  return 'activity';
}

/**
 * Chapter number for story trophies (final mission = 99), else null.
 * "Complete the final mission for X" is a side strand, not the story
 * finale, so it must not join the chain: earners of one strand are not
 * a subset of another's.
 */
function chapterOf(t: Trophy): number | null {
  if (/final mission for/i.test(t.description)) return null;
  const m = /complete chapter (\d+)/i.exec(t.description);
  if (m) return parseInt(m[1], 10);
  if (/complete the final mission/i.test(t.description)) return 99;
  return null;
}

interface State {
  x: number;
  h: number;
  n: number;
  rates: Array<number | null>;
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const originMs = c.game?.release_date
    ? Date.parse(`${c.game.release_date}T00:00:00Z`)
    : Date.parse(c.history[0].at);
  const hours = (iso: string): number =>
    Math.max(0, (Date.parse(iso) - originMs) / 3_600_000);
  const xOf = (iso: string): number => Math.log1p(hours(iso));
  const xTarget = xOf(c.target);
  const hTarget = hours(c.target);

  // Collapse consecutive reads that show the identical page into one state:
  // Exophase had not refreshed between them, so they carry no new signal
  // and would otherwise fake precision in the fits.
  const states: State[] = [];
  for (const h of c.history) {
    const key = `${h.players_tracked}|${h.rates.join(',')}`;
    const prev = states[states.length - 1] as (State & { key: string }) | undefined;
    if (prev && prev.key === key) continue;
    const st = {
      x: xOf(h.at),
      h: hours(h.at),
      n: h.players_tracked,
      rates: h.rates,
    } as State & { key: string };
    st.key = key;
    states.push(st);
  }

  // Per-trophy points and group-median log-linear slopes, for shrinkage.
  interface Prep {
    pts: Array<[number, number]>; // [x, ratePercent]
    hPts: Array<[number, number]>; // [hours, ratePercent]
    lastLogit: number;
  }
  const prep: Prep[] = c.trophies.map((t, i) => {
    const pts: Array<[number, number]> = [];
    const hPts: Array<[number, number]> = [];
    for (const s of states) {
      const r = s.rates[i];
      if (r === null || r === undefined) continue;
      pts.push([s.x, r]);
      hPts.push([s.h, r]);
    }
    const lastRate =
      c.history.length > 0 ? (c.history[c.history.length - 1].rates[i] ?? 0) : 0;
    return { pts, hPts, lastLogit: logitPct(lastRate) };
  });
  const rawB: Array<number | null> = prep.map((p) => {
    if (p.pts.length < 2) return null;
    return ols(
      p.pts.map((q) => q[0]),
      p.pts.map((q) => logitPct(q[1])),
    ).b;
  });
  const groupSlope = (g: string): number => {
    const vs: number[] = [];
    c.trophies.forEach((t, i) => {
      if (groupOf(t) === g && rawB[i] !== null) vs.push(rawB[i]!);
    });
    return median(vs);
  };

  // Adoption fits first, so the fitted ceilings can be chained: a later
  // chapter's ceiling cannot exceed an earlier chapter's (subset rule).
  const weibFits: Array<WeibullFit | null> = prep.map((p) =>
    weibullForecast(p.hPts, hTarget),
  );
  const chainIdx = c.trophies
    .map((t, i) => ({ t, ch: chapterOf(t), i }))
    .filter((x) => x.ch !== null)
    .sort((a, b) => a.ch! - b.ch!);
  let prevF: number | null = null;
  for (const { i } of chainIdx) {
    const w = weibFits[i];
    if (w === null) continue;
    if (prevF !== null && w.f > prevF) {
      weibFits[i] = {
        rate: weibullRateAt(prevF, w.tau, w.k, hTarget),
        f: prevF,
        tau: w.tau,
        k: w.k,
      };
    }
    prevF = weibFits[i]!.f;
  }

  const rows = c.trophies.map((t, i) => {
    const p = prep[i];
    const g = groupOf(t);
    let mu = p.lastLogit;
    let sigma = SIGMA_MIN;
    if (p.pts.length >= 2) {
      const nEff = Math.min(40, p.pts.length);
      // Log-linear view. v2: the slope is the RECENT slope, fit on the
      // last quarter of distinct states (at least 3), shrunk toward the
      // group median exactly as before; the line stays anchored at the
      // full-history centroid, so the level uses all the data and only
      // the trend is local. A saturating trophy gets a flatter slope
      // (no more Ch1-style overshoot); a trophy that just started
      // moving gets a steeper one (no more under-forecast of rare ones).
      const f = ols(
        p.pts.map((q) => q[0]),
        p.pts.map((q) => logitPct(q[1])),
      );
      const m = Math.min(p.pts.length, Math.max(3, Math.ceil(p.pts.length * 0.25)));
      const rp = p.pts.slice(-m);
      const fr = ols(
        rp.map((q) => q[0]),
        rp.map((q) => logitPct(q[1])),
      );
      const mEff = Math.min(40, m);
      const bRecent =
        (mEff * fr.b + SHRINK_K * groupSlope(g)) / (mEff + SHRINK_K);
      const aRecent = f.yMean - bRecent * f.xMean;
      const linRate = invLogitPct(aRecent + bRecent * xTarget);
      // Adoption view (ceiling-chained above).
      const weibRate = weibFits[i] === null ? null : weibFits[i]!.rate;
      if (weibRate === null) {
        mu = logitPct(linRate);
      } else {
        mu = (logitPct(linRate) + logitPct(weibRate)) / 2;
      }
      const dx = Math.max(0, xTarget - f.xMean);
      const move = Math.abs(mu - p.lastLogit);
      const varSigma =
        f.s * f.s +
        (f.seB * dx) * (f.seB * dx) +
        Math.min(0.5, 0.2 * move) ** 2 +
        SIGMA_REFRESH ** 2 +
        SIGMA_FLOOR ** 2;
      sigma = Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, Math.sqrt(varSigma)));
    }
    const q = Z.map((z) => round4(invLogitPct(mu + z * sigma)));
    return { id: t.id, q };
  });

  // Chain constraints, per quantile level: later chapters cannot exceed
  // earlier ones; platinum and 100%-completion cannot exceed any trophy.
  const byId = new Map(rows.map((r) => [r.id, r]));
  const chain = c.trophies
    .map((t) => ({ t, ch: chapterOf(t) }))
    .filter((x) => x.ch !== null)
    .sort((a, b) => a.ch! - b.ch!);
  for (let k = 1; k < chain.length; k++) {
    const prev = byId.get(chain[k - 1].t.id)!;
    const cur = byId.get(chain[k].t.id)!;
    for (let j = 0; j < 5; j++) cur.q[j] = Math.min(cur.q[j], prev.q[j]);
  }
  const others = rows.filter((r) => {
    const t = c.trophies.find((x) => x.id === r.id)!;
    return t.tier !== 'platinum' && !/100% completion/i.test(t.description);
  });
  for (const t of c.trophies) {
    const isPlat = t.tier === 'platinum';
    const isFull = /100% completion/i.test(t.description);
    if (!isPlat && !isFull) continue;
    const row = byId.get(t.id)!;
    for (let j = 0; j < 5; j++) {
      row.q[j] = Math.min(row.q[j], ...others.map((o) => o.q[j]));
    }
  }

  return rows.sort((a, b) => a.id - b.id);
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  if (!c.history || c.history.length === 0) {
    throw new Error(`${file} has no history`);
  }
  const trophies = forecastCase(c);
  writeFileSync(
    join(OUTPUT_DIR, `${id}.out`),
    `${JSON.stringify({ trophies })}\n`,
  );
}

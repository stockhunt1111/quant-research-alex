// What the Research page derives from the rows and the meta: a list's size in words, a single asset's verdict, the
// robustness count's order and colour, the slots of the chart's lines.
import type { AssetInfo, AssetRow, ListInfo, ListRow, Meta, Robustness, Row, Targets } from "../api/types";
import { pct, pct0 } from "../format";

export const TFS = ["1h", "4h", "1d"] as const;
// a line's colour and stroke, on the chart and beside its row: the palette's eight colours, then the same eight dashed,
// so that no two lines look alike (the server's curves endpoint takes as many keys)
export type Slot = { color: string; dash?: string };
export const SLOTS: Slot[] = Array.from({ length: 16 }, (_, i) =>
  (i < 8 ? { color: `--s${i + 1}` } : { color: `--s${i - 7}`, dash: "8 6" }));
export const TOP = 5;                 // the lines a sheet opens with: its top 5
export const PAGE = 100;              // the Assets view draws its rows a page at a time: thousands at once stall the browser
export const QUICK = 14;              // the quick-pick chips of the Asset row: the most liquid names of a market
export const VERDICTS = ["All", "beats B&H", "makes money", "loses"];
// the targets in the order of a row's `targets`
export const TARGETS = ["avg_monthly", "green", "max_dd", "sharpe", "beats_bh"] as const;
export const target = (r: Row, k: (typeof TARGETS)[number]): boolean => r.targets[TARGETS.indexOf(k)];

// what the K-ratio is, wherever it is shown
export const K_RATIO_TIP =
  "Kestner's K-ratio (2013): how steadily the account grew — the slope of a straight line fitted to the log of the account day by day, over the slope's standard error, times √365 over the days; about 1.1 × Sharpe when the days are independent draws, higher for a steadier climb, lower for one made in a few jumps or broken by long flat or losing stretches";

// buy & hold holds no currency pair and no crude: a list or asset of nothing else is compared with cash
export const CASH_TIP =
  "nothing here can be held (currency pairs, crude): compared with cash at T-bills, the strategy not sized, its idle cash earning T-bills too";
// the market's index beside buy & hold (server/market_index.py)
export const INDEX_TIP =
  "the market's index, bought on the record's first day and held over the same days. A list's: SPY with its dividends for stocks " +
  "and ETFs, BTC on spot for crypto, gold for the spot commodities, gold's future for the CME list. An asset alone has its own " +
  "market's, where an index measures it: SPY beside a US stock or an ETF of US stocks, BTC beside a coin, and beside WTI's spot " +
  "quote, which cannot be held, crude's future held through its rolls; a metal, a future or an ETF of bonds, commodities or " +
  "foreign stocks has its own buy & hold only. What a client could have bought instead: a reference, not the target, which is " +
  "holding the same list or asset at equal risk";
export const INDEX_LINE = { color: "--ink2", dash: "1 3", w: 1.5 };      // dotted beside buy & hold's dashes

export type Lookup = { lists: Map<string, ListInfo>; assets: Map<string, AssetInfo>; byLabel: Map<string, AssetInfo>;
  checks: Map<string, string>; targets: Targets; minTrades: number };
export function lookup(meta: Meta): Lookup {
  return {
    lists: new Map(meta.lists.map((u) => [u.id, u])),
    assets: new Map(meta.assets.map((a) => [a.id, a])),
    byLabel: new Map(meta.assets.map((a) => [a.label.toLowerCase(), a])),
    checks: new Map(meta.checks.map((c) => [c.id, c.label])),
    targets: meta.targets,
    minTrades: meta.min_trades,
  };
}

// the average month sized to the target's drawdown over all the record's years, the size it took, and the same over
// its last five years alone, the years every list has; then buy & hold's at the same drawdown
export function sizedTip(r: Row, minTrades: number): string {
  return sizedOwn(r, minTrades) + sizedHeld(r);
}

function sizedOwn(r: Row, minTrades: number): string {
  if (r.trades != null && r.trades < minTrades)
    return `too few trades to tell (${r.trades}): under ${minTrades} a drawdown measures nothing, as the research desk greys a strategy's trade figures`;
  const k = r.at_target_dd_multiple;
  if (k == null) return r.at_target_dd == null ? "not worked out: a record that never lost a day, or without a whole month"
    : "no position held: nothing to scale";
  const years = r.start && r.end ? `, ${r.start.slice(0, 4)}–${r.end.slice(0, 4)}` : "";
  const size = k < 1 ? `the positions at ${pct0(k)} of their size` : `the positions at ${k.toFixed(2)}× their size`;
  const recent = r.at_target_dd_5y == null ? "last five years: the whole record, five years or less"
    : `last five years alone, the years every list has: ${pct(r.at_target_dd_5y)}`;
  return `all years${years}: ${pct(r.at_target_dd)}, ${size} · ${recent}`;
}

// buy & hold is fully invested: at the target's drawdown it holds that share of the money (or borrows past all of it)
function sizedHeld(r: Row): string {
  const k = r.bh_at_target_dd_multiple;
  if (r.bh_at_target_dd == null || k == null) return "";
  const size = k < 1 ? `${pct0(k)} of the money in it` : `${k.toFixed(2)}× the money in it`;
  return ` · buy & hold at the same drawdown: ${pct(r.bh_at_target_dd)}, ${size}`;
}

export function sizeLong(u: ListInfo | undefined): string {
  if (!u?.size) return u?.venue ?? "";
  const s = u.size;
  return (u.venue ? u.venue + ", " : "") + (s.fixed != null
    ? `${s.fixed} instruments` + (s.trading !== s.fixed ? ` (${s.trading} trading today)` : "")
    : `${s.held} at a time, re-picked monthly with a buffer; ${s.ever} different names have been in it over its history`);
}

export function sizeShort(u: ListInfo | undefined): string {
  if (!u?.size) return "";
  return u.size.fixed != null ? `${u.size.fixed} instruments` : `${u.size.held} names at a time, re-picked monthly, ${u.size.ever} over its history`;
}

// the list a single asset's run came from
export const takenFrom = (u: ListInfo | undefined): string => (u ? `taken from ${u.title}${u.venue ? ": " + u.venue : ""}` : "");

export const verdict = (r: AssetRow): string =>
  (r.sharpe ?? 0) > 0 && (r.cagr ?? 0) > 0 ? (r.beats_bh ? "beats B&H" : "makes money") : "loses";

export const earns = (r: { sharpe: number | null; cagr: number | null }): boolean => (r.sharpe ?? 0) > 0 && (r.cagr ?? 0) > 0;

// Robustness: the checks a row passes of those that apply to it; green only when every check that applies passed and
// none is missing; a row that loses money has none (—)
export const robFull = (rob: Robustness | null): boolean =>
  !!rob && rob.applicable > 0 && rob.passed === rob.applicable && !rob.not_computed;
export const robRank = (rob: Robustness | null): number =>
  !rob ? -1 : (rob.applicable ? rob.passed / rob.applicable : 0) + rob.passed / 1000;
export const robMeasured = (rob: Robustness | null): boolean => !!rob && rob.not_computed < rob.checks.length;
export const STATE: Record<string, string> = {
  passed: "passed", failed: "failed", too_short: "too short", not_applicable: "n/a", not_computed: "not computed",
};

// a row's market and name for the page
export const marketOf = (r: Row, L: Lookup): string =>
  "instrument_id" in r ? L.assets.get(r.instrument_id)?.market ?? "?" : L.lists.get(r.list_id)?.market ?? "?";
export const labelOf = (r: AssetRow, L: Lookup): string => L.assets.get(r.instrument_id)?.label ?? r.instrument_id;

export const isList = (r: Row): r is ListRow => !("instrument_id" in r);

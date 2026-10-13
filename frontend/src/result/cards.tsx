// The parts of a result's popup, as the mockup drew them: tiles against the target, the metrics against buy & hold,
// the robustness checks, the same strategy elsewhere, the parameters each window traded, asset by asset, the calendar.
import { useMemo, useState, type ReactNode } from "react";
import type { BenchmarkFigures, Candidate, Cell, Figures, ListInfo, Months, NameView, ResultView, Robustness, Targets, WindowsView } from "../api/types";
import { count, daysOf, divBg, isNum, monthsOf, n2, pct, pct0, windowsOf } from "../format";
import { RobRows } from "../research/robustness";
import { INDEX_TIP, K_RATIO_TIP, SLOTS, TFS, earns, robMeasured } from "../research/model";
import { SortTh } from "../ui/SortTh";
import { useTip } from "../ui/Tip";

// ---------------------------------------------------------------------------------------------------- tiles
export const Tile = ({ k, v, s, ok = false, title }: { k: string; v: string; s: ReactNode; ok?: boolean; title?: string }) => (
  <div className="tile" title={title}>
    <div className="k">{k}</div>
    <div className={"v" + (ok ? " ok" : "")}>{v}</div>
    <div className="s">{s}</div>
  </div>
);

// ---------------------------------------------------------------------------------------------------- metrics
// held: compared with holding (its column shown); cash: nothing can be held; unkept: imported without its series
export type Holding = "held" | "cash" | "unkept";

const NONE = () => "—";

// I: the market's index over the same days (idx its name), a reference beside buy & hold
// dd: the target's max drawdown, the one each column's average month is sized to on the row that says so
export function MetricsCard({ o, B, I, idx, tf, dd, holding, what }: { o: Figures; B: BenchmarkFigures | null;
  I: BenchmarkFigures | null; idx: string | null; tf: string; dd: number; holding: Holding; what: string }) {
  const calmar = (x: { cagr?: number | null; max_dd?: number | null }) =>
    isNum(x.cagr) && isNum(x.max_dd) && x.max_dd < 0 ? x.cagr / -x.max_dd : null;
  const b = B ?? ({} as Partial<BenchmarkFigures>), ix = I ?? ({} as Partial<BenchmarkFigures>);
  const held = holding === "held" && B != null, indexed = I != null && idx != null;
  // [metric, the strategy's, buy & hold's and the index's (from either's figures), hover]
  const rows: [string, string, (x: Partial<BenchmarkFigures>) => string, string][] = [
    ["Avg / month", pct(o.avg_monthly), (x) => pct(x.avg_monthly), "the account's growth a month, compounded"],
    ["Max drawdown", pct(o.max_dd, 1), (x) => pct(x.max_dd, 1), ""],
    [`At ${pct(-dd, 0, false)} DD`, pct(o.at_target_dd), (x) => pct(x.at_target_dd), sizedHover(dd)],
    ["Months +", pct0(o.pct_green_active), (x) => pct0(x.pct_green), "share of the months that ended in profit: the strategy's of the months it held a position, holding's of all months"],
    ["Return / yr", pct(o.cagr, 1), (x) => pct(x.cagr, 1), "compounded annual growth"],
    ["Volatility / yr", pct(o.ann_vol, 1, false), (x) => pct(x.ann_vol, 1, false), "standard deviation of the daily returns, annualised"],
    ["Sharpe", n2(o.sharpe), (x) => n2(x.sharpe), ""],
    ["Sortino", n2(o.sortino), (x) => n2(x.sortino), ""],
    ["K-ratio", n2(o.k_ratio), (x) => n2(x.k_ratio), K_RATIO_TIP],
    ["Calmar", n2(calmar(o)), (x) => n2(calmar(x)), "return a year over the max drawdown"],
    ["Longest drawdown", daysOf(o.max_dd_days), (x) => daysOf(x.max_dd_days), "the longest time below a previous high"],
    ["Worst month", pct(o.worst_month, 1), (x) => pct(x.worst_month, 1), ""],
    ["Best month", pct(o.best_month, 1), (x) => pct(x.best_month, 1), ""],
    ["Longest losing run", monthsOf(o.longest_red_streak), (x) => monthsOf(x.longest_red_streak), "red months in a row"],
    ["Time in market", pct0(o.time_in_market), NONE, "share of the days with a position"],
    ["Trades", isNum(o.n_trades) ? count(o.n_trades) + (isNum(o.trades_per_month) ? " · " + o.trades_per_month.toFixed(1) + " a month" : "") : "—", NONE, ""],
    ["Win rate", pct0(o.win_rate), NONE, "share of the trades that made money after costs"],
    ["Average win", pct(o.avg_win), NONE, "a winning trade's return after costs"],
    ["Average loss", pct(o.avg_loss), NONE, "a losing trade's return after costs"],
    ["Profit factor", n2(o.profit_factor), NONE, "the winning trades' returns summed over the losing trades'"],
    ["Median holding", isNum(o.median_trade_bars) ? `${o.median_trade_bars} bars of ${tf}` : "—", NONE, ""],
  ];
  const beside = indexed ? ` and ${idx}, the market's index` : "";
  const cap = holding === "held" ? `out-of-sample, against holding ${what}${beside} over the same days`
    : holding === "cash" ? `out-of-sample, against cash: nothing in ${what} can be held`
      + (indexed ? `; ${idx}, the market's index, over the same days` : "")
    : "out-of-sample; the buy & hold series is not kept for results imported from the files before the database"
      + (indexed ? `; ${idx}, the market's index, over the same days` : "");
  return (
    <div className="card">
      <h3>Performance metrics</h3>
      <p className="cap">{cap}</p>
      <table className="mt">
        <thead>
          <tr>
            <th className="l">Metric</th><th>Strategy</th>
            {held && <th title={`${what} bought at the record's start and held on spot; its only trades are its changes`}>Buy &amp; hold</th>}
            {indexed && <th title={INDEX_TIP}>{idx}</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, a, f, tip]) => (
            <tr key={k}><td className="l" title={tip || undefined}>{k}</td><td>{a}</td>{held && <td>{f(b)}</td>}{indexed && <td>{f(ix)}</td>}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// the row of the average month at the target's drawdown: every column sized the same way, so they compare at one risk
const sizedHover = (dd: number) =>
  `the average month with the positions scaled so that the max drawdown is the target's ${pct(dd, 0)}, buy & hold and ` +
  "the index scaled the same way (holding part of the money, or more than all of it): which makes more at one drawdown; " +
  "the money left idle earns nothing, money borrowed past the equity pays T-bills + 1.5%; the size is chosen knowing " +
  "the record's worst drawdown: it compares records and is not a size to trade; — for a strategy of too few trades to tell";

// ---------------------------------------------------------------------------------------------------- robustness
export function RobCard({ rob, loses, labels }: { rob: Robustness | null; loses: string | null; labels: Map<string, string> }) {
  const head = !rob ? "not checked" : !robMeasured(rob) ? "not measured: computed by code older than the checks"
    : `${rob.passed} of ${rob.applicable} checks passed${rob.not_computed ? `, ${rob.not_computed} not measured` : ""}`;
  return (
    <div className="card">
      <h3>Robustness <span className="muted" style={{ fontWeight: 400 }}>· {head}</span></h3>
      <p className="cap">each measured in this evaluation{rob?.checks.some((c) => c[0] === "peers") ? " (its market: from the other instruments' own evaluations)" : ""} and judged against its threshold; n/a: it does not apply to this strategy</p>
      {rob ? <RobRows rob={rob} labels={labels} /> : <p className="muted">{loses || "it loses money out-of-sample"}: nothing to check</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------- elsewhere
export const GRID: Record<string, [string, (x: Cell) => number | null, (v: number | null) => string, number]> = {
  sharpe: ["Sharpe", (x) => x.sharpe, n2, 1.5],
  avg: ["Avg / month", (x) => x.avg_monthly, (v) => pct(v), 0.02],
  bh: ["vs B&H / yr", (x) => x.vs_bh, (v) => pct(v, 1), 0.1],
};

type ElseProps = { res: ResultView; lists: ListInfo[]; metric: string; setMetric: (m: string) => void; onOpen: (key: string) => void;
  aloneLabel: string | null };

export function ElseCard({ res, lists, metric, setMetric, onOpen, aloneLabel }: ElseProps) {
  const [, get, fmt, scale] = GRID[metric] ?? GRID.sharpe;
  const mine = res.elsewhere.lists;
  const at = new Map(mine.map((x) => [x.list_id + "|" + x.timeframe, x]));
  const alone = new Map(res.elsewhere.alone.map((x) => [x.timeframe, x]));
  const sh = mine.map((x) => x.sharpe).filter(isNum).toSorted((a, b) => a - b);
  const med = sh.length ? (sh[(sh.length - 1) >> 1] + sh[sh.length >> 1]) / 2 : null;
  const cell = (x: Cell | undefined, here: boolean, where: string, tf: string) => !x ? <td key={tf} className="muted">·</td> : (
    <td key={tf} className={"pick" + (x.stale ? " old" : "")} onClick={() => onOpen(x.key)}
      style={{ ...divBg(get(x), scale), ...(here ? { outline: "2px solid var(--ink)", outlineOffset: -2 } : {}) }}
      title={`${where} · ${x.timeframe}: Sharpe ${n2(x.sharpe)}, avg month ${pct(x.avg_monthly)}, vs ${x.bh_cash ? "cash" : "B&H"} ${pct(x.vs_bh, 1)} a year, ${x.targets_met} of 5 targets${x.stale ? " · older code" : ""}`}>
      {fmt(get(x))}
    </td>
  );
  const isList = res.kind === "list";
  return (
    <div className="card">
      <div className="pnlhead">
        <h3>Same strategy elsewhere</h3>
        <span className="cap">
          <span className="seg">
            {Object.entries(GRID).map(([k, [lab]]) => (
              <button key={k} className={metric === k ? "on" : ""} onClick={() => setMetric(k)}>{lab}</button>
            ))}
          </span>
        </span>
      </div>
      <p className="cap">
        {mine.length
          ? `on ${mine.length} lists × timeframes it makes money on ${mine.filter(earns).length}, beats buy & hold on ${mine.filter((x) => x.beats_bh).length} · median Sharpe ${n2(med)}`
          : "not run on any list"}
      </p>
      <table className="heat">
        <thead><tr><th className="l" aria-label="list" />{TFS.map((tf) => <th key={tf}>{tf}</th>)}</tr></thead>
        <tbody>
          {aloneLabel && (
            <tr>
              <td className="l">{aloneLabel} alone</td>
              {TFS.map((tf) => cell(alone.get(tf), tf === res.row.timeframe, aloneLabel + " alone", tf))}
            </tr>
          )}
          {lists.map((u) => (
            <tr key={u.id}>
              <td className="l">{u.title}</td>
              {TFS.map((tf) => cell(at.get(u.id + "|" + tf), isList && u.id === res.row.list_id && tf === res.row.timeframe, u.title, tf))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="cap" style={{ marginTop: 4 }}>outlined: this result · faded: older code · a dot: not run · click a cell to open it</p>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------- parameters
// a parameter's value as a strategy's grid writes it, in words where it is not set: slots (every name of the list gets
// its share of capital) and the ATR exits
const NONE_AS = new Map([["slots", "all"], ["stop", "off"], ["stop_atr", "off"], ["take_atr", "off"], ["trail_atr", "off"]]);
export const paramValue = (k: string, v: unknown): string =>
  v === null ? NONE_AS.get(k) ?? "None" : v === true ? "True" : v === false ? "False" : String(v);
export const paramText = (p: Record<string, unknown> | null | undefined): string =>
  !p || !Object.keys(p).length ? "—" : Object.entries(p).map(([k, v]) => `${k} ${paramValue(k, v)}`).join(" · ");
// a walk-forward's pick by the parameters it chose between: those with more than one value in the grid (without a grid,
// all of them); a window with none (no configuration had traded enough before it to be chosen) holds no position
export function pickText(grid: Record<string, unknown[]> | null, p: Record<string, unknown>): string {
  if (!Object.keys(p).length) return "no position";
  const keys = Object.entries(grid ?? {}).filter(([, v]) => v.length > 1).map(([k]) => k);
  return paramText(keys.length ? Object.fromEntries(keys.map((k) => [k, p[k]])) : p);
}
export const windowDays = (f: WindowsView): number =>
  Math.round(((f.w.length > 1 ? Date.parse(f.w[1][0]) : Date.parse(f.end)) - Date.parse(f.w[0][0])) / 864e5);

// the last run of windows trading the same pick: its first window
export function since(W: WindowsView["w"]): number {
  let k = W.length - 1;
  while (k && W[k - 1][1] === W[k][1]) k--;
  return k;
}

type ParamProps = { f: WindowsView | null; grid: Record<string, unknown[]> | null; fixed: Record<string, unknown> | null; end: string | null;
  strategy?: boolean };

// the grid tried and what each walk-forward window traded, named by the parameters with a choice, the fixed ones said
// once: colours follow the picks by use, the sixth and later grey. `strategy`: the windows chose a strategy, not
// parameters (strategy_pick)
export function ParamCard({ f, grid, fixed, end, strategy = false }: ParamProps) {
  const tip = useTip();
  const title = strategy ? "Strategy chosen" : "Parameters";
  if (!f) {
    return (
      <div className="card">
        <h3>{title}</h3>
        <p className="cap">set before the test and never changed: nothing is chosen on the data</p>
        <div className="kv"><div>Fixed</div><div>{paramText(fixed)}</div></div>
      </div>
    );
  }
  const entries = Object.entries(grid ?? {}), varied = entries.filter(([, v]) => v.length > 1), single = entries.filter(([, v]) => v.length === 1);
  const n = varied.reduce((a, [, v]) => a * v.length, 1), W = f.w, sets = f.sets;
  const uses = sets.map((_, i) => W.filter((x) => x[1] === i).length);
  const order = uses.map((c, i) => [c, i]).toSorted((a, b) => b[0] - a[0]).map((x) => x[1]), rest = order.slice(SLOTS.length);
  const color = (i: number) => (order.indexOf(i) < SLOTS.length ? SLOTS[order.indexOf(i)] : "--muted");
  const changes = W.filter((x, i) => i && x[1] !== W[i - 1][1]).length;
  const first = since(W), run = W.length - first;
  // never chosen: one parameter's values by name, several parameters' combinations by count
  const picked = new Set(sets.map((s) => JSON.stringify(varied.map(([k]) => s[k]))));
  const unused = varied.length === 1 ? varied[0][1].filter((v) => !picked.has(JSON.stringify([v]))) : [];
  const never = unused.length ? unused.map((v) => paramValue(varied[0][0], v)).join(", ") + ": never chosen"
    : varied.length > 1 && n > picked.size ? `${n - picked.size} of ${n} combinations never chosen` : "";
  const label = (i: number) => pickText(grid, sets[i]);
  const windowTip = (k: number) => (
    <><div className="t">{W[k][0]} → {k + 1 < W.length ? W[k + 1][0] : end}</div>{label(W[k][1])}</>
  );
  return (
    <div className="card">
      <h3>{title}</h3>
      <p className="cap">
        re-chosen every {windowDays(f)} days: the best Sharpe on all the history before that day, traded unchanged until the next choice
      </p>
      <div className="kv">
        <div>Tried</div>
        <div>
          {varied.map(([k, v]) => `${k} ${v.map((x) => paramValue(k, x)).join(", ")}`).join(" · ") || "—"}
          {varied.length > 1 && <span className="muted"> ({n} combinations)</span>}
          {single.length > 0 && <span className="muted"> · fixed: {single.map(([k, v]) => `${k} ${paramValue(k, v[0])}`).join(", ")}</span>}
        </div>
        <div>Now</div>
        <div>
          {label(W[W.length - 1][1])} <span className="muted">· since {W[first][0]}
            {run > 1 ? (run === W.length ? `, all ${run} windows` : `, ${run} windows in a row`) : ""}</span>
        </div>
        <div>Changes</div><div>{changes} in {windowsOf(W.length)}</div>
      </div>
      <div className="foldstrip" onPointerLeave={tip.hide}>
        {W.map((x, i) => (
          <span key={x[0]} style={{ background: `var(${color(x[1])})` }} onPointerMove={(ev) => tip.show(ev, windowTip(i))} />
        ))}
      </div>
      <div className="cap" style={{ display: "flex", justifyContent: "space-between", margin: 0 }}><span>{W[0][0]}</span><span>{end}</span></div>
      <div className="legend" style={{ marginTop: 6 }}>
        {order.slice(0, SLOTS.length).map((i) => (
          <span key={i}><i className="box" style={{ background: `var(${color(i)})` }} />{label(i)} · {windowsOf(uses[i])}</span>
        ))}
        {rest.length > 0 && (
          <span><i className="box" style={{ background: "var(--muted)" }} />{rest.length} other{rest.length === 1 ? "" : "s"} · {windowsOf(rest.reduce((a, i) => a + uses[i], 0))}</span>
        )}
        {never && <span className="muted">{never}</span>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------- asset by asset
type NameSort = { k: "l" | "n" | "w" | "r" | "s" | "b"; dir: 1 | -1 } | null;

export function NamesCard({ names, target, onOpen }: { names: NameView[]; target: number; onOpen: (key: string) => void }) {
  const [sort, setSort] = useState<NameSort>({ k: "r", dir: -1 });          // the names that made the most first
  const alone = names.some((x) => x.alone);
  const shown = useMemo(() => {
    const get: Record<string, (x: NameView) => string | number | null> = {
      l: (x) => x.label, n: (x) => x.trades, w: (x) => x.won, r: (x) => x.compounded,
      s: (x) => x.alone?.sharpe ?? null, b: (x) => x.alone?.vs_bh ?? null,
    };
    if (!sort) return names;
    return names.toSorted((a, b) => {
      const va = get[sort.k](a), vb = get[sort.k](b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return (va > vb ? 1 : va < vb ? -1 : 0) * sort.dir;
    });
  }, [names, sort]);
  const abs = names.map((x) => Math.abs(x.compounded ?? 0)).toSorted((a, b) => a - b), scale = abs[Math.floor(abs.length * 0.9)] || 1;
  const th = (k: NonNullable<NameSort>["k"], label: string, title = "", cls = "") => (
    <SortTh label={label} title={title} cls={cls} dir={sort?.k === k ? sort.dir : null}
      onSort={() => setSort({ k, dir: sort?.k === k ? (-sort.dir as 1 | -1) : -1 })} />
  );
  return (
    <div className="card">
      <h3>Asset by asset</h3>
      <p className="cap">
        {names.filter((x) => (x.compounded ?? 0) > 0).length} of {names.length} names made money on their trades in this book
        {alone ? " · alone: the same strategy on the name by itself, with all the capital, walk-forward on its own past" : ""}
      </p>
      <div className="scroll" style={{ maxHeight: 330 }}>
        <table>
          <thead>
            <tr>
              {th("l", "Asset", "", "l")}{th("n", "Trades")}{th("w", "Won", "share of its trades that made money after costs")}
              {th("r", "Return", "its trades in this book compounded, after costs")}<th aria-label="return, drawn" />
              {alone && <>{th("s", "Sharpe alone", "the same strategy on this name by itself, out-of-sample; green: meets the target ≥ 1")}
                {th("b", "vs its B&H alone", "alone, a year above holding the name at equal risk; green: more than holding it")}</>}
            </tr>
          </thead>
          <tbody>
            {shown.map((x) => (
              <tr key={x.instrument_id}>
                <td className="l">{x.label}</td><td>{count(x.trades)}</td><td>{pct0(x.won)}</td>
                <td className={(x.compounded ?? 0) > 0 ? "ok" : undefined}>{pct(x.compounded, 1)}</td>
                <td className="l" style={{ width: 76 }}>
                  {isNum(x.compounded) && (
                    <span className={"bar" + (x.compounded < 0 ? " neg" : "")}
                      style={{ width: Math.max(1, Math.round(Math.min(Math.abs(x.compounded) / scale, 1) * 70)) }} />
                  )}
                </td>
                {alone && (
                  <>
                    <td className={"pick" + (x.alone?.stale ? " old" : "") + (x.alone && (x.alone.sharpe ?? -9) >= target ? " ok" : "")}
                      onClick={() => x.alone && onOpen(x.alone.key)}>{x.alone ? n2(x.alone.sharpe) : "—"}</td>
                    <td className={x.alone?.beats_bh ? "ok" : undefined}>{x.alone ? pct(x.alone.vs_bh, 1) : "—"}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------- candidates
export function CandidatesCard({ items, onOpen }: { items: Candidate[]; onOpen: (key: string) => void }) {
  return (
    <div className="card">
      <h3>Candidates</h3>
      <p className="cap">each strategy on this instrument alone, walk-forward on its own past; every 91 days the one with the best Sharpe so far is held until the next choice · click one to open it</p>
      <div className="scroll" style={{ maxHeight: 330 }}>
        <table>
          <thead><tr><th className="l">Strategy</th><th title="alone on this instrument, out-of-sample">Sharpe alone</th><th title="windows in which it was the one held">Chosen</th></tr></thead>
          <tbody>
            {items.map((c) => (
              <tr key={c.key} className={c.stale ? "stale" : undefined} style={{ cursor: "pointer" }} onClick={() => onOpen(c.key)}>
                <td className="l"><b>{c.strategy}</b></td><td>{n2(c.sharpe)}</td><td>{c.windows ? windowsOf(c.windows) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------- calendar
// {first: 'YYYY-MM', v: [basis points, month after month]} -> [year, month 0-11, return]
function calMonths(m: Months | null): [number, number, number | null][] {
  if (!m) return [];
  const [y0, m0] = m.first.split("-").map(Number);
  return m.v.map((v, i) => [y0 + Math.floor((m0 - 1 + i) / 12), (m0 - 1 + i) % 12, v == null ? null : v / 1e4]);
}
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function Calendar({ m, bh, what }: { m: Months | null; bh: Months | null; what: string }) {
  const by: Record<number, (number | null)[]> = {}, held: Record<number, number> = {};
  for (const [y, k, v] of calMonths(m)) (by[y] ??= Array(12).fill(null))[k] = v;
  for (const [y, , v] of calMonths(bh)) if (v != null) held[y] = (held[y] ?? 1) * (1 + v);
  const cell = (v: number | null, max: number, bold: boolean, key: string) => (
    <td key={key} style={divBg(v, max)}>{!isNum(v) ? "" : bold ? <b>{(v * 100).toFixed(1)}</b> : (v * 100).toFixed(1)}</td>
  );
  return (
    <div className="card" style={{ marginTop: 10 }}>
      <h3>Calendar months, %</h3>
      <p className="cap">out-of-sample, complete months · Year: its months compounded{bh ? ` · B&H: holding ${what} over the same months` : ""}</p>
      <div className="scroll">
        <table className="heat">
          <thead>
            <tr>
              <th className="l">Year</th>{MONTHS.map((x) => <th key={x}>{x}</th>)}
              <th title="the year's months compounded">Year</th>{bh && <th title={`holding ${what} over the same months`}>B&amp;H</th>}
            </tr>
          </thead>
          <tbody>
            {Object.keys(by).map(Number).toSorted((a, b) => a - b).map((y) => {
              const got = by[y].filter((v): v is number => v != null);
              const year = got.length ? got.reduce((a, v) => a * (1 + v), 1) - 1 : null;
              return (
                <tr key={y}>
                  <td className="l">{y}</td>
                  {by[y].map((v, k) => cell(v, 0.06, false, "m" + k))}
                  {cell(year, 0.3, true, "y")}
                  {bh && cell(held[y] != null ? held[y] - 1 : null, 0.3, false, "b")}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export const targetsLine = (T: Targets) => ({
  avg: `target ≥ ${pct(T.avg_monthly, 1, false)}`, green: `target ≥ ${pct0(T.pct_green)}`, dd: `target ≥ ${pct(T.max_dd, 0)}`,
  sharpe: `target ≥ ${T.sharpe}`,
});

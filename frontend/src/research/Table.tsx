// Research's table, as the mockup drew it: a list's results with their Robustness, or single assets a page at a time;
// a click on a header sorts by it (again: the other way), a click on a row opens its result, a tick draws its line.
import { memo, useEffect, useRef } from "react";
import type { AssetRow, Row } from "../api/types";
import { swatch } from "../charts/TimeChart";
import { count, isNum, n2, pct, pct0 } from "../format";
import { SortTh } from "../ui/SortTh";
import { useTip } from "../ui/Tip";
import { CASH_TIP, K_RATIO_TIP, PAGE, labelOf, marketOf, robMeasured, robRank, sizedTip, sizeLong, takenFrom, target, type Lookup, type Slot } from "./model";
import { robClass, robTip } from "./robustness";

export type SortKey = "strategy" | "list" | "asset" | "cls" | "tf" | "oos" | "avg" | "green" | "dd" | "atdd" | "sharpe" | "k" | "bh"
  | "trades" | "tm" | "rob" | "mo";
export type Sort = { key: SortKey; dir: 1 | -1 } | null;

export function sortValue(r: Row, k: SortKey, L: Lookup): string | number | null {
  switch (k) {
    case "strategy": return r.strategy;
    case "list": return L.lists.get(r.list_id)?.title ?? r.list_id;
    case "asset": return labelOf(r as AssetRow, L);
    case "cls": return marketOf(r, L);
    case "tf": return r.timeframe;
    case "oos": return r.start;                       // the record's first day: ▲ the longest records first
    case "avg": return r.avg_monthly;
    case "green": return r.green;
    case "dd": return r.max_dd;
    case "atdd": return r.at_target_dd;
    case "sharpe": return r.sharpe;
    case "k": return r.k_ratio;
    case "bh": return r.vs_bh;
    case "trades": return r.trades;
    case "tm": return r.targets_met;
    case "rob": return robRank(r.robustness ?? null);
    case "mo": return r.months;
  }
}

// nulls last whichever way
export function sorted<R extends Row>(rows: R[], sort: Sort, L: Lookup): R[] {
  if (!sort) return rows;
  const { key, dir } = sort;
  return rows.toSorted((a, b) => {
    const va = sortValue(a, key, L), vb = sortValue(b, key, L);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    return (va > vb ? 1 : va < vb ? -1 : 0) * dir;
  });
}

const Pips = ({ n }: { n: number }) => (
  <span className="pips" title={`${n} of 5 targets`}>{[0, 1, 2, 3, 4].map((i) => <i key={i} className={i < n ? "on" : ""} />)}</span>
);

const ok = (hit: boolean) => (hit ? "ok" : undefined);

// the figures judged against the target: green when met
function Figures({ r, L }: { r: Row; L: Lookup }) {
  return (
    <>
      <td className={ok(target(r, "avg_monthly"))}>{pct(r.avg_monthly)}</td>
      <td className={ok(target(r, "green"))}>{pct0(r.green)}</td>
      <td className={ok(target(r, "max_dd"))}>{pct(r.max_dd, 1)}</td>
      <td className={ok(isNum(r.at_target_dd) && r.at_target_dd >= L.targets.avg_monthly)} title={sizedTip(r, L.minTrades)}>{pct(r.at_target_dd)}</td>
      <td className={ok(target(r, "sharpe"))} title={r.bh_cash ? "compared with cash: nothing here can be held" : "buy & hold: Sharpe " + n2(r.bh_sharpe)}>{n2(r.sharpe)}</td>
    </>
  );
}

type RowProps = { r: Row; L: Lookup; every: boolean; slot: Slot | undefined; onTick: (key: string) => void;
  onOpen: (key: string) => void; hl: boolean };

const TableRow = memo(function TableRow({ r, L, every, slot, onTick, onOpen, hl }: RowProps) {
  const tip = useTip();
  const asset = "instrument_id" in r;
  const cls = [r.stale ? "stale" : "", slot ? "sel" : "", hl ? "hl" : ""].filter(Boolean).join(" ");
  const tick = (
    <td className="tickc" onClick={(e) => e.stopPropagation()}>
      <input type="checkbox" checked={!!slot} onChange={() => onTick(r.key)} aria-label="draw on the chart" />
      <i className="key" style={slot ? { background: swatch(slot.color, !!slot.dash) } : undefined} />
    </td>
  );
  const oos = <td className="muted" title={`${r.start} → ${r.end}`}>{r.start?.slice(0, 4) ?? "—"}</td>;
  const k = (
    <td title={r.bh_cash ? "compared with cash: nothing here can be held" : r.bh_k_ratio != null ? "buy & hold: K-ratio " + n2(r.bh_k_ratio) : undefined}>
      {n2(r.k_ratio)}
    </td>
  );
  const bh = <td className={ok(r.beats_bh)} title={r.bh_cash ? "against cash: " + CASH_TIP : undefined}>{pct(r.vs_bh, 1)}</td>;
  const rob = r.robustness ?? null;
  const robCell = (
    <td className={robClass(rob)} onMouseMove={(e) => tip.show(e, robTip(r, L.checks), true)} onMouseLeave={tip.hide}>
      {robMeasured(rob) ? `${rob!.passed}/${rob!.applicable}` : "—"}
    </td>
  );
  if (asset) {
    const a = r as AssetRow;
    return (
      <tr data-key={r.key} className={cls || undefined} onClick={() => onOpen(r.key)}>
        {tick}
        <td className="l"><b>{r.strategy}</b></td>
        {every && <><td className="l" title={takenFrom(L.lists.get(a.list_id))}>{labelOf(a, L)}</td><td className="l">{marketOf(a, L)}</td></>}
        <td>{r.timeframe}</td>{oos}<td>{r.months ?? "—"}</td>
        <Figures r={r} L={L} />{k}{bh}<td><Pips n={r.targets_met} /></td>{robCell}<td>{r.trades == null ? "—" : count(r.trades)}</td>
      </tr>
    );
  }
  const u = L.lists.get(r.list_id);
  return (
    <tr data-key={r.key} className={cls || undefined} onClick={() => onOpen(r.key)}>
      {tick}
      <td className="l"><b>{r.strategy}</b></td>
      <td className="l" title={sizeLong(u)}>{u ? u.title : r.list_id}</td>
      <td>{r.timeframe}</td>{oos}
      <Figures r={r} L={L} />{k}{bh}<td><Pips n={r.targets_met} /></td>{robCell}
      <td>{r.trades == null ? "—" : count(r.trades)}</td>
    </tr>
  );
});

type Props = { rows: Row[]; assets: boolean; every: boolean; L: Lookup; ticks: Map<string, Slot>; sort: Sort;
  setSort: (s: Sort) => void; shown: number; more: () => void; onTick: (key: string) => void; onOpen: (key: string) => void;
  highlight: string | null };

export function Table({ rows, assets, every, L, ticks, sort, setSort, shown, more, onTick, onOpen, highlight }: Props) {
  const body = useRef<HTMLTableSectionElement>(null);
  useEffect(() => {
    if (!highlight) return;
    body.current?.querySelector(`tr[data-key="${CSS.escape(highlight)}"]`)?.scrollIntoView({ block: "nearest" });
  }, [highlight]);
  const th = (k: SortKey, label: string, title: string, cls = "") => (
    <SortTh label={label} title={title} cls={cls} dir={sort?.key === k ? sort.dir : null}
      onSort={() => setSort({ key: k, dir: sort?.key === k ? (-sort.dir as 1 | -1) : -1 })} />
  );
  const last = rows.reduce((m, r) => (r.end && r.end > m ? r.end : m), "").slice(0, 7) || "—";
  const oosTh = th("oos", "OOS", `out-of-sample from this year to the last bar (${last}); point at a year for the exact dates`);
  const T = L.targets;
  const figures = (
    <>
      {th("avg", "Avg / month", "the account's growth a month, compounded (1.5% a month = 19.6% a year); green = meets the target ≥ 1.5%")}
      {th("green", "Months +, %", "share of the months with a position that ended in profit; green = meets the target ≥ 70%")}
      {th("dd", "Max DD", "max drawdown; green = meets the target ≥ −10%")}
      {th("atdd", `At ${pct(-T.max_dd, 0, false)} DD`, `the average month over all the out-of-sample years with the strategy's positions scaled so that its max drawdown is the target's ${pct(T.max_dd, 0)}, to compare returns that come with different drawdowns: one that fell 25% held at 40% of its size, one that fell 5% at twice it; the money left idle earns nothing, as in Avg / month; money borrowed past the equity pays T-bills + 1.5%; the size is chosen knowing the record's worst drawdown: it compares records and is not a size to trade; — under ${L.minTrades} trades, too few to tell; green = meets the target ≥ ${pct(T.avg_monthly, 1, false)}; point at a figure for its size, the same over the last five years alone, the years every list has, and buy & hold's at the same drawdown`)}
      {th("sharpe", "Sharpe", "out-of-sample (walk-forward), the Sharpe the research desk shows; green = meets the target ≥ 1; point at it for buy & hold's")}
      {th("k", "K-ratio", `out-of-sample (walk-forward), ${K_RATIO_TIP}; point at it for buy & hold's`)}
    </>
  );
  const head = assets ? (
    <tr>
      <th title="draw on the chart" aria-label="draw on the chart" />
      {th("strategy", "Strategy", "strategy", "l")}
      {every && <>{th("asset", "Asset", "the asset the strategy trades alone, with all the capital", "l")}{th("cls", "Market", "the asset's market", "l")}</>}
      {th("tf", "TF", "timeframe")}{oosTh}{th("mo", "OOS months", "out-of-sample months: a short record is luck-prone")}
      {figures}
      {th("bh", "vs its B&H / yr", "return a year above holding the asset, the strategy sized to its risk from the last 90 days (at most 2x) with its idle cash at T-bills; a currency pair or crude, which cannot be held: above cash at T-bills, not sized; green = more: the target")}
      {th("tm", "Targets", "targets met of 5")}
      {th("rob", "Robustness", "robustness checks passed of those that apply to the row, its luck counted against the tries on this asset; green = all passed; — = it loses money, nothing to check; point at it for each check")}
      {th("trades", "Trades", "out-of-sample trades")}
    </tr>
  ) : (
    <tr>
      <th title="draw on the chart" aria-label="draw on the chart" />
      {th("strategy", "Strategy", "strategy", "l")}{th("list", "List", "the instrument list the strategy was run on", "l")}
      {th("tf", "TF", "timeframe")}{oosTh}{figures}
      {th("bh", "vs B&H / yr", "return a year above holding the same list, the strategy sized to its risk from the last 90 days (at most 2x) with its idle cash at T-bills, as the research desk counts \"more money than buy-and-hold once sized to equal risk\"; FX, which cannot be held: above cash at T-bills, not sized; green = more: the target")}
      {th("tm", "Targets", "targets met of 5")}
      {th("rob", "Robustness", "robustness checks passed of those that apply to the row; green = all passed; — = it loses money, nothing to check; point at it for each check")}
      {th("trades", "Trades", "out-of-sample trades")}
    </tr>
  );
  const drawn = assets ? rows.slice(0, shown) : rows;
  return (
    <table id="btable">
      <thead>{head}</thead>
      <tbody ref={body}>
        {drawn.map((r) => (
          <TableRow key={r.key} r={r} L={L} every={every} slot={ticks.get(r.key)} onTick={onTick} onOpen={onOpen} hl={highlight === r.key} />
        ))}
        {assets && rows.length > shown && (
          <tr className="more">
            <td className="l" colSpan={every ? 16 : 14}>
              <button className="btn" onClick={more}>Show {Math.min(PAGE, rows.length - shown)} more</button>{" "}
              <span className="muted">{count(shown)} of {count(rows.length)} shown</span>
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

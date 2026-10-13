// A result's popup: what its row cannot hold. The strategy's rule and how it was measured, its figures against the
// target and against holding the same list or asset, its cumulative P&L and drawdown, what shows whether it holds up,
// the same strategy elsewhere, the calendar, the parameters each window traded. ‹ › (and ← →) step through the table's
// rows as sorted and filtered, the rows next to it read ahead; Esc, ✕ or a click outside closes it. One popup a result
// (App keys it): it opens scrolled to its top, focused.
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useEffectEvent, useRef } from "react";
import { resultQuery, useResult } from "../api/queries";
import type { AssetRow, ListRow, Meta, Point, ResultView, Row } from "../api/types";
import { TimeChart, Legend, type Line } from "../charts/TimeChart";
import { cumPct, money, n2, pct, pct0, yearsOf } from "../format";
import { CASH_TIP, INDEX_LINE, INDEX_TIP, labelOf, robFull, robMeasured, sizeLong, sizeShort, takenFrom, target, type Lookup } from "../research/model";
import { Boundary } from "../ui/Boundary";
import {
  Calendar, CandidatesCard, ElseCard, MetricsCard, NamesCard, ParamCard, RobCard, Tile, paramText, since, targetsLine, windowDays,
  type Holding,
} from "./cards";

type Shown = { showBH: boolean; setShowBH: (v: boolean) => void; showIdx: boolean; setShowIdx: (v: boolean) => void };
type Props = { k: string; meta: Meta; L: Lookup; rows: string[]; onOpen: (key: string) => void; onClose: () => void;
  metric: string; setMetric: (m: string) => void } & Shown;

export function Popup({ k, meta, L, rows, onOpen, onClose, metric, setMetric, ...shown }: Props) {
  const q = useResult(k);
  const qc = useQueryClient();
  const sheet = useRef<HTMLDivElement>(null);
  const pos = rows.indexOf(k);
  const step = (d: number) => {
    const next = pos < 0 ? undefined : rows[pos + d];
    if (next) onOpen(next);
  };
  const onKey = useEffectEvent((ev: KeyboardEvent) => {
    if (ev.key === "Escape") {
      onClose();
      ev.preventDefault();
    } else if ((ev.key === "ArrowLeft" || ev.key === "ArrowRight") && !/^(INPUT|SELECT|TEXTAREA)$/.test((ev.target as HTMLElement).tagName)) {
      step(ev.key === "ArrowLeft" ? -1 : 1);
      ev.preventDefault();
    }
  });
  useEffect(() => {
    const key = (ev: KeyboardEvent) => onKey(ev);
    addEventListener("keydown", key);
    document.body.style.overflow = "hidden";
    sheet.current?.focus({ preventScroll: true });
    return () => {
      removeEventListener("keydown", key);
      document.body.style.overflow = "";
    };
  }, []);
  const before = pos > 0 ? rows[pos - 1] : null, after = pos >= 0 ? rows[pos + 1] ?? null : null;
  useEffect(() => {                                 // ‹ › open at once: the rows next to this one are read ahead
    for (const n of [after, before]) if (n) void qc.prefetchQuery(resultQuery(n));
  }, [qc, before, after]);

  const res = q.data, r: Row | undefined = res?.row;
  const asset = r && "instrument_id" in r ? (r as AssetRow) : null;
  const u = r ? L.lists.get(r.list_id) : undefined;
  const where = !r ? "" : asset ? labelOf(asset, L) + " alone" : u ? u.title : r.list_id;
  return (
    <div className="modal on" role="presentation" onClick={(ev) => { if (ev.target === ev.currentTarget) onClose(); }}>
      <div className="sheet" ref={sheet} role="dialog" aria-modal="true" aria-labelledby="card-title" tabIndex={-1}>
        <div className="sheet-head">
          <h2 id="card-title">{r?.strategy ?? k.split("@")[0]}</h2>
          {r && (
            <>
              <span className="tag" title={asset ? "one strategy with all the capital on this asset, " + takenFrom(u) : sizeLong(u)}>{where}</span>
              <span className="tag">{r.timeframe}</span>
              {r.stale && <span className="tag" title="computed by code older than the latest changes: re-run it before use">older code</span>}
            </>
          )}
          <span className="right">
            {pos >= 0 && (
              <>
                <button className="btn" title="previous row (←)" disabled={pos === 0} onClick={() => step(-1)}>‹</button>
                <span className="cnt">{pos + 1} of {rows.length}</span>
                <button className="btn" title="next row (→)" disabled={pos >= rows.length - 1} onClick={() => step(1)}>›</button>
              </>
            )}
            <button className="btn" title="close (Esc)" onClick={onClose}>✕</button>
          </span>
        </div>
        <div className="sheet-body">
          <Boundary what="This result">
            {q.error ? <div className="note">Could not load this result: {q.error.message}</div>
              : !res || !r ? <div className="loading">Loading…</div>
              : res.kind === "list" ? <ListBody res={res} r={r as ListRow} meta={meta} L={L} onOpen={onOpen} metric={metric} setMetric={setMetric} {...shown} />
              : <AssetBody res={res} r={r as AssetRow} meta={meta} L={L} onOpen={onOpen} metric={metric} setMetric={setMetric} {...shown} />}
          </Boundary>
        </div>
      </div>
    </div>
  );
}

type BodyProps<R> = { res: ResultView; r: R; meta: Meta; L: Lookup; onOpen: (key: string) => void; metric: string;
  setMetric: (m: string) => void } & Shown;

// the cumulative P&L with its buy & hold and its market's index, and the drawdown with the target
function Charts({ res, name, heldName, showBH, setShowBH, showIdx, setShowIdx, maxDD }: { res: ResultView; name: string;
  heldName: string | null; maxDD: number } & Shown) {
  const pts = res.curve.pts as Point[], bh = res.benchmark?.pts as Point[] | undefined, ix = res.index;
  const held: Line[] = showBH && bh?.length ? [{ name: "Buy & hold", color: "--muted", pts: bh, dash: "4 3", w: 1.5 }] : [];
  const index: Line[] = showIdx && ix?.pts.length ? [{ name: ix.name, ...INDEX_LINE, pts: ix.pts as Point[] }] : [];
  const lines = held.concat(index, [{ name, color: "--s1", pts }]);
  const dd: Point[] = pts.map((p, i) => [p[0], (res.curve.dd[i] ?? 0) / 1e4]);
  const keys: [string, string, (boolean | string)?][] = [["--s1", name]];
  if (heldName) keys.push(["--muted", `Buy & hold (${heldName})`, true]);
  if (ix) keys.push([INDEX_LINE.color, `${ix.name}, the market's index`, INDEX_LINE.dash]);
  return (
    <div className="card">
      <div className="pnlhead">
        <h3>Cumulative P&amp;L</h3>
        {(heldName || ix) && (
          <span className="cap">
            {heldName && <label className="toggle"><input type="checkbox" checked={showBH} onChange={(e) => setShowBH(e.target.checked)} /> Buy &amp; hold</label>}
            {heldName && ix && " · "}
            {ix && <label className="toggle" title={INDEX_TIP}><input type="checkbox" checked={showIdx} onChange={(e) => setShowIdx(e.target.checked)} /> {ix.name}</label>}
          </span>
        )}
      </div>
      <Legend items={keys} />
      <TimeChart series={lines} base={1} fmt={cumPct} height={280} />
      <h3 style={{ marginTop: 10 }}>Drawdown</h3>
      <TimeChart series={[{ name: "drawdown", color: "--neg", pts: dd }]} fmt={(v) => pct(v, 1)} height={120} area zero endLabels={false}
        refs={[{ v: maxDD, label: "target " + pct(maxDD, 0) }]} />
    </div>
  );
}

const fillText = (fill: string | null) => (fill === "next_open" ? ", fills at the next bar's open" : fill === "next_close" ? ", fills at the next bar's close" : "");

function ListBody({ res, r, meta, L, onOpen, metric, setMetric, ...shown }: BodyProps<ListRow>) {
  const T = meta.targets, t = targetsLine(T), u = L.lists.get(r.list_id), cash = r.bh_cash;
  const B = res.benchmark?.figures ?? null, W = res.windows, rob = r.robustness;
  const holding: Holding = cash ? "cash" : res.benchmark ? "held" : "unkept";
  const vs = (x: string) => (B ? " · B&H " + x : "");
  const title = u?.title ?? r.list_id;
  const how = (W ? `Walk-forward out-of-sample: ${W.w.length} windows of ${windowDays(W)} days, each trading the parameters that did best on all the history before it (the first on ${W.train} days)`
    : "One configuration fixed before the test, nothing fitted: the whole record is out-of-sample")
    + `. One book of ${title}${u?.size ? ` (${sizeShort(u)})` : ""}, ${r.start} → ${r.end}, ${yearsOf(r.start!, r.end!)} years`
    + (u?.costs ? `; costs ${u.costs} a side` : "") + fillText(res.fill) + ".";
  return (
    <>
      <p className="rule">{res.description}</p>
      <p className="sub">{how}</p>
      <div className="tiles">
        <Tile k="Avg / month" v={pct(r.avg_monthly)} s={t.avg + vs(pct(B?.avg_monthly))} ok={target(r, "avg_monthly")} title="the account's growth a month, compounded" />
        <Tile k="Months +" v={pct0(r.green)} s={t.green} ok={target(r, "green")} title="share of the months with a position that ended in profit" />
        <Tile k="Max drawdown" v={pct(r.max_dd, 1)} s={t.dd + vs(pct(B?.max_dd, 1))} ok={target(r, "max_dd")} />
        <Tile k="Sharpe" v={n2(r.sharpe)} s={t.sharpe + vs(n2(B?.sharpe))} ok={target(r, "sharpe")} title="out-of-sample (walk-forward)" />
        {cash ? <Tile k="vs cash / yr" v={pct(r.vs_bh, 1)} s="T-bills · target > 0" ok={r.beats_bh} title={"return a year above cash; " + CASH_TIP} />
          : <Tile k="vs B&H / yr" v={pct(r.vs_bh, 1)} s="at equal risk · target > 0" ok={r.beats_bh}
              title="return a year above holding the same list, the strategy sized to its risk from the last 90 days (at most 2x) with its idle cash at T-bills" />}
        <Tile k="Return / yr" v={pct(r.cagr, 1)} s={B ? `B&H ${pct(B.cagr, 1)}` : ""} title="compounded annual growth" />
        <Tile k="$10k became" v={money(meta.capital * (res.curve.growth ?? NaN))} s={res.benchmark ? `B&H ${money(meta.capital * (res.benchmark.growth ?? NaN))}` : ""} />
        <Tile k="Robustness" v={robMeasured(rob) ? `${rob!.passed}/${rob!.applicable}` : "—"} ok={robFull(rob)}
          s={!rob ? "loses money: not checked" : robMeasured(rob) ? "checks passed" + (rob.not_computed ? ` · ${rob.not_computed} not measured` : "") : "not measured: older code"} />
      </div>
      <div className="grid-card">
        <Charts res={res} name={r.strategy} heldName={res.benchmark ? title : null} maxDD={T.max_dd} {...shown} />
        <MetricsCard o={res.figures} B={B} I={res.index?.figures ?? null} idx={res.index?.name ?? null} tf={r.timeframe} dd={T.max_dd} holding={holding} what="the same list" />
      </div>
      <div className="grid2" style={{ marginTop: 10 }}>
        <RobCard rob={rob} loses={r.loses} labels={L.checks} />
        <ElseCard res={res} lists={meta.lists.filter((x) => x.research)} metric={metric} setMetric={setMetric} onOpen={onOpen} aloneLabel={null} />
      </div>
      <Calendar m={res.curve.months} bh={res.benchmark?.months ?? null} what="the same list" />
      <div className="grid2" style={{ marginTop: 10 }}>
        <ParamCard f={W} grid={res.grid} fixed={res.params} end={r.end} />
        <NamesCard names={res.names} target={T.sharpe} onOpen={onOpen} />
      </div>
      {res.notes.length > 0 && <p className="cap" style={{ marginTop: 12 }}><b>Caveats:</b> {res.notes.join(" · ")}</p>}
    </>
  );
}

function AssetBody({ res, r, meta, L, onOpen, metric, setMetric, ...shown }: BodyProps<AssetRow>) {
  const T = meta.targets, t = targetsLine(T), label = labelOf(r, L), cash = r.bh_cash, pick = res.kind === "pick";
  const W = res.windows, B = res.benchmark?.figures ?? null, rob = r.robustness ?? null;
  const holding: Holding = cash ? "cash" : res.benchmark ? "held" : "unkept";
  // alone, a strategy has no seats to share: its grid is what the walk-forward chose from
  const fitted = W != null || Object.values(res.grid ?? {}).some((v) => v.length > 1);
  const vs = (x: string) => (cash ? "" : " · B&H " + x);
  const now = W ? String(W.sets[W.w[W.w.length - 1][1]].strategy ?? "") : "";
  const how = pick
    ? `The strategy held on ${label} is re-chosen every ${W ? windowDays(W) : 91} days among ${res.candidates.length} strategies run on it alone: the one with the best Sharpe on all the history before that day; out-of-sample ${r.start} → ${r.end}, ${r.months} months, against ${cash ? "cash: " + label + " cannot be held" : "holding " + label}${fillText(res.fill)}.`
    : `One strategy with all the capital on ${label}, as the firm's products run it: ${fitted ? "walk-forward on its own past, each window trading the parameters that did best on all the history before it" : "one configuration fixed before the test, nothing fitted"}; out-of-sample ${r.start} → ${r.end}, ${r.months} months, against ${cash ? "cash: " + label + " cannot be held" : "holding " + label}${fillText(res.fill)}. Parameters${fitted ? " now" : ""}: ${paramText(r.params_now)}.`;
  return (
    <>
      <p className="rule">{res.description}</p>
      {pick && W && (
        <p className="now">Now: <b>{now || "no position"}</b> <span className="muted">since {W.w[since(W.w)][0]} · chosen in {W.w.filter((x) => x[1] === W.w[W.w.length - 1][1]).length} of {W.w.length} windows</span></p>
      )}
      <p className="sub">{how}</p>
      <div className="tiles">
        <Tile k="Avg / month" v={pct(r.avg_monthly)} s={t.avg + vs(pct(B?.avg_monthly ?? r.bh_avg_monthly))} ok={target(r, "avg_monthly")} title="the account's growth a month, compounded" />
        <Tile k="Months +" v={pct0(r.green)} s={t.green} ok={target(r, "green")} title="share of the months with a position that ended in profit" />
        <Tile k="Max drawdown" v={pct(r.max_dd, 1)} s={t.dd + vs(pct(B?.max_dd ?? r.bh_max_dd, 1))} ok={target(r, "max_dd")} />
        <Tile k="Sharpe" v={n2(r.sharpe)} s={t.sharpe + vs(n2(B?.sharpe ?? r.bh_sharpe))} ok={target(r, "sharpe")} title="out-of-sample (walk-forward)" />
        {cash ? <Tile k="vs cash / yr" v={pct(r.vs_bh, 1)} s="T-bills · target > 0" ok={r.beats_bh} title={"return a year above cash; " + CASH_TIP} />
          : <Tile k="vs its B&H / yr" v={pct(r.vs_bh, 1)} s="at equal risk · target > 0" ok={r.beats_bh}
              title={`return a year above holding ${label}, the strategy sized to its risk from the last 90 days (at most 2x) with its idle cash at T-bills`} />}
        <Tile k="Return / yr" v={pct(r.cagr, 1)} s={`${r.months} months out-of-sample`} title="compounded annual growth" />
        <Tile k="$10k became" v={money(meta.capital * (res.curve.growth ?? NaN))} s={res.benchmark ? `B&H ${money(meta.capital * (res.benchmark.growth ?? NaN))}` : ""} />
        {pick ? <Tile k="Beyond luck" v={pct0(res.beyond_luck)} s="of the instruments' picks looked at together"
                  title="the probability that this record is better than the best of as many worthless picks as the instruments' records add up to (deflated Sharpe)" />
          : <Tile k="Robustness" v={robMeasured(rob) ? `${rob!.passed}/${rob!.applicable}` : "—"} ok={robFull(rob)}
              s={!rob ? "loses money: not checked" : robMeasured(rob) ? "checks passed" + (rob.not_computed ? ` · ${rob.not_computed} not measured` : "") : "not measured: older code"} />}
      </div>
      <div className="grid-card">
        <Charts res={res} name={r.strategy} heldName={res.benchmark ? label : null} maxDD={T.max_dd} {...shown} />
        <MetricsCard o={res.figures} B={B} I={res.index?.figures ?? null} idx={res.index?.name ?? null} tf={r.timeframe} dd={T.max_dd} holding={holding} what={label} />
      </div>
      <div className="grid2" style={{ marginTop: 10 }}>
        {pick ? null : <RobCard rob={rob} loses={r.loses} labels={L.checks} />}
        <ElseCard res={res} lists={meta.lists.filter((x) => x.research)} metric={metric} setMetric={setMetric} onOpen={onOpen} aloneLabel={label} />
        {pick && <CandidatesCard items={res.candidates} onOpen={onOpen} />}
      </div>
      <Calendar m={res.curve.months} bh={res.benchmark?.months ?? null} what={label} />
      <div style={{ marginTop: 10 }}><ParamCard f={W} grid={res.grid} fixed={res.params} end={r.end} strategy={pick} /></div>
      {res.notes.length > 0 && <p className="cap" style={{ marginTop: 12 }}><b>Caveats:</b> {res.notes.join(" · ")}</p>}
    </>
  );
}

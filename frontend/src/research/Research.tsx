// Research: every strategy's results, on a list as one book or on one asset alone, against the firm's target. Two
// views on one page, as in the mockup: the filters, the ticked rows' cumulative P&L, the table; a row opens its result.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAssetRows, useListRows } from "../api/queries";
import type { AssetRow, ListRow, Meta, Row } from "../api/types";
import { count, pct, pct0 } from "../format";
import { useToast } from "../ui/Toast";
import type { Filters as F } from "../url";
import { Filters, filtersKey } from "./Filters";
import { PAGE, SLOTS, TOP, labelOf, marketOf, verdict, type Lookup, type Slot } from "./model";
import { Pnl } from "./Pnl";
import { Table, sorted, type Sort } from "./Table";

type Props = { meta: Meta; L: Lookup; f: F; setF: (f: F) => void; onOpen: (key: string) => void; highlight: string | null;
  onTableRows: (keys: string[]) => void };

const bySharpe = (a: Row, b: Row) => b.targets_met - a.targets_met || (b.sharpe ?? -Infinity) - (a.sharpe ?? -Infinity);
// a name as the filter reads it: lower case, "_" and "-" as spaces ("ibs ml" finds ibs_ml_filter)
const words = (s: string) => s.toLowerCase().replace(/[_\s-]+/g, " ").trim();

export function Research({ meta, L, f, setF, onOpen, highlight, onTableRows }: Props) {
  const AM = f.mode === "assets";
  const lists = useListRows(!AM), assets = useAssetRows(AM);
  const q = AM ? assets : lists;
  const all: Row[] = useMemo(() => q.data?.rows ?? [], [q.data]);
  const toast = useToast();
  const A = AM && f.asset ? L.assets.get(f.asset) : undefined;

  // the filter above the table: every word it holds is in the row's strategy or, alone, in its asset's name
  const found = useCallback((r: Row) => {
    const needles = words(f.text).split(" ").filter(Boolean);
    if (!needles.length) return true;
    const hay = words("instrument_id" in r ? `${r.strategy} ${labelOf(r, L)} ${r.instrument_id}` : r.strategy);
    return needles.every((n) => hay.includes(n));
  }, [f.text, L]);
  const match = useCallback((r: AssetRow) =>
    (f.asset ? r.instrument_id === f.asset : f.cls === "All" || marketOf(r, L) === f.cls) && (f.tf === "All" || r.timeframe === f.tf)
    && (f.ver === "All" || verdict(r) === f.ver) && !(f.hideStale && r.stale) && found(r), [f, L, found]);
  const cur = useMemo<Row[]>(() => AM
    ? (all as AssetRow[]).filter((r) => match(r) && (r.months ?? 0) >= f.min).toSorted(bySharpe)
    : (all as ListRow[]).filter((r) => (f.cls === "All" || marketOf(r, L) === f.cls) && (f.uni === "All" || r.list_id === f.uni)
        && (f.tf === "All" || r.timeframe === f.tf) && !(f.hideStale && r.stale) && found(r)), [AM, all, f, L, match, found]);
  const shorter = useMemo(() => (AM ? (all as AssetRow[]).filter((r) => match(r) && (r.months ?? 0) < f.min).length : 0),
    [AM, all, f.min, match]);

  const [sort, setSort] = useState<Sort>(null);
  const [shown, setShown] = useState(PAGE);
  const [ticks, setTicks] = useState<Map<string, Slot>>(new Map());
  const [showBH, setShowBH] = useState(true);
  const [showIdx, setShowIdx] = useState(true);
  // new filters: the table's own order and first page again, and the chart's lines the sheet's top 5 (a sort does not
  // change them), once the rows are there
  const key = filtersKey(f);
  const [seen, setSeen] = useState<string | null>(null);
  if (seen !== key && !q.isPending) {
    setSeen(key);
    setSort(null);
    setShown(PAGE);
    setTicks(new Map(cur.slice(0, TOP).map((r, i) => [r.key, SLOTS[i]])));
  }
  const ticksNow = useRef(ticks);
  useEffect(() => {
    ticksNow.current = ticks;
  });
  const onTick = useCallback((k: string) => {
    if (!ticksNow.current.has(k) && ticksNow.current.size >= SLOTS.length) {
      toast(`At most ${SLOTS.length} lines on the chart`);
      return;
    }
    setTicks((t) => {
      const next = new Map(t);
      if (next.has(k)) next.delete(k);
      else {
        const free = SLOTS.find((s) => ![...next.values()].includes(s));   // a line keeps its colour while it stays on
        if (free) next.set(k, free);
      }
      return next;
    });
  }, [toast]);
  const rows = useMemo(() => sorted(cur, sort, L), [cur, sort, L]);
  useEffect(() => onTableRows(rows.map((r) => r.key)), [rows, onTableRows]);

  const T = meta.targets;
  const bySymbol = meta.strategies.assets.filter((s) => !meta.strategies.lists.includes(s));
  const total = all.length;
  const counted = AM
    ? (A ? `${rows.length} results for ${A.label} alone` : `${count(rows.length)} results`)
      + (shorter ? ` · ${count(shorter)} more with fewer than ${f.min} OOS months` : "")
    : `${rows.length} results` + (rows.length !== total ? ` (filtered from ${total})` : "");
  return (
    <>
      <h1>
        Research{" "}
        <span className="muted" style={{ fontWeight: 400 }}
          title={`${meta.strategies.lists.length} run on the lists` + (bySymbol.length ? `; ${bySymbol.join(", ")} only symbol by symbol` : "")}>
          · {meta.strategies.lists.length + bySymbol.length} strategies
        </span>
      </h1>
      <div className="targetline">
        <span>Target:</span><b>avg month ≥ {pct(T.avg_monthly, 1, false)}</b><span>(stretch {pct(T.avg_monthly_stretch, 0, false)})</span>
        <b>green months ≥ {pct0(T.pct_green)}</b><span>(plan {pct0(T.pct_green_stretch)})</span><b>max DD ≥ {pct(T.max_dd, 0)}</b>
        <b>Sharpe ≥ {T.sharpe}</b><b>beats buy &amp; hold</b><span>· no leverage</span>
      </div>
      <Filters meta={meta} L={L} f={f} set={setF} />
      <Pnl assets={AM} rows={all} ticks={ticks} L={L} showBH={showBH} setShowBH={setShowBH} showIdx={showIdx} setShowIdx={setShowIdx} />
      <div className="counthead">
        <input type="search" className="search tablefilter" value={f.text} onChange={(e) => setF({ ...f, text: e.target.value })}
          placeholder={AM ? "Filter by strategy or asset" : "Filter by strategy"} aria-label="filter the results by name"
          title={AM ? "every word in the strategy's or the asset's name, e.g. \"rsi2 btc\"" : "every word in the strategy's name, e.g. \"ibs ml\""} />
        <h2 id="bcount">{q.isPending ? "Loading…" : q.error ? `Could not load the results: ${q.error.message}` : counted}</h2>
      </div>
      <div className="card" style={{ padding: 0 }}>
        <div className="scroll" style={{ maxHeight: 640 }}>
          <Table rows={rows} assets={AM} every={AM && !A} L={L} ticks={ticks} sort={sort} setSort={setSort} shown={shown}
            more={() => setShown((s) => s + PAGE)} onTick={onTick} onOpen={onOpen} highlight={highlight} />
        </div>
      </div>
    </>
  );
}

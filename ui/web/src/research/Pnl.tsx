// Research's cumulative P&L: the ticked rows' out-of-sample records from the first day they all have, with the buy &
// hold they were compared with (the one of the line that starts last, on the first line's list; on one asset, that
// asset's), dashed, and their market's index over the same days (that of the line that starts last), dotted.
import type { AssetRow, Point, Row } from "../api/types";
import { useCurves } from "../api/queries";
import { TimeChart, type Line } from "../charts/TimeChart";
import { END_CHARS, bisect } from "../charts/scale";
import { cumPct, ymd } from "../format";
import { CASH_TIP, INDEX_LINE, INDEX_TIP, SLOTS, TOP, labelOf, marketOf, type Lookup, type Slot } from "./model";

type Props = { assets: boolean; rows: Row[]; ticks: Map<string, Slot>; L: Lookup; showBH: boolean; setShowBH: (v: boolean) => void;
  showIdx: boolean; setShowIdx: (v: boolean) => void };
type Drawn = { r: Row; slot: Slot; pts: Point[]; bench: Point[] | null; index: Point[] | null; indexName: string | null };
const MONTH_MS = 31 * 864e5;           // lines that start further apart than this have their common start named

export function Pnl({ assets, rows, ticks, L, showBH, setShowBH, showIdx, setShowIdx }: Props) {
  const curves = useCurves([...ticks.keys()]);
  const byKey = new Map(rows.map((r) => [r.key, r]));
  const drawn: Drawn[] = [...ticks.entries()].flatMap(([key, slot]) => {
    const r = byKey.get(key), c = curves.data?.curves[key];
    return r && c && c.pts.length ? [{ r, slot, pts: c.pts as Point[], bench: (c.benchmark ?? null) as Point[] | null,
      index: (c.index ?? null) as Point[] | null, indexName: c.index_name ?? null }] : [];
  });
  const inst = (x: Drawn) => (assets ? (x.r as AssetRow).instrument_id : "");
  const insts = new Set(drawn.map(inst));
  const sheet = (x: Drawn) => x.r.list_id + "@" + x.r.timeframe;
  // a line's name: its strategy and, where the lines differ, what tells them apart: the asset, the list, the timeframe
  const varies = (f: (x: Drawn) => string) => new Set(drawn.map(f)).size > 1;
  const byList = assets ? new Set(drawn.map((x) => inst(x) + "@" + x.r.list_id)).size > insts.size : varies((x) => x.r.list_id);
  const titled = !assets && varies((x) => marketOf(x.r, L));    // lists of several markets: each named with its market
  const listName = (r: Row) => {
    const u = L.lists.get(r.list_id);
    return u ? (titled ? u.title : u.label) : r.list_id;
  };
  const byTf = varies((x) => x.r.timeframe);
  const parts = (x: Drawn) => [insts.size > 1 ? labelOf(x.r as AssetRow, L) : "", x.r.strategy, byList ? listName(x.r) : "",
    byTf ? x.r.timeframe : ""].filter(Boolean);
  // at the line's end the strategy gives up letters first: the rest is what tells lines of one strategy apart
  const short = (x: Drawn) => {
    const p = parts(x), i = p.indexOf(x.r.strategy), s = x.r.strategy, over = p.join(" · ").length - END_CHARS;
    const keep = Math.max(5, s.length - over - 1);
    if (over > 0 && keep < s.length) p[i] = s.slice(0, keep) + "…";
    return p.join(" · ");
  };
  // several assets have no buy & hold in common: the tick box waits until every line is on one asset
  const common = !assets || insts.size <= 1, cash = common && drawn.length > 0 && drawn[0].r.bh_cash;
  const unkept = common && drawn.length > 0 && !cash && drawn.every((x) => !x.bench);
  const off = !common || cash || unkept;
  const why = !common ? "drawn when every line is on one asset: several assets have no buy & hold in common"
    : cash ? CASH_TIP
    : unkept ? "no buy & hold series is kept for results imported from the files before the database: a re-run keeps it" : "";
  let bh: Point[] | null = null, bhName = "Buy & hold";
  if (showBH && drawn.length && !cash) {
    const pool = assets ? drawn : drawn.filter((x) => sheet(x) === sheet(drawn[0]));
    const late = pool.reduce((a, x) => (x.pts[0][0] > a.pts[0][0] ? x : a));
    if (late.bench?.length && (!assets || insts.size === 1)) {
      bh = late.bench;
      if (assets) bhName = labelOf(late.r as AssetRow, L) + " buy & hold";
      else if (byList) bhName = listName(late.r) + " buy & hold";
    }
  }
  // the market's index: each list's market has its own and an asset alone its own market's, so it is drawn when every
  // line has the same one
  const indexes = new Set(drawn.map((x) => x.indexName)), indexName = indexes.size === 1 ? drawn[0].indexName : null;
  const idxWhy = !drawn.length ? "" : indexName ? INDEX_TIP
    : indexes.size > 1 ? "drawn when every line has the same one: each market has its own index, and an asset alone its own market's"
    : "none beside these lines: FX is compared with cash, an asset that is its market's index is its own buy & hold, and a metal, " +
      "a future or an ETF of bonds, commodities or foreign stocks has its own buy & hold only";
  let ix: Point[] | null = null;
  if (showIdx && indexName) ix = drawn.reduce((a, x) => (x.pts[0][0] > a.pts[0][0] ? x : a)).index;
  const all = drawn.map((x) => x.pts).concat(bh ? [bh] : [], ix ? [ix] : []);
  const t0 = all.length ? Math.max(...all.map((p) => p[0][0])) : 0;          // the first date every line has
  const first = drawn.length ? Math.min(...drawn.map((x) => x.pts[0][0])) : 0;
  let chart;
  if (!all.length) {
    chart = <p className="muted" style={{ padding: "40px 0", textAlign: "center" }}>{curves.isFetching ? "loading…" : "tick a row below to draw it"}</p>;
  } else {
    const rebase = (pts: Point[]): Point[] => {
      const k = bisect(pts, t0), b = pts[k][1];
      return pts.slice(k).map((p) => [p[0], p[1] / b]);
    };
    const held: Line[] = bh ? [{ name: bhName, color: "--muted", pts: rebase(bh), dash: "4 3", w: 1.5 }] : [];
    const index: Line[] = ix && indexName ? [{ name: indexName, ...INDEX_LINE, pts: rebase(ix) }] : [];
    const series = held.concat(index, drawn.map((x) =>
      ({ name: parts(x).join(" · "), short: short(x), color: x.slot.color, dash: x.slot.dash, pts: rebase(x.pts) })));
    chart = <TimeChart series={series} base={1} fmt={cumPct} height={300} />;
  }
  return (
    <div className="card">
      <div className="pnlhead">
        <h3>Cumulative P&amp;L</h3>
        <span className="cap">
          the top {TOP} on this sheet · tick or untick below to change, up to {SLOTS.length} lines ·{" "}
          {t0 - first > MONTH_MS && <>from {ymd(t0)}, the first day every line has (the earliest from {ymd(first)}) · </>}
          <label className={"toggle" + (off ? " off" : "")} title={why}>
            <input type="checkbox" checked={showBH} disabled={off} onChange={(e) => setShowBH(e.target.checked)} /> Buy &amp; hold
          </label>
          {" · "}
          <label className={"toggle" + (indexName ? "" : " off")} title={idxWhy}>
            <input type="checkbox" checked={showIdx} disabled={!indexName} onChange={(e) => setShowIdx(e.target.checked)} /> {indexName ?? "Index"}
          </label>
        </span>
      </div>
      {chart}
    </div>
  );
}

// A time chart as the mockup drew it: linear levels, years below, each line's name and last value at its end, a
// crosshair with every line's value under the pointer. Lines: [{name, color: '--s1', pts: [[t, v]]}], a dashed one
// with its dash pattern.
import { useLayoutEffect, useMemo, useRef, useState, type PointerEvent, type RefObject } from "react";
import type { Point } from "../api/types";
import { ymd } from "../format";
import { TipRow, useTip } from "../ui/Tip";
import { TICK_GAP, bisect, endName, linTicks, textWidth, tickRank, yearTicks } from "./scale";

// short: what the line's end shows of a long name, where its owner shortens it its own way
export type Line = { name: string; short?: string; color: string; pts: Point[]; dash?: string; w?: number; faint?: boolean };
const id = (s: Line) => s.name + s.color + (s.dash ?? "");

// a line's key beside its name (a table's tick, the tooltip, a legend): its colour, dashed where the line is, dotted
// where its pattern's first stroke is a dot (the market's index beside buy & hold's dashes)
export const swatch = (color: string, dash: boolean | string = false): string =>
  typeof dash === "string" && parseFloat(dash) <= 1 ? `repeating-linear-gradient(90deg, var(${color}) 0 2px, transparent 2px 5px)`
  : dash ? `repeating-linear-gradient(90deg, var(${color}) 0 4px, transparent 4px 6px)` : `var(${color})`;

export type ChartOptions = {
  base?: number;                          // the value the labels read as zero (a growth of 1 is +0%)
  height?: number;
  fmt?: (v: number) => string;
  area?: boolean;
  zero?: boolean;
  endLabels?: boolean;
  refs?: { v: number; label: string }[];  // reference levels: a target
};

function useWidth(ref: RefObject<HTMLDivElement | null>): number {
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(el.clientWidth);
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return w || 600;
}

const ML = 58, MT = 8, MB = 22;
const LABEL_GAP = 14;                   // the rows of the names at the lines' ends
const NO_REFS: { v: number; label: string }[] = [];

export function TimeChart({ series, base = 0, height = 240, fmt = String, area = false, zero = false, endLabels = true, refs = NO_REFS }:
  { series: Line[] } & ChartOptions) {
  const box = useRef<HTMLDivElement>(null);
  const W = useWidth(box), H = height;
  const tip = useTip();
  const [hover, setHover] = useState<{ px: number; t: number } | null>(null);

  const g = useMemo(() => {
    const drawn = series.filter((s) => s.pts.length);
    const all = drawn.flatMap((s) => s.pts);
    if (!all.length) return null;
    const mr = !endLabels ? 12 : 15 + Math.ceil(Math.max(...drawn.map((s) =>
      textWidth(endName(s.short ?? s.name) + " ") + textWidth(fmt(s.pts[s.pts.length - 1][1]), 600))));
    let t0 = Infinity, t1 = -Infinity, lo = Infinity, hi = -Infinity;
    for (const [t, v] of all) {
      if (t < t0) t0 = t;
      if (t > t1) t1 = t;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (zero) hi = Math.max(hi, 0);
    for (const r of refs) {
      lo = Math.min(lo, r.v);
      hi = Math.max(hi, r.v);
    }
    const ylo = lo, yhi = hi === lo ? lo + 1 : hi;
    const X = (t: number) => ML + ((t - t0) / (t1 - t0 || 1)) * (W - ML - mr);
    const Y = (v: number) => MT + (1 - (v - ylo) / (yhi - ylo)) * (H - MT - MB);
    const rank = (v: number) => tickRank(v - base), ticks: number[] = [];
    const levels = linTicks(lo - base, hi - base, 4).map((v) => +(v + base).toFixed(10));
    for (const v of levels.toSorted((p, q) => rank(p) - rank(q))) if (ticks.every((t) => Math.abs(Y(t) - Y(v)) >= TICK_GAP)) ticks.push(v);
    const lines = drawn.map((s) => {
      const step = Math.max(1, Math.floor(s.pts.length / 900));
      const pts = s.pts.filter((_, i) => i % step === 0 || i === s.pts.length - 1);
      const line = pts.map((p) => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join(" ");
      const fill = area ? `${X(pts[0][0]).toFixed(1)},${Y(0)} ${line} ${X(pts[pts.length - 1][0]).toFixed(1)},${Y(0)}` : null;
      return { s, line, fill };
    });
    const ends = drawn.map((s) => {
      const last = s.pts[s.pts.length - 1];
      return { y: Y(last[1]), x: X(last[0]), ly: 0, id: id(s), name: endName(s.short ?? s.name), v: last[1], color: s.color };
    }).toSorted((a, b) => a.y - b.y);
    // the names in the order of the lines' ends, a row apart: pushed down from the top, then up from the plot's floor
    // where the lowest would fall below it
    ends.forEach((e, i) => (e.ly = i ? Math.max(e.y, ends[i - 1].ly + LABEL_GAP) : e.y));
    for (let i = ends.length - 1; i >= 0; i--) ends[i].ly = Math.min(ends[i].ly, i === ends.length - 1 ? H - MB : ends[i + 1].ly - LABEL_GAP);
    return { drawn, mr, t0, t1, X, Y, ticks, lines, ends, years: yearTicks(t0, t1, Math.max(3, Math.floor((W - ML - mr) / 60))) };
  }, [series, W, H, base, area, zero, endLabels, refs, fmt]);

  if (!g) return <div className="chart" ref={box} />;
  const { drawn, mr, t0, t1, X, Y } = g;
  const move = (ev: PointerEvent<SVGRectElement>) => {
    const r = (ev.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const px = ((ev.clientX - r.left) * W) / r.width, t = t0 + ((px - ML) / (W - ML - mr)) * (t1 - t0);
    setHover({ px, t });
    const rows = drawn.filter((s) => t >= s.pts[0][0] && t <= s.pts[s.pts.length - 1][0]).map((s) => {
      const p = s.pts[bisect(s.pts, t)];
      return <TipRow key={id(s)} swatch={swatch(s.color, s.dash)} value={fmt(p[1])} name={s.name} />;
    });
    tip.show(ev, <><div className="t">{ymd(t)}</div>{rows}</>, true);        // rows that never wrap: as wide as the longest
  };
  const leave = () => {
    setHover(null);
    tip.hide();
  };
  return (
    <div className="chart" ref={box}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={drawn.map((s) => s.name).join(", ")}>
        {g.ticks.map((v) => (
          <g key={"y" + v}>
            <line x1={ML} x2={W - mr} y1={Y(v)} y2={Y(v)} style={{ stroke: "var(--grid)" }} />
            <text x={ML - 6} y={Y(v) + 4} textAnchor="end" fontSize="11" style={{ fill: "var(--muted)" }}>{fmt(v)}</text>
          </g>
        ))}
        {g.years.map(([t, lab]) => (
          <text key={"x" + lab} x={X(t)} y={H - 5} textAnchor="middle" fontSize="11" style={{ fill: "var(--muted)" }}>{lab}</text>
        ))}
        {zero && <line x1={ML} x2={W - mr} y1={Y(0)} y2={Y(0)} style={{ stroke: "var(--axis)" }} />}
        {refs.map((r) => (
          <g key={"r" + r.v}>
            <line x1={ML} x2={W - mr} y1={Y(r.v)} y2={Y(r.v)} style={{ stroke: "var(--crit)", strokeOpacity: 0.6 }} />
            <text x={ML + 4} y={Y(r.v) - 4} fontSize="11" style={{ fill: "var(--ink2)" }}>{r.label}</text>
          </g>
        ))}
        {g.lines.map(({ s, line, fill }) => (
          <g key={id(s)}>
            {fill && <polygon points={fill} style={{ fill: `var(${s.color})`, fillOpacity: 0.12 }} />}
            <polyline points={line} style={{
              fill: "none", stroke: `var(${s.color})`, strokeWidth: s.w ?? 2, strokeLinejoin: "round", strokeLinecap: "round",
              strokeDasharray: s.dash, strokeOpacity: s.faint ? 0.55 : undefined,
            }} />
          </g>
        ))}
        {endLabels && g.ends.map((e) => (
          <g key={"e" + e.id}>
            <line x1={e.x + 2} x2={W - mr + 8} y1={e.y} y2={e.ly} style={{ stroke: "var(--axis)" }} />
            <circle cx={e.x} cy={e.y} r="3.5" style={{ fill: `var(${e.color})`, stroke: "var(--surface)", strokeWidth: 2 }} />
            <text x={W - mr + 11} y={e.ly + 4} fontSize="11" style={{ fill: "var(--ink2)" }}>
              {e.name} <tspan style={{ fill: "var(--ink)", fontWeight: 600 }}>{fmt(e.v)}</tspan>
            </text>
          </g>
        ))}
        {hover && (
          <>
            <line x1={hover.px} x2={hover.px} y1={MT} y2={H - MB} style={{ stroke: "var(--muted)" }} />
            {drawn.filter((s) => hover.t >= s.pts[0][0] && hover.t <= s.pts[s.pts.length - 1][0]).map((s) => {
              const p = s.pts[bisect(s.pts, hover.t)];
              return <circle key={id(s)} cx={X(p[0])} cy={Y(p[1])} r="4" style={{ fill: `var(${s.color})`, stroke: "var(--surface)", strokeWidth: 2 }} />;
            })}
          </>
        )}
        <rect x={ML} y={MT} width={Math.max(0, W - ML - mr)} height={H - MT - MB} fill="transparent" onPointerMove={move} onPointerLeave={leave} />
      </svg>
    </div>
  );
}

// items: [colour, name, dashed (or the line's dash pattern)]
export const Legend = ({ items, box = false }: { items: [string, string, (boolean | string)?][]; box?: boolean }) => (
  <div className="legend">
    {items.map(([c, n, dash]) => (
      <span key={c + n}><i className={box ? "box" : ""} style={{ background: swatch(c, dash) }} />{n}</span>
    ))}
  </div>
);

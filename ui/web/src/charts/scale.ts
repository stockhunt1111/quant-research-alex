// A chart's levels and years, as the mockup placed them: levels step evenly (1, 2 or 5 times a power of ten) from the
// value its labels read as zero; where two would sit closer than two label heights the rounder one stays.
import type { Point } from "../api/types";

export function niceStep(span: number, n: number): number {
  const raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}

export function linTicks(lo: number, hi: number, n = 5): number[] {
  const s = niceStep(hi - lo || 1, n), out: number[] = [];
  for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-12; v += s) out.push(+v.toFixed(10));
  return out;
}

export const TICK_GAP = 24;
// zero first, then 1 x 10^k, then 2 and 5, then the rest
export function tickRank(v: number): number {
  if (v === 0) return -1;
  const a = Math.abs(v), m = +(a / Math.pow(10, Math.floor(Math.log10(a)))).toPrecision(6);
  return m === 1 ? 0 : m === 2 || m === 5 ? 1 : 2;
}

export function yearTicks(t0: number, t1: number, maxN: number): [number, string][] {
  const y0 = new Date(t0).getUTCFullYear() + 1, y1 = new Date(t1).getUTCFullYear();
  let step = 1;
  while ((y1 - y0) / step > maxN) step = step === 1 ? 2 : step === 2 ? 5 : step * 2;
  const out: [number, string][] = [];
  for (let y = y0; y <= y1; y++) if (y % step === 0) out.push([Date.UTC(y, 0, 1), String(y)]);
  return out;
}

// the point nearest to time t
export function bisect(pts: Point[], t: number): number {
  let lo = 0, hi = pts.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (pts[m][0] <= t) lo = m;
    else hi = m;
  }
  return Math.abs(pts[lo][0] - t) <= Math.abs(pts[hi][0] - t) ? lo : hi;
}

let measure: CanvasRenderingContext2D | null = null;
// the on-screen width of an 11px chart label
export function textWidth(text: string, weight = 400): number {
  measure ??= document.createElement("canvas").getContext("2d");
  if (!measure) return text.length * 6.5;
  measure.font = `${weight} 11px ${getComputedStyle(document.body).fontFamily}`;
  return measure.measureText(text).width;
}

export const END_CHARS = 28;
// a line's name at its end, at most END_CHARS: letters come off the longest part of a name ("ibs · Top-10 · 1d") one at
// a time, so that its short parts stay whole; a name whose longest part is down to three letters is cut at its end
export function endName(n: string): string {
  const parts = n.split(" · "), keep = parts.map((p) => p.length);
  const shown = () => parts.map((p, i) => (keep[i] < p.length ? p.slice(0, keep[i]) + "…" : p));
  for (let s = shown(); s.join(" · ").length > END_CHARS; s = shown()) {
    const i = s.reduce((m, p, j) => (p.length > s[m].length ? j : m), 0);
    if (keep[i] <= 3) return n.slice(0, END_CHARS - 1) + "…";
    keep[i]--;
  }
  return shown().join(" · ");
}

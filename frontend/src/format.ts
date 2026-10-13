// Numbers as the mockup wrote them: returns in percent with a sign, Sharpe with two decimals, money in $k / $M.
import type { CSSProperties } from "react";

export const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
export const pct = (v: number | null | undefined, d = 2, sign = true): string =>
  !isNum(v) ? "—" : (sign && v > 0 ? "+" : "") + (v * 100).toFixed(d) + "%";
export const pct0 = (v: number | null | undefined): string => (!isNum(v) ? "—" : Math.round(v * 100) + "%");
export const n2 = (v: number | null | undefined): string => (!isNum(v) ? "—" : v.toFixed(2));
export const money = (v: number | null | undefined): string =>
  !isNum(v) ? "—" : Math.abs(v) >= 1e6 ? "$" + (v / 1e6).toFixed(2) + "M" : "$" + (v / 1e3).toFixed(v >= 1e5 ? 0 : 1) + "k";
export const count = (v: number): string => v.toLocaleString("en-US");
export const ymd = (t: number): string => new Date(t).toISOString().slice(0, 10);
// a growth of 1 as the cumulative return: 1.5 -> +50%
export const cumPct = (v: number): string => (v >= 1 ? "+" : "") + Math.round((v - 1) * 100).toLocaleString("en-US") + "%";
// a cell's background from blue (up) to red (down), full at |v| >= max
export const divBg = (v: number | null | undefined, max: number): CSSProperties =>
  !isNum(v)
    ? {}
    : { background: `color-mix(in oklab, var(${v >= 0 ? "--pos" : "--neg"}) ${Math.round(Math.min(Math.abs(v) / max, 1) * 55)}%, var(--mid))` };
export const yearsOf = (a: string, b: string): string => ((Date.parse(b) - Date.parse(a)) / 864e5 / 365.25).toFixed(1);
export const daysOf = (v: number | null | undefined): string => (isNum(v) ? v.toLocaleString("en-US") + " days" : "—");
export const monthsOf = (v: number | null | undefined): string => (isNum(v) ? v + (v === 1 ? " month" : " months") : "—");
export const windowsOf = (v: number): string => v + (v === 1 ? " window" : " windows");
// a span of seconds as "1h 20m", "12m", "40s"
export function span(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return s + "s";
  const m = Math.round(s / 60);
  if (m < 60) return m + "m";
  const h = Math.floor(m / 60);
  return h + "h" + (m % 60 ? " " + (m % 60) + "m" : "");
}
// a moment as the page's header shows it: 27.09 14:03, this machine's clock
export function moment(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  const two = (n: number) => String(n).padStart(2, "0");
  return `${two(d.getDate())}.${two(d.getMonth() + 1)} ${two(d.getHours())}:${two(d.getMinutes())}`;
}

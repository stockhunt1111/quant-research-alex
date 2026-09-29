// The page's state in its address, so a reload or a link opens the same view: the filters, the open result, the theme.
export type Mode = "lists" | "assets";
export type Filters = {
  mode: Mode;
  cls: string;          // a market, or All
  uni: string;          // a list (Lists view), or All
  asset: string | null; // an instrument (Assets view)
  tf: string;           // 1h, 4h, 1d or All
  ver: string;          // a single asset's verdict, or All
  min: number;          // the fewest out-of-sample months a single asset's record may have
  hideStale: boolean;
  text: string;         // the words a row's strategy (or, alone, its asset) must hold
};
export const DEFAULTS: Filters = { mode: "lists", cls: "All", uni: "All", asset: null, tf: "All", ver: "All", min: 24, hideStale: false,
  text: "" };
export const MIN_MONTHS = [6, 24, 60];

export function readUrl(): { filters: Filters; result: string | null; theme: string | null } {
  const q = new URLSearchParams(location.search);
  const f: Filters = { ...DEFAULTS };
  if (q.get("mode") === "assets") f.mode = "assets";
  if (q.get("cls")) f.cls = q.get("cls")!;
  if (q.get("uni")) f.uni = q.get("uni")!;
  if (q.get("asset")) {
    f.mode = "assets";
    f.asset = q.get("asset");
  }
  if (q.get("tf")) f.tf = q.get("tf")!;
  if (q.get("ver")) f.ver = q.get("ver")!;
  const min = Number(q.get("min"));
  if (MIN_MONTHS.includes(min)) f.min = min;
  f.hideStale = q.get("stale") === "hide";
  f.text = q.get("q") ?? "";
  const theme = q.get("theme");
  return { filters: f, result: q.get("result"), theme: theme === "dark" || theme === "light" ? theme : null };
}

export function writeUrl(f: Filters, result: string | null, theme: string | null): void {
  const q = new URLSearchParams();
  if (theme) q.set("theme", theme);
  if (f.mode !== DEFAULTS.mode) q.set("mode", f.mode);
  if (f.cls !== DEFAULTS.cls) q.set("cls", f.cls);
  if (f.mode === "lists" && f.uni !== DEFAULTS.uni) q.set("uni", f.uni);
  if (f.mode === "assets" && f.asset) q.set("asset", f.asset);
  if (f.tf !== DEFAULTS.tf) q.set("tf", f.tf);
  if (f.mode === "assets" && f.ver !== DEFAULTS.ver) q.set("ver", f.ver);
  if (f.mode === "assets" && f.min !== DEFAULTS.min) q.set("min", String(f.min));
  if (f.hideStale) q.set("stale", "hide");
  if (f.text.trim()) q.set("q", f.text.trim());
  if (result) q.set("result", result);
  const s = q.toString();
  const url = location.pathname + (s ? "?" + s : "");
  if (url !== location.pathname + location.search) history.replaceState(null, "", url);
}

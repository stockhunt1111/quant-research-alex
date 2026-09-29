// Every view of the page rendered to HTML in node from a running server's answers: a component that fails to render,
// a section missing from a popup or a word that is not English shows here before a browser shows it.
//
//     npm run check:render -- http://127.0.0.1:8600          (tests/test_server.py runs it on the fixture results)
/* oxlint-disable no-await-in-loop -- the views render one at a time: each sets the address the page reads */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderToString } from "react-dom/server";
import { App } from "../src/App";
import type { AssetRow, ListRow, Meta, RunCurrent } from "../src/api/types";
import { TipProvider } from "../src/ui/Tip";
import { ToastProvider } from "../src/ui/Toast";

type Node = { process: { argv: string[]; exitCode?: number } };
const node = globalThis as unknown as Node & Record<string, unknown>;
const base = node.process.argv[2] ?? "http://127.0.0.1:8600";

// what the page reads of the browser while it renders (its effects do not run here)
node.history = { replaceState() {} };
node.matchMedia = () => ({ matches: false });
node.document = { createElement: () => ({ getContext: () => null }), body: {}, documentElement: { dataset: {} } };
node.addEventListener = () => {};
node.removeEventListener = () => {};

async function get<T>(url: string): Promise<T> {
  const r = await fetch(base + url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return (await r.json()) as T;
}

const meta = await get<Meta>("/api/research/meta");
const run = await get<RunCurrent>("/api/runs/current");
const lists = await get<{ rows: ListRow[] }>("/api/research/lists");
const assets = await get<{ rows: AssetRow[] }>("/api/research/assets");
const problems: string[] = [];

async function page(search: string, result: string | null, wants: string[], rowsWith?: string): Promise<void> {
  node.location = { search, pathname: "/" };
  const qc = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity, retry: false } } });
  qc.setQueryData(["meta"], meta);
  qc.setQueryData(["run"], run);
  qc.setQueryData(["rows", "lists"], lists);
  qc.setQueryData(["rows", "assets"], assets);
  if (result) qc.setQueryData(["result", result], await get(`/api/research/result?key=${encodeURIComponent(result)}`));
  let html: string;
  try {
    html = renderToString(
      <QueryClientProvider client={qc}><TipProvider><ToastProvider><App /></ToastProvider></TipProvider></QueryClientProvider>);
  } catch (e) {
    problems.push(`${search}: rendering threw ${e instanceof Error ? e.stack : e}`);
    return;
  }
  const missing = wants.filter((w) => !html.includes(w));
  const keys = [...html.matchAll(/<tr data-key="([^"]+)"/g)].map((m) => m[1]);
  if (rowsWith !== undefined && (!keys.length || keys.some((k) => !k.toLowerCase().includes(rowsWith.toLowerCase()))))
    problems.push(`${search}: ${keys.length} rows, not all of them ${rowsWith}'s`);
  const foreign = html.match(/[Ѐ-ӿ]+/g);
  if (missing.length) problems.push(`${search}: no ${missing.join(", ")}`);
  if (foreign) problems.push(`${search}: not English: ${foreign.slice(0, 5).join(" ")}`);
  console.log(`${(search || "(Lists)").padEnd(90)} ${String(Math.round(html.length / 1024)).padStart(5)} KB, ` +
    `${(html.match(/<tr data-key/g) ?? []).length} rows`);
}

const q = (k: string) => encodeURIComponent(k);
await page("", null, ["Research", `${lists.rows.length} results`, "K-ratio", "Robustness", "Filter by strategy"]);
await page("?mode=assets", null, ["Research", "OOS months", "K-ratio", "Filter by strategy or asset"]);
const some = lists.rows[0].strategy.split("_")[0];            // a word of a strategy's name: fewer rows, each holding it
await page(`?q=${encodeURIComponent(some)}`, null, [`(filtered from ${lists.rows.length})`], some);
const asset = meta.assets.find((a) => assets.rows.some((r) => r.instrument_id === a.id && (r.months ?? 0) >= 24));
if (asset) await page(`?mode=assets&q=${encodeURIComponent(asset.label)}`, null, [], asset.id);
const listPicks = [lists.rows[0], lists.rows.find((r) => !r.robustness), lists.rows.find((r) => r.bh_cash)];
for (const r of listPicks.filter((x): x is ListRow => x != null)) {
  await page(`?result=${q(r.key)}`, r.key, ["Performance metrics", "Robustness", "Same strategy elsewhere", "Calendar months",
    "Asset by asset", "Cumulative P&amp;L", "Drawdown", 'aria-sort="descending"', "▼"]);
}
const assetPicks = [assets.rows.find((r) => r.strategy !== "strategy_pick"), assets.rows.find((r) => r.held),
  assets.rows.find((r) => r.strategy === "strategy_pick")];
for (const r of assetPicks.filter((x): x is AssetRow => x != null)) {
  const pick = r.strategy === "strategy_pick";
  await page(`?mode=assets&asset=${q(r.instrument_id)}&result=${q(r.key)}`, r.key,
    ["Performance metrics", "Same strategy elsewhere", pick ? "Candidates" : "Parameters", ...(pick ? ["Now:"] : [])]);
}
if (problems.length) {
  console.error(problems.join("\n"));
  node.process.exitCode = 1;
} else console.log("every view rendered");

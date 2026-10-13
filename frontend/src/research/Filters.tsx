// The filters above Research's table: which view, market, list or asset, timeframe, record length, verdict, stale.
import { useState } from "react";
import type { Meta } from "../api/types";
import { useToast } from "../ui/Toast";
import { DEFAULTS, MIN_MONTHS, type Filters as F } from "../url";
import { QUICK, VERDICTS, sizeLong, type Lookup } from "./model";

type Props = { meta: Meta; L: Lookup; f: F; set: (f: F) => void };

export function Filters({ meta, L, f, set }: Props) {
  const toast = useToast();
  const [typed, setTyped] = useState("");
  const AM = f.mode === "assets";
  const markets = ["All", ...meta.markets];
  const chip = (on: boolean, label: string, click: () => void, title = "") => (
    <button key={label + title} className={"chip" + (on ? " on" : "")} title={title} onClick={click}>{label}</button>
  );
  const research = meta.lists.filter((u) => u.research);
  const listChips = (market: string) =>
    research.filter((u) => u.market === market).map((u) => chip(f.uni === u.id, u.label, () => set({ ...f, uni: u.id }), sizeLong(u)));
  const pool = meta.assets.filter((a) => f.cls === "All" || a.market === f.cls);
  const quick = f.cls === "All" ? [] : pool.slice(0, QUICK);
  const picked = f.asset ? L.assets.get(f.asset) : undefined;
  const pick = (text: string) => {
    const a = L.byLabel.get(text.trim().toLowerCase());
    if (a) {
      set({ ...f, asset: a.id, cls: a.market });
      setTyped("");
    } else if (text.trim()) toast(`No results for "${text}"`);
  };
  return (
    <div className="filters col" id="bfilters">
      <div className="frow">
        <span className="lbl">Run on</span>
        {chip(!AM, "Lists", () => set({ ...f, mode: "lists", asset: null, uni: "All", ver: "All" }), "a strategy on a whole list as one book")}
        {chip(AM, "Assets", () => set({ ...f, mode: "assets", asset: null, uni: "All", ver: "All" }),
          "a strategy on one asset alone, with all the capital: as the firm's products run it")}
      </div>
      <div className="frow">
        <span className="lbl">Market</span>
        {markets.map((m) => chip(f.cls === m, m, () =>
          set({ ...f, cls: m, uni: "All", asset: f.asset && L.assets.get(f.asset)?.market !== m ? null : f.asset })))}
      </div>
      {!AM ? (
        <div className="frow">
          <span className="lbl">List</span>
          {chip(f.uni === "All", "All", () => set({ ...f, uni: "All" }))}
          {f.cls === "All"
            ? meta.markets.filter((m) => research.some((u) => u.market === m)).map((m) => (
                <span key={m} style={{ display: "contents" }}><span className="grp">{m}</span>{listChips(m)}</span>
              ))
            : listChips(f.cls)}
        </div>
      ) : (
        <div className="frow">
          <span className="lbl">Asset</span>
          {chip(!f.asset, "All", () => set({ ...f, asset: null }), "every asset ranked together")}
          {quick.map((a) => chip(f.asset === a.id, a.label, () => set({ ...f, asset: a.id, cls: a.market })))}
          {picked && !quick.some((a) => a.id === picked.id) && chip(true, picked.label, () => set({ ...f, asset: null }))}
          <input className="search" list="adl" autoComplete="off" value={typed} aria-label="pick an asset"
            placeholder={f.cls === "All" ? `any symbol (${pool.length})…` : `or type one of ${pool.length}…`}
            onChange={(e) => {
              setTyped(e.target.value);
              const how = (e.nativeEvent as InputEvent).inputType;
              if (!how || how === "insertReplacementText") pick(e.target.value);     // chosen from the list
            }}
            onKeyDown={(e) => { if (e.key === "Enter") pick(typed); }} />
          <datalist id="adl">{pool.map((a) => <option key={a.id} value={a.label}>{a.market}</option>)}</datalist>
        </div>
      )}
      <div className="frow">
        <span className="lbl">Timeframe</span>
        {["All", "1h", "4h", "1d"].map((t) => chip(f.tf === t, t, () => set({ ...f, tf: t })))}
        <span className="gap" />
        {AM && (
          <>
            <span className="lbl">OOS months ≥</span>
            {MIN_MONTHS.map((m) => chip(f.min === m, String(m), () => set({ ...f, min: m }),
              "leave out a record shorter than this: a few months are luck-prone"))}
            <span className="gap" />
            <span className="lbl">Verdict</span>
            {VERDICTS.map((v) => chip(f.ver === v, v, () => set({ ...f, ver: v })))}
            <span className="gap" />
          </>
        )}
        {chip(f.hideStale, "Hide stale", () => set({ ...f, hideStale: !f.hideStale }))}
      </div>
    </div>
  );
}

export const filtersKey = (f: F) => JSON.stringify({ ...DEFAULTS, ...f });

// The page: the header (the results' age, the re-run, the theme), Research, and the popup of the result opened.
import { useCallback, useEffect, useMemo, useState } from "react";
import { Boundary } from "./ui/Boundary";
import { useMeta } from "./api/queries";
import { useLive } from "./live/useLive";
import { Popup } from "./result/Popup";
import { lookup } from "./research/model";
import { Research } from "./research/Research";
import { RunControl } from "./run/RunControl";
import { count, moment } from "./format";
import { filtersKey } from "./research/Filters";
import { readUrl, writeUrl, type Filters } from "./url";

export function App() {
  const [first] = useState(readUrl);
  const [f, setF] = useState<Filters>(first.filters);
  const [openKey, setOpenKey] = useState<string | null>(first.result);
  const [theme, setTheme] = useState<string | null>(first.theme);
  const [highlight, setHighlight] = useState<string | null>(null);
  const [tableKeys, setTableKeys] = useState<string[]>([]);
  const [metric, setMetric] = useState("sharpe");          // Same strategy elsewhere keeps its metric from popup to popup
  const [cardBH, setCardBH] = useState(true);
  const [cardIdx, setCardIdx] = useState(true);
  const meta = useMeta();
  const L = useMemo(() => (meta.data ? lookup(meta.data) : null), [meta.data]);
  useLive();

  useEffect(() => writeUrl(f, openKey, theme), [f, openKey, theme]);
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
  }, [theme]);
  const onTableRows = useCallback((keys: string[]) =>
    setTableKeys((was) => (was.length === keys.length && was.every((k, i) => k === keys[i]) ? was : keys)), []);
  const open = useCallback((key: string) => setOpenKey(key), []);
  const close = useCallback(() => {
    setHighlight(openKey);                                  // back to the row last shown
    setOpenKey(null);
  }, [openKey]);
  useEffect(() => {
    if (!highlight) return;
    const t = window.setTimeout(() => setHighlight(null), 1500);
    return () => window.clearTimeout(t);
  }, [highlight]);
  const toggleTheme = () => {
    const dark = theme ? theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(dark ? "light" : "dark");
  };

  const m = meta.data, stale = m ? m.stale[f.mode] : 0;
  return (
    <>
      <header className="top" inert={openKey != null}>
        <div className="top-in">
          <a className="brand" href="/" aria-label="Strategy Lab: Research">
            <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
              <rect width="24" height="24" rx="6.5" style={{ fill: "var(--s1)" }} />
              <path d="M5 17.5 9.5 12.5 12.5 15 18.2 8.3V18.5H5Z" fill="#fff" fillOpacity=".2" />
              <path d="M5 17.5 9.5 12.5 12.5 15 18.2 8.3" fill="none" stroke="#fff" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round" />
              <circle cx="18.2" cy="8.3" r="1.9" fill="#fff" />
            </svg>
            <span>Strategy <b>Lab</b></span>
          </a>
          <nav className="nav">
            <a href="/" className="on">Research</a>
            <a href="/portfolios">Portfolios</a>
          </nav>
          <div className="right">
            {m && (
              <span className="fresh" title={`${count(m.stale.lists)} list results and ${count(m.stale.assets)} single-asset results were computed by code older than the latest changes: re-run them before use`}>
                updated {moment(m.updated_at)} · {count(stale)} stale
              </span>
            )}
            <RunControl />
            <button className="btn" title="light / dark" onClick={toggleTheme}>◐</button>
          </div>
        </div>
      </header>
      <main inert={openKey != null}>
        {meta.error ? <div className="note">Could not reach the server: {meta.error.message}</div>
          : !m || !L ? <p className="loading">Loading…</p>
          : (
            <Boundary what="The results" reset={filtersKey(f)}>
              <Research meta={m} L={L} f={f} setF={setF} onOpen={open} highlight={highlight} onTableRows={onTableRows} />
            </Boundary>
          )}
      </main>
      {openKey && m && L && (
        <Popup key={openKey} k={openKey} meta={m} L={L} rows={tableKeys} onOpen={open} onClose={close} metric={metric}
          setMetric={setMetric} showBH={cardBH} setShowBH={setCardBH} showIdx={cardIdx} setShowIdx={setCardIdx} />
      )}
    </>
  );
}

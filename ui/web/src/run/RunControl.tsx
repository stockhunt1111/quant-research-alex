// The re-run in the header: a Re-run button when nothing runs; while a run is on, a chip with how far it is and how
// long it has left, and a panel with its stages, the jobs running now and those that failed, and Stop; a stopped run
// can be resumed where it stopped, or a new one started.
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Refused, post } from "../api/client";
import { useRun } from "../api/queries";
import type { RunView } from "../api/types";
import { count, moment, span } from "../format";

const STAGE = { lists: "Every strategy on every list", single_assets: "Each strategy on each asset alone" } as const;
const LATER = { picks: "The ML task's choice of a strategy for each asset", summaries: "The lists' and assets' figures the pages show" } as const;

function useNow(on: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!on) return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [on]);
  return now;
}

const finished = (run: RunView) => run.stages.reduce((a, s) => a + s.done + s.skipped + s.failed, 0);
const total = (run: RunView) => run.stages.reduce((a, s) => a + s.total, 0);
const failed = (run: RunView) => run.stages.reduce((a, s) => a + s.failed, 0);

export function RunControl() {
  const run = useRun().data?.run ?? null;
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [single, setSingle] = useState(false);
  const [everything, setEverything] = useState(false);
  const [ask, setAsk] = useState<{ message: string; go: () => void } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const active = !!run?.active;
  const now = useNow(active && open);

  useEffect(() => {
    if (!open) return;
    const away = (ev: MouseEvent) => { if (box.current && !box.current.contains(ev.target as Node)) setOpen(false); };
    const esc = (ev: KeyboardEvent) => { if (ev.key === "Escape") setOpen(false); };
    addEventListener("mousedown", away);
    addEventListener("keydown", esc);
    return () => {
      removeEventListener("mousedown", away);
      removeEventListener("keydown", esc);
    };
  }, [open]);
  const runNow = run ? `${run.id}:${run.state}` : "";            // a question or an error was about the run as it was
  const [runSeen, setRunSeen] = useState(runNow);
  if (runSeen !== runNow) {
    setRunSeen(runNow);
    setAsk(null);
    setError(null);
    setStopping(false);
  }

  const call = async (url: string, body: Record<string, unknown>) => {
    setBusy(true);
    setError(null);
    setAsk(null);
    try {
      await post(url, body);
      await qc.invalidateQueries({ queryKey: ["run"] });
    } catch (e) {
      if (e instanceof Refused && e.confirm) setAsk({ message: e.message, go: () => void call(url, { ...body, confirm: true }) });
      else setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const start = () => call("/api/runs", { single_assets: single, everything });
  const resume = (id: number) => call(`/api/runs/${id}/resume`, {});
  const stop = (id: number) => call(`/api/runs/${id}/stop`, {});

  const n = run ? finished(run) : 0, of = run ? total(run) : 0, bad = run ? failed(run) : 0;
  const halted = run && (run.state === "stopped" || run.state === "interrupted");
  const chip = !run || (!active && !halted)
    ? <button className="btn" onClick={() => setOpen(!open)} title="re-run the research: every strategy on every list">Re-run</button>
    : (
      <button className={"run-chip" + (active ? " on" : "") + (halted ? " halted" : "") + (bad ? " failed" : "")} onClick={() => setOpen(!open)}>
        <i className="dot" />
        {active ? `Re-run ${count(n)} / ${count(of)}` + (run.eta_seconds != null ? ` · about ${span(run.eta_seconds)} left` : "")
          : `Re-run ${run.state} ${count(n)} / ${count(of)}`}
      </button>
    );

  const options = (
    <>
      <label className="chk"><input type="checkbox" checked={single} onChange={(e) => setSingle(e.target.checked)} />
        <span><b>Single assets</b> — each strategy on each asset alone too (the widest list of each market and the ML task's), then the ML task's choice of a strategy for each asset</span></label>
      <label className="chk"><input type="checkbox" checked={everything} onChange={(e) => setEverything(e.target.checked)} />
        <span><b>Everything</b> — also what the current code already computed (after a data refresh); without it those results are kept</span></label>
    </>
  );
  let body;
  if (run && active) {
    body = (
      <>
        <h3>Re-run {run.id}{run.state === "stopping" ? " · stopping…" : run.state === "starting" ? " · starting…" : ""}</h3>
        <p className="cap">started {moment(run.started_at ?? run.requested_at)}{run.resumed ? ` · resumed ${run.resumed}×` : ""}
          {run.single_assets ? " · with single assets" : ""}{run.everything ? " · everything" : ""}{run.narrowed_to ? ` · ${run.narrowed_to} only` : ""}</p>
        <Stages run={run} />
        {run.running.length > 0 && (
          <table className="jobs"><tbody>
            {run.running.map((j) => (
              <tr key={j.stage + j.strategy + j.list_id + j.timeframe}>
                <td className="l">{j.strategy} · {j.list_id} · {j.timeframe}</td>
                <td className="muted" title={j.expected_seconds ? `about ${span(j.expected_seconds)} the last time` : undefined}>
                  {j.started_at ? span((now - Date.parse(j.started_at)) / 1000) : "—"}
                </td>
              </tr>
            ))}
          </tbody></table>
        )}
        <Failed run={run} />
        <div className="row">
          <span className="grow cap" style={{ margin: 0 }}>{run.eta_seconds != null ? `about ${span(run.eta_seconds)} left` : "working out how long it has left…"}</span>
          {run.state !== "stopping" && (stopping
            ? <><span>Stop it now?</span><button className="btn danger" disabled={busy} onClick={() => stop(run.id)}>Stop</button>
                <button className="btn" onClick={() => setStopping(false)}>No</button></>
            : <button className="btn danger" onClick={() => setStopping(true)}
                title="the evaluations running now are cut off and start over on resume; everything finished is kept">Stop…</button>)}
        </div>
      </>
    );
  } else if (run && halted) {
    body = (
      <>
        <h3>Re-run {run.id} {run.state === "interrupted" ? "was interrupted" : "is stopped"}</h3>
        <p className="cap">{count(n)} of {count(of)} jobs done, {count(of - n)} left{bad ? `, ${bad} failed` : ""} · {moment(run.finished_at)}
          {run.state === "interrupted" && run.error ? " · its process ended without a stop" : ""}</p>
        <Stages run={run} />
        <Failed run={run} />
        {run.code_changed && <p className="ask">The evaluation code changed since it started: what it finished was computed by the older code.</p>}
        <div className="row"><span className="grow" /><button className="btn primary" disabled={busy} onClick={() => resume(run.id)}>Resume</button></div>
        <h3 style={{ marginTop: 14 }}>Or start a new re-run</h3>
        {options}
        <div className="row"><span className="grow" /><button className="btn" disabled={busy} onClick={start}>Start new</button></div>
      </>
    );
  } else {
    body = (
      <>
        {run && (
          <p className="cap">Last re-run ({run.id}): done {moment(run.finished_at)}{bad ? ` · ${bad} failed` : " · nothing failed"}
            {run.cpu_seconds ? ` · ${span(run.cpu_seconds)} of CPU` : ""}</p>
        )}
        {run && bad > 0 && (
          <>
            <Failed run={run} />
            <div className="row"><span className="grow" /><button className="btn" disabled={busy} onClick={() => resume(run.id)}>Retry failed</button></div>
          </>
        )}
        <h3 style={{ marginTop: run ? 12 : 0 }}>Re-run the research</h3>
        <p className="cap">every strategy on every list, 1h, 4h and 1d; results appear on this page as they are saved</p>
        {options}
        <div className="row"><span className="grow" /><button className="btn primary" disabled={busy} onClick={start}>Start</button></div>
      </>
    );
  }
  return (
    <div className="runbox" ref={box}>
      {chip}
      {open && (
        <div className="panel">
          {body}
          {ask && (
            <div className="ask">{ask.message}
              <div className="row"><span className="grow" /><button className="btn" onClick={() => setAsk(null)}>Cancel</button>
                <button className="btn primary" disabled={busy} onClick={ask.go}>Go ahead</button></div>
            </div>
          )}
          {error && <p className="err">{error}</p>}
        </div>
      )}
    </div>
  );
}

function Stages({ run }: { run: RunView }) {
  const pct = (v: number, of: number) => (of ? (100 * v) / of : 0) + "%";
  return (
    <>
      {run.stages.map((s) => (
        <div className="stage" key={s.stage}>
          <div className="head"><span>{STAGE[s.stage]}</span>
            <span><b>{count(s.done + s.skipped + s.failed)}</b> / {count(s.total)}{s.skipped ? ` · ${count(s.skipped)} already current` : ""}{s.failed ? ` · ${s.failed} failed` : ""}</span></div>
          <div className="progress">
            <i className="p-done" style={{ width: pct(s.done, s.total) }} /><i className="p-skip" style={{ width: pct(s.skipped, s.total) }} />
            <i className="p-fail" style={{ width: pct(s.failed, s.total) }} /><i className="p-run" style={{ width: pct(s.running, s.total) }} />
          </div>
        </div>
      ))}
      {(["picks", "summaries"] as const).filter((s) => s !== "picks" || run.single_assets).map((s) => (
        <div className="stage" key={s}>
          <div className="head"><span>{LATER[s]}</span>
            <span>{run.stage === s ? (run.active ? "running" : run.state) : run.state === "done" ? "done" : "after them"}</span></div>
        </div>
      ))}
    </>
  );
}

function Failed({ run }: { run: RunView }) {
  if (!run.failed.length) return null;
  return (
    <table className="jobs"><tbody>
      {run.failed.map((j) => (
        <tr key={j.stage + j.strategy + j.list_id + j.timeframe}>
          <td className="l bad" title={j.error ?? ""}>failed: {j.strategy} · {j.list_id} · {j.timeframe}</td>
          <td className="muted">{j.attempts}×</td>
        </tr>
      ))}
    </tbody></table>
  );
}

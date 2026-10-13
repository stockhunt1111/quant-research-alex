// The server's events turned into what the page reads again: a result saved refetches the rows of its view and any open
// popup or chart line it touches; a new luck bar re-judges the lists; an edit of the evaluation code makes results stale
// everywhere; a run's progress replaces the run shown. Refetches are gathered and made at most once a second, however
// fast a run saves; after a dropped connection everything is read again.
import { useQueryClient, type Query } from "@tanstack/react-query";
import { useRef } from "react";
import type { RunView } from "../api/types";
import { useSseEvent } from "./sse";

const EVERY_MS = 1000;
type Results = { seq: number; kinds: string[]; keys: string[]; every: boolean };
type Pending = { all: boolean; lists: boolean; assets: boolean; meta: boolean; results: boolean; keys: Set<string>; strategies: Set<string> };
const empty = (): Pending => ({ all: false, lists: false, assets: false, meta: false, results: false, keys: new Set(), strategies: new Set() });

export function useLive(): void {
  const qc = useQueryClient();
  const pending = useRef<Pending>(empty());
  const timer = useRef<number | null>(null);
  const last = useRef(0);
  const seq = useRef<number | null>(null);

  const flush = () => {
    timer.current = null;
    last.current = Date.now();
    const p = pending.current;
    pending.current = empty();
    if (p.all) {
      void qc.invalidateQueries();
      return;
    }
    if (p.meta) void qc.invalidateQueries({ queryKey: ["meta"] });
    if (p.lists) void qc.invalidateQueries({ queryKey: ["rows", "lists"] });
    if (p.assets) void qc.invalidateQueries({ queryKey: ["rows", "assets"] });
    if (p.results) {
      void qc.invalidateQueries({ queryKey: ["result"] });
      void qc.invalidateQueries({ queryKey: ["curves"] });
    } else if (p.keys.size) {
      // a popup shows its result and, in Same strategy elsewhere, the strategy's other results
      const touches = (q: Query) =>
        q.queryKey[0] === "result"
          ? p.keys.has(String(q.queryKey[1])) || p.strategies.has(String(q.queryKey[1]).split("@")[0])
          : q.queryKey[0] === "curves" && q.queryKey.slice(1).some((k) => p.keys.has(String(k)));
      void qc.invalidateQueries({ predicate: touches });
    }
  };
  const later = (change: (p: Pending) => void) => {
    change(pending.current);
    if (timer.current == null) timer.current = window.setTimeout(flush, Math.max(0, last.current + EVERY_MS - Date.now()));
  };

  useSseEvent("hello", (d) => {
    const h = d as { seq: number | null; run: RunView | null };
    qc.setQueryData(["run"], { run: h.run });
    if (seq.current != null && h.seq !== seq.current) later((p) => (p.all = true));   // a new server, or missed events
    seq.current = h.seq;
  });
  useSseEvent(["reconnect", "reset", "code"], () => later((p) => (p.all = true)));
  useSseEvent("results", (d) => {
    const r = d as Results;
    seq.current = r.seq;
    later((p) => {
      p.meta = true;
      if (r.kinds.includes("list")) p.lists = true;
      if (r.kinds.includes("asset") || r.kinds.includes("pick")) p.assets = true;
      if (r.every) p.results = true;
      for (const k of r.keys) {
        p.keys.add(k);
        p.strategies.add(k.split("@")[0]);
      }
    });
  });
  useSseEvent("judged", () =>
    later((p) => {
      p.meta = p.lists = p.assets = p.results = true;
    }),
  );
  useSseEvent("run", (d) => qc.setQueryData(["run"], d));
}

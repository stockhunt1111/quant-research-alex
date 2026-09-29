// One EventSource for the page, shared through context (a browser keeps a handful of connections to a server, and an
// EventSource holds one for good). A component subscribes with useSseEvent(name, handler); the provider binds one native
// listener per event name and hands each parsed frame to every subscriber. A synthetic "reconnect" fires on every open
// after the first (EventSource reconnects by itself after a drop): what came meanwhile was missed.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, type ReactNode } from "react";

type Handler = (data: unknown) => void;
interface Hub {
  subscribe: (event: string, handler: Handler) => () => void;
}
const RECONNECT = "reconnect";
const Ctx = createContext<Hub | null>(null);

export function SseProvider({ url, children }: { url: string; children: ReactNode }) {
  const source = useRef<EventSource | null>(null);
  const subs = useRef<Map<string, Set<Handler>>>(new Map());
  const bound = useRef<Set<string>>(new Set());

  const bind = useCallback((es: EventSource, event: string) => {
    if (event === RECONNECT || bound.current.has(event)) return;
    bound.current.add(event);
    es.addEventListener(event, (ev: MessageEvent) => {
      let data: unknown = null;
      try {
        data = ev.data ? JSON.parse(ev.data) : null;
      } catch {
        console.warn("an event that is not JSON:", event, ev.data);
        return;
      }
      subs.current.get(event)?.forEach((h) => h(data));
    });
  }, []);

  useEffect(() => {
    const es = new EventSource(url);
    const names = bound.current;             // the events bound on this source: bound again on the next one
    source.current = es;
    let first = true;
    es.addEventListener("open", () => {
      if (first) {
        first = false;
        return;
      }
      subs.current.get(RECONNECT)?.forEach((h) => h(null));
    });
    for (const event of subs.current.keys()) bind(es, event);
    return () => {
      es.close();
      source.current = null;
      names.clear();
    };
  }, [url, bind]);

  const subscribe = useCallback<Hub["subscribe"]>(
    (event, handler) => {
      let set = subs.current.get(event);
      if (!set) subs.current.set(event, (set = new Set()));
      set.add(handler);
      if (source.current) bind(source.current, event);
      return () => {
        subs.current.get(event)?.delete(handler);
      };
    },
    [bind],
  );
  const hub = useMemo<Hub>(() => ({ subscribe }), [subscribe]);
  return <Ctx.Provider value={hub}>{children}</Ctx.Provider>;
}

// Subscribe to events for the life of the component; the handler may change every render (kept in a ref).
export function useSseEvent(event: string | string[], handler: Handler): void {
  const hub = useContext(Ctx);
  const ref = useRef(handler);
  useEffect(() => {
    ref.current = handler;
  });
  const events = Array.isArray(event) ? event : [event];
  const key = events.join("|");
  useEffect(() => {
    if (!hub) return;
    const offs = key.split("|").map((e) => hub.subscribe(e, (d) => ref.current(d)));
    return () => offs.forEach((off) => off());
  }, [hub, key]);
}

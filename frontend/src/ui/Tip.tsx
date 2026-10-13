// One tooltip for the page, placed by the pointer and kept inside the window (the mockup's showTip / hideTip).
import { createContext, useContext, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";

type At = { clientX: number; clientY: number };
type Shown = { x: number; y: number; content: ReactNode; wide: boolean } | null;
type Api = { show: (at: At, content: ReactNode, wide?: boolean) => void; hide: () => void };
const Ctx = createContext<Api>({ show: () => {}, hide: () => {} });
export const useTip = () => useContext(Ctx);

export function TipProvider({ children }: { children: ReactNode }) {
  const [shown, setShown] = useState<Shown>(null);
  const api = useMemo<Api>(
    () => ({
      show: (at, content, wide = false) => setShown({ x: at.clientX, y: at.clientY, content, wide }),
      hide: () => setShown(null),
    }),
    [],
  );
  return (
    <Ctx.Provider value={api}>
      {children}
      <Box shown={shown} />
    </Ctx.Provider>
  );
}

function Box({ shown }: { shown: Shown }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !shown) return;
    const w = el.offsetWidth, h = el.offsetHeight;
    let x = shown.x + 14, y = shown.y + 14;
    if (x + w > innerWidth - 8) x = shown.x - w - 14;
    if (y + h > innerHeight - 8) y = shown.y - h - 14;
    el.style.left = Math.max(8, x) + "px";
    el.style.top = Math.max(8, y) + "px";
  });
  return (
    <div ref={ref} className={"tip" + (shown?.wide ? " wide" : "")} style={{ display: shown ? "block" : "none" }}>
      {shown?.content}
    </div>
  );
}

// a tooltip row: a line's key (a CSS background), a value, a name
export const TipRow = ({ swatch, value, name }: { swatch: string; value: string; name: string }) => (
  <div className="r">
    <i style={{ background: swatch }} />
    <b>{value}</b>
    <span>{name}</span>
  </div>
);

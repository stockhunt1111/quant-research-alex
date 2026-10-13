// A column header that sorts its table: a click or Enter sorts by it, again the other way. The column sorted by shows
// its arrow (▼ the largest first, ▲ the smallest), and screen readers hear which way (aria-sort).
import type { ReactNode } from "react";

type Props = { label: ReactNode; title?: string; cls?: string; dir: 1 | -1 | null; onSort: () => void };

export function SortTh({ label, title, cls = "", dir, onSort }: Props) {
  return (
    <th className={"sort " + cls} title={title} tabIndex={0} onClick={onSort}
      aria-sort={dir === 1 ? "ascending" : dir === -1 ? "descending" : "none"}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSort();
        }
      }}>
      {label}
      {dir != null && <span className="arrow" aria-hidden="true">{dir === 1 ? "▲" : "▼"}</span>}
    </th>
  );
}

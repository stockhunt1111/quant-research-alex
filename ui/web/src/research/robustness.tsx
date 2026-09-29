// The Robustness column and its tooltip: each check's value, threshold and state as the board judged it.
import type { ReactNode } from "react";
import type { Robustness, Row } from "../api/types";
import { STATE, robFull } from "./model";

export function RobRows({ rob, labels }: { rob: Robustness; labels: Map<string, string> }) {
  return (
    <table className="rob">
      <tbody>
        {rob.checks.map(([id, value, threshold, state]) => (
          <tr key={id}>
            <td>{labels.get(id) ?? id}</td>
            <td>{value}{threshold ? <span className="muted"> · {threshold}</span> : null}</td>
            <td className={"st st-" + state}>{STATE[state]}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function robTip(r: Row, labels: Map<string, string>): ReactNode {
  const rob = r.robustness ?? null;
  if (!rob) return <><div className="t">Robustness</div>{r.loses || "it loses money out-of-sample"}: nothing to check</>;
  const missing = rob.not_computed ? ` · ${rob.not_computed} not computed` : "";
  return (
    <>
      <div className="t">Robustness: {rob.passed} of {rob.applicable} checks passed{missing}</div>
      <RobRows rob={rob} labels={labels} />
    </>
  );
}

export const robClass = (rob: Robustness | null): string =>
  !rob ? "muted robc" : "robc" + (robFull(rob) ? " ok" : "") + (rob.not_computed ? " muted" : "");

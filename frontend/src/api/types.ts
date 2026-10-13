// The server's answers, named: every shape is generated from the server's models (schema.ts, `make api`).
import type { components } from "./schema";

type S = components["schemas"];
export type Meta = S["Meta"];
export type ListInfo = S["ListInfo"];
export type AssetInfo = S["AssetInfo"];
export type Targets = S["Targets"];
export type ListRow = S["ListRow"];
export type AssetRow = S["AssetRow"];
export type Robustness = S["Robustness"];
export type ResultView = S["ResultView"];
export type Figures = S["Figures"];
export type BenchmarkFigures = S["BenchmarkFigures"];
export type Months = S["Months"];
export type WindowsView = S["WindowsView"];
export type NameView = S["NameView"];
export type Cell = S["Cell"];
export type Candidate = S["Candidate"];
export type Curves = S["Curves"];
export type RunView = S["RunView"];
export type RunCurrent = S["RunCurrent"];
export type Refusal = S["Refusal"];
export type Point = [number, number];
export type Row = ListRow | AssetRow;
export type Timeframe = ListRow["timeframe"];

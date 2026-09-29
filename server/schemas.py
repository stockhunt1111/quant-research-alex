"""The shapes the server answers with. The web UI's TypeScript types are generated from them (ui/web/src/api/schema.ts,
`make api`), and the tests hold every answer to them, so the page and the server cannot drift apart.

The heavy answers (a view's rows) are written as JSON straight from the database and kept compressed: these models
describe them without validating them on every request.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, create_model

from strategy_lab import db


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


Timeframe = Literal["1h", "4h", "1d"]
CheckState = Literal["passed", "failed", "too_short", "not_applicable", "not_computed"]
Point = tuple[int, float]              # (milliseconds since 1970 UTC, growth of 1)


# ---------------------------------------------------------------------------------------------------- the page's meta
class Targets(Model):
    avg_monthly: float
    avg_monthly_stretch: float
    pct_green: float
    pct_green_stretch: float
    max_dd: float
    sharpe: float


class ListSize(Model):
    fixed: int | None           # a fixed list: its instruments with bars
    trading: int | None         # ... and those trading in its last ten days
    held: int | None            # a ranked list: the most names held at once
    ever: int | None            # ... and all it ever held


class ListInfo(Model):
    id: str
    market: str
    label: str
    title: str
    research: bool              # one of our lists, ranked as a book (the ML task's own lists are not)
    per_instrument: bool        # its instruments are run alone
    venue: str | None
    costs: str | None
    size: ListSize | None


class AssetInfo(Model):
    id: str
    label: str
    market: str
    liquidity: float | None


class CheckInfo(Model):
    id: str
    label: str


class TriesInfo(Model):
    saved: int
    independent: float
    best_95: float


class StaleCounts(Model):
    lists: int
    assets: int


class StrategyNames(Model):
    lists: list[str]
    assets: list[str]


class Meta(Model):
    targets: Targets
    capital: float
    min_trades: int             # a row of fewer trades has no At 10% DD (research.MIN_TRADES)
    markets: list[str]
    lists: list[ListInfo]
    checks: list[CheckInfo]
    assets: list[AssetInfo]
    strategies: StrategyNames
    tries: TriesInfo | None
    updated_at: str | None
    stale: StaleCounts


# ---------------------------------------------------------------------------------------------------- rows
class Robustness(Model):
    passed: int
    applicable: int
    not_computed: int
    checks: list[tuple[str, str, str, CheckState]]      # (check id, value, threshold, state)


class Row(Model):
    key: str
    id: int
    strategy: str
    list_id: str
    timeframe: Timeframe
    start: str | None
    end: str | None
    months: int | None
    avg_monthly: float | None
    cagr: float | None
    green: float | None
    max_dd: float | None
    at_target_dd: float | None              # the average month sized to the target's drawdown, all the record's years
                                            # (db.at_target_dd); None: fewer trades than Meta.min_trades, or a record
                                            # that never lost a day
    at_target_dd_multiple: float | None     # its positions' multiple; None: nothing to scale (no position, no loss)
    at_target_dd_5y: float | None           # the same over the record's last five years; None: a record no longer
    sharpe: float | None
    k_ratio: float | None       # None: not worked out (an account that lost everything)
    trades: int | None          # None: not counted (a strategy_pick's trades are its candidates')
    beats_bh: bool
    vs_bh: float | None
    bh_cash: bool
    bh_sharpe: float | None
    bh_max_dd: float | None
    bh_k_ratio: float | None    # None: no buy & hold series kept (cash, or a result imported from files)
    bh_at_target_dd: float | None           # buy & hold's average month sized to the target's drawdown, fully invested
                                            # (db.HELD_GROSS); None: no buy & hold series kept, or one never down a day
    bh_at_target_dd_multiple: float | None  # the share of the money it holds at that size
    targets: list[bool]         # avg month, green months, max drawdown, Sharpe, beats buy & hold
    targets_met: int
    robustness: Robustness | None   # None: not checked (it loses money, or a strategy_pick's choice)
    loses: str | None               # why it is not checked
    stale: bool


class ListRow(Row):
    worst_month: float | None
    evaluated_at: str


class AssetRow(Row):
    instrument_id: str
    bh_avg_monthly: float | None
    params_now: dict[str, Any] | None
    held: bool                  # its buy & hold is kept (a result imported from files before the database has none)


class ListRows(Model):
    rows: list[ListRow]


class AssetRows(Model):
    rows: list[AssetRow]


# ---------------------------------------------------------------------------------------------------- a result's popup
Figures = create_model("Figures", __base__=Model,
                       **{k: ((str | None) if k in ("start", "end") else (float | None), ...) for k in db.FIGURES})
BenchmarkFigures = create_model("BenchmarkFigures", __base__=Model, **{
    k: ((str | None) if k in ("start", "end") else (float | None), ...) for k in db.BENCHMARK_FIGURES})


class Months(Model):
    first: str                  # YYYY-MM
    v: list[int | None]         # basis points, month after month


class CurveView(Model):
    pts: list[Point]
    dd: list[int | None]        # basis points: the drawdown's low since the point before
    months: Months | None
    growth: float | None


class BenchmarkView(Model):
    pts: list[Point]
    months: Months | None
    growth: float | None
    figures: BenchmarkFigures


class IndexView(BenchmarkView):
    name: str                   # the market's index held over the record's days: SPY, BTC, Gold, Gold future, WTI future


class WindowsView(Model):
    sets: list[dict[str, Any]]
    end: str
    train: int
    w: list[tuple[str, int]]    # (first day, index into sets)


class Alone(Model):
    key: str
    sharpe: float | None
    vs_bh: float | None
    beats_bh: bool
    stale: bool


class NameView(Model):
    instrument_id: str
    label: str
    trades: int
    won: float | None
    compounded: float | None
    alone: Alone | None


class Cell(Model):
    key: str
    list_id: str
    timeframe: Timeframe
    sharpe: float | None
    avg_monthly: float | None
    cagr: float | None
    vs_bh: float | None
    beats_bh: bool
    bh_cash: bool
    targets_met: int
    stale: bool


class Elsewhere(Model):
    lists: list[Cell]
    alone: list[Cell]


class Candidate(Model):
    key: str
    strategy: str
    sharpe: float | None
    windows: int
    stale: bool


class ResultView(Model):
    key: str
    kind: Literal["list", "asset", "pick"]
    row: ListRow | AssetRow
    description: str
    notes: list[str]
    fill: str | None
    evaluated_at: str
    figures: Figures
    curve: CurveView
    benchmark: BenchmarkView | None
    index: IndexView | None
    windows: WindowsView | None
    params: dict[str, Any] | None
    grid: dict[str, list[Any]] | None
    params_now: dict[str, Any] | None
    names: list[NameView]
    elsewhere: Elsewhere
    candidates: list[Candidate]
    beyond_luck: float | None


class CurveLine(Model):
    pts: list[Point]
    benchmark: list[Point] | None
    index: list[Point] | None
    index_name: str | None


class Curves(Model):
    curves: dict[str, CurveLine]


# ---------------------------------------------------------------------------------------------------- runs
RunState = Literal["starting", "running", "stopping", "stopped", "interrupted", "done"]
RunStage = Literal["lists", "single_assets", "picks", "summaries"]     # summaries: the lists' figures (`catalog`)


class StageProgress(Model):
    stage: Literal["lists", "single_assets"]
    total: int
    done: int
    failed: int
    skipped: int
    stopped: int
    queued: int
    running: int


class RunningJob(Model):
    stage: Literal["lists", "single_assets"]
    strategy: str
    list_id: str
    timeframe: Timeframe
    started_at: str | None
    expected_seconds: float


class FailedJob(Model):
    stage: Literal["lists", "single_assets"]
    strategy: str
    list_id: str
    timeframe: Timeframe
    attempts: int
    error: str | None


class RunView(Model):
    id: int
    state: RunState
    stage: RunStage | None
    active: bool                # its process is on (or it is starting)
    requested_by: Literal["ui", "cli"]
    requested_at: str
    started_at: str | None
    finished_at: str | None
    single_assets: bool
    everything: bool
    narrowed_to: str | None     # the lists, strategies and timeframes a run asked for some only covers (the page's
                                # names); None: everything. It ends with the picks and the lists' figures all the same
    resumed: int
    code_changed: bool          # the evaluation code changed since it started
    stages: list[StageProgress]
    running: list[RunningJob]
    failed: list[FailedJob]
    eta_seconds: float | None
    cpu_seconds: float | None
    error: str | None
    log_path: str | None


class RunCurrent(Model):
    run: RunView | None


class RunOnly(Model):
    strategies: list[str] | None = None
    lists: list[str] | None = None
    timeframes: list[Timeframe] | None = None


class RunStart(Model):
    single_assets: bool = False
    everything: bool = False
    only: RunOnly | None = None         # a narrower run, for tests and the terminal; the page always runs everything
    confirm: bool = False               # start although the evaluation code has uncommitted changes


class RunResume(Model):
    confirm: bool = False               # resume although the evaluation code changed since the run started


class RunAccepted(Model):
    id: int


class Refusal(Model):
    message: str
    confirm: bool                       # asking again with confirm: true goes ahead

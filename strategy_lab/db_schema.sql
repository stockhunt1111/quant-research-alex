-- The app's database (db/app.sqlite): every evaluation with its figures and series, the runs that produce them, and
-- the luck bar derived from them. Written by strategy_lab.db; read by the server, the summaries and the batch scripts.
-- Days are 'YYYY-MM-DD' (UTC), moments ISO-8601 UTC; a figure that is not a number (NaN, ±inf) is NULL. A JSON column
-- holds what a strategy or a check shapes itself: a strategy's parameters, a check's measurement.

-- A strategy as the result that names it last described it (strategies/<name>.py, Strategy.description): a result
-- of a strategy that no longer exists keeps its rule's text.
CREATE TABLE strategy (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    updated_at  TEXT NOT NULL
) STRICT;

-- The code an evaluation ran on (provenance.stamp): the hash and the files it covers, so whether a result is stale is
-- worked out once per stamp.
CREATE TABLE code_stamp (
    sha   TEXT PRIMARY KEY,
    files TEXT NOT NULL CHECK (json_valid(files) AND json_type(files) = 'array')
) STRICT;

-- One re-run of the research, started from the UI or a terminal: its settings, where it is, and its process.
CREATE TABLE run (
    id                INTEGER PRIMARY KEY,
    requested_at      TEXT NOT NULL,
    requested_by      TEXT NOT NULL CHECK (requested_by IN ('ui', 'cli')),
    single_assets     INTEGER NOT NULL CHECK (single_assets IN (0, 1)),
    everything        INTEGER NOT NULL CHECK (everything IN (0, 1)),
    only              TEXT CHECK (only IS NULL OR json_valid(only)),
    state             TEXT NOT NULL CHECK (state IN ('starting', 'running', 'stopping', 'stopped', 'interrupted', 'done')),
    stage             TEXT CHECK (stage IS NULL OR stage IN ('lists', 'single_assets', 'picks', 'summaries')),
    code_sha_at_start TEXT NOT NULL,
    pid               INTEGER,
    pgid              INTEGER,
    started_at        TEXT,
    finished_at       TEXT,
    heartbeat_at      TEXT,
    resumed           INTEGER NOT NULL DEFAULT 0,
    cpu_seconds       REAL,
    log_path          TEXT,
    error             TEXT
) STRICT;

-- One evaluation inside a run: a strategy on a list as one book (stage lists) or on each instrument of a list alone
-- (stage single_assets). A run's jobs are fixed when it starts: resuming it runs those not done.
CREATE TABLE run_job (
    id               INTEGER PRIMARY KEY,
    run_id           INTEGER NOT NULL REFERENCES run (id) ON DELETE CASCADE,
    stage            TEXT NOT NULL CHECK (stage IN ('lists', 'single_assets')),
    strategy         TEXT NOT NULL,
    list_id          TEXT NOT NULL,
    timeframe        TEXT NOT NULL CHECK (timeframe IN ('1h', '4h', '1d')),
    state            TEXT NOT NULL CHECK (state IN ('queued', 'running', 'done', 'failed', 'skipped', 'stopped')),
    expected_seconds REAL NOT NULL DEFAULT 0,
    attempts         INTEGER NOT NULL DEFAULT 0,
    started_at       TEXT,
    finished_at      TEXT,
    seconds          REAL,
    worker_pid       INTEGER,
    error            TEXT,
    UNIQUE (run_id, stage, strategy, list_id, timeframe)
) STRICT;
CREATE INDEX run_job_by_state ON run_job (run_id, state);

-- Buy-and-hold of a list or of one instrument over a record's days (engine.hold): the series every result on the
-- same days shares, and its figures (metrics.core), the popup's Buy & hold column; beside them its K-ratio (db.k_ratio)
-- and its average month at the target's drawdown fully invested (db.at_target_dd, db.HELD_GROSS), with the multiple
-- and the same over its last five years: what holding makes at the drawdown a result's own figure is sized to.
CREATE TABLE benchmark (
    id                    INTEGER PRIMARY KEY,
    list_id               TEXT,
    instrument_id         TEXT,
    timeframe             TEXT NOT NULL CHECK (timeframe IN ('1h', '4h', '1d')),
    first_day             TEXT NOT NULL,
    days                  INTEGER NOT NULL CHECK (days > 0),
    series_sha            TEXT NOT NULL,
    series                BLOB NOT NULL,
    start                 TEXT,
    end                   TEXT,
    months                INTEGER,
    avg_monthly           REAL,
    median_monthly        REAL,
    pct_green             REAL,
    pct_red               REAL,
    months_active         INTEGER,
    pct_months_active     REAL,
    pct_green_active      REAL,
    pct_red_active        REAL,
    worst_month           REAL,
    best_month            REAL,
    longest_red_streak    INTEGER,
    cagr                  REAL,
    ann_vol               REAL,
    sharpe                REAL,
    sortino               REAL,
    max_dd                REAL,
    max_dd_days           INTEGER,
    k_ratio               REAL,
    at_target_dd          REAL,
    at_target_dd_multiple REAL,
    at_target_dd_5y       REAL,
    CHECK ((list_id IS NULL) <> (instrument_id IS NULL))
) STRICT;
CREATE UNIQUE INDEX benchmark_by_series ON benchmark (coalesce(list_id, ''), coalesce(instrument_id, ''), timeframe,
                                                      series_sha);

-- One evaluation of a strategy on a timeframe: on a list as one book (instrument_id NULL), or on one instrument alone
-- with all the capital, taken from a list. What the evaluation computed; judged figures are worked out when read.
CREATE TABLE result (
    id                   INTEGER PRIMARY KEY,
    strategy             TEXT NOT NULL REFERENCES strategy (name),
    list_id              TEXT NOT NULL,
    instrument_id        TEXT,
    instrument_key       TEXT GENERATED ALWAYS AS (coalesce(instrument_id, '')) VIRTUAL,
    timeframe            TEXT NOT NULL CHECK (timeframe IN ('1h', '4h', '1d')),
    evaluated_at         TEXT NOT NULL,
    code_sha             TEXT NOT NULL REFERENCES code_stamp (sha),
    fill                 TEXT CHECK (fill IS NULL OR fill IN ('next_open', 'next_close')),
    seconds              REAL,
    notes                TEXT NOT NULL CHECK (json_valid(notes) AND json_type(notes) = 'array'),
    grid                 TEXT CHECK (grid IS NULL OR (json_valid(grid) AND json_type(grid) = 'object')),
    params_in_sample     TEXT CHECK (params_in_sample IS NULL OR json_valid(params_in_sample)),
    params_now           TEXT CHECK (params_now IS NULL OR json_valid(params_now)),
    vs_bh                REAL,
    beats_bh             INTEGER NOT NULL CHECK (beats_bh IN (0, 1)),
    bh_cash              INTEGER CHECK (bh_cash IS NULL OR bh_cash IN (0, 1)),
    bh_avg_monthly       REAL,
    bh_pct_green         REAL,
    bh_sharpe            REAL,
    bh_max_dd            REAL,
    bh_cagr              REAL,
    benchmark_id         INTEGER REFERENCES benchmark (id),
    target_avg_monthly   INTEGER NOT NULL CHECK (target_avg_monthly IN (0, 1)),
    target_green_months  INTEGER NOT NULL CHECK (target_green_months IN (0, 1)),
    target_max_dd        INTEGER NOT NULL CHECK (target_max_dd IN (0, 1)),
    target_sharpe        INTEGER NOT NULL CHECK (target_sharpe IN (0, 1)),
    target_beats_bh      INTEGER NOT NULL CHECK (target_beats_bh IN (0, 1)),
    targets_met          INTEGER NOT NULL CHECK (targets_met BETWEEN 0 AND 5),
    rt_sharpe            REAL,
    rt_null_median       REAL,
    rt_null_p95          REAL,
    rt_p                 REAL,
    rt_rotations         INTEGER,
    best5_share          REAL,
    UNIQUE (strategy, list_id, instrument_key, timeframe)
) STRICT;
CREATE INDEX result_by_instrument ON result (instrument_id, timeframe);
CREATE INDEX result_by_list ON result (list_id, timeframe);
CREATE INDEX result_by_benchmark ON result (benchmark_id);

-- A result's figures (metrics.scorecard): out-of-sample, and the configuration best on the whole history over the same
-- days (in_sample); avg_win, avg_loss and profit_factor from its trades; the K-ratio (db.k_ratio) and the average month
-- with the positions scaled so that the max drawdown is the target's (db.at_target_dd), with the multiple and the same
-- over the record's last five years, from its series: a scope without a kept series (an instrument alone's in-sample
-- record) has none.
CREATE TABLE result_figures (
    result_id             INTEGER NOT NULL REFERENCES result (id) ON DELETE CASCADE,
    scope                 TEXT NOT NULL CHECK (scope IN ('out_of_sample', 'in_sample')),
    start                 TEXT,
    end                   TEXT,
    months                INTEGER,
    avg_monthly           REAL,
    median_monthly        REAL,
    pct_green             REAL,
    pct_red               REAL,
    months_active         INTEGER,
    pct_months_active     REAL,
    pct_green_active      REAL,
    pct_red_active        REAL,
    worst_month           REAL,
    best_month            REAL,
    longest_red_streak    INTEGER,
    cagr                  REAL,
    ann_vol               REAL,
    sharpe                REAL,
    sortino               REAL,
    max_dd                REAL,
    max_dd_days           INTEGER,
    n_trades              INTEGER,
    trades_per_month      REAL,
    win_rate              REAL,
    avg_trade             REAL,
    median_trade_bars     REAL,
    time_in_market        REAL,
    avg_gross             REAL,
    avg_win               REAL,
    avg_loss              REAL,
    profit_factor         REAL,
    k_ratio               REAL,
    at_target_dd          REAL,
    at_target_dd_multiple REAL,
    at_target_dd_5y       REAL,
    PRIMARY KEY (result_id, scope)
) STRICT, WITHOUT ROWID;

-- A result's daily series, one value a UTC day from first_day on (float64, zstd): its out-of-sample record, the
-- in-sample configuration's over the same days, and for an instrument alone its position at each day's end.
CREATE TABLE result_series (
    result_id INTEGER NOT NULL REFERENCES result (id) ON DELETE CASCADE,
    kind      TEXT NOT NULL CHECK (kind IN ('out_of_sample', 'in_sample', 'position')),
    first_day TEXT NOT NULL,
    days      INTEGER NOT NULL CHECK (days > 0),
    series    BLOB NOT NULL,
    PRIMARY KEY (result_id, kind)
) STRICT, WITHOUT ROWID;

-- The walk-forward's windows: each one's training span, its test span, and the parameters it traded (the strategy it
-- traded, for strategy_pick); train_sharpe_daily NULL: no configuration held a position on 60 days of the window.
CREATE TABLE result_window (
    result_id          INTEGER NOT NULL REFERENCES result (id) ON DELETE CASCADE,
    n                  INTEGER NOT NULL,
    train_start        TEXT NOT NULL,
    train_end          TEXT NOT NULL,
    test_start         TEXT NOT NULL,
    test_end           TEXT NOT NULL,
    train_sharpe_daily REAL,
    params             TEXT NOT NULL CHECK (json_valid(params)),
    PRIMARY KEY (result_id, n)
) STRICT, WITHOUT ROWID;

-- A list's result by instrument: each name's trades in the book, the share that made money after costs, and their
-- returns compounded (the popup's Asset by asset).
CREATE TABLE result_name (
    result_id     INTEGER NOT NULL REFERENCES result (id) ON DELETE CASCADE,
    instrument_id TEXT NOT NULL,
    trades        INTEGER NOT NULL,
    won           REAL NOT NULL,
    compounded    REAL,
    PRIMARY KEY (result_id, instrument_id)
) STRICT, WITHOUT ROWID;

-- A list's result's out-of-sample trades (engine.trades.ledger), one Parquet file.
CREATE TABLE result_trades (
    result_id INTEGER PRIMARY KEY REFERENCES result (id) ON DELETE CASCADE,
    n_trades  INTEGER NOT NULL,
    payload   BLOB NOT NULL
) STRICT;

-- Monte Carlo of the out-of-sample record (montecarlo.bootstrap); no row: not run (a record that loses money, --no-mc).
CREATE TABLE result_monte_carlo (
    result_id                     INTEGER PRIMARY KEY REFERENCES result (id) ON DELETE CASCADE,
    reps                          INTEGER NOT NULL,
    block_days                    REAL,
    note                          TEXT,
    avg_monthly_p5                REAL,
    avg_monthly_p50               REAL,
    avg_monthly_p95               REAL,
    pct_green_active_p5           REAL,
    pct_green_active_p50          REAL,
    pct_green_active_p95          REAL,
    max_dd_p5                     REAL,
    max_dd_p50                    REAL,
    max_dd_p95                    REAL,
    sharpe_p5                     REAL,
    sharpe_p50                    REAL,
    sharpe_p95                    REAL,
    worst_month_p5                REAL,
    worst_month_p50               REAL,
    worst_month_p95               REAL,
    prob_avg_monthly_below_target REAL,
    prob_max_dd_beyond_target     REAL
) STRICT;

-- A robustness check as the evaluation measured it (robustness.measure): its figures, or {"na": why} / {"too_short":
-- why}; NULL: not measured. Judged against board's thresholds when read. Only a result that makes money has checks.
CREATE TABLE result_check (
    result_id INTEGER NOT NULL REFERENCES result (id) ON DELETE CASCADE,
    check_id  TEXT NOT NULL CHECK (check_id IN ('luck', 'vs_hold', 'timing', 'pbo', 'plateau', 'eras', 'delay',
                                                'costs', 'names', 'neighbour_lists', 'seeds', 'vs_rule')),
    measure   TEXT CHECK (measure IS NULL OR (json_valid(measure) AND json_type(measure) = 'object')),
    seconds   REAL,
    PRIMARY KEY (result_id, check_id)
) STRICT, WITHOUT ROWID;

-- The candidates a strategy_pick result chose among on its instrument and timeframe, and in how many windows each.
CREATE TABLE pick_candidate (
    result_id    INTEGER NOT NULL REFERENCES result (id) ON DELETE CASCADE,
    candidate_id INTEGER NOT NULL REFERENCES result (id) ON DELETE CASCADE,
    windows      INTEGER NOT NULL,
    PRIMARY KEY (result_id, candidate_id)
) STRICT, WITHOUT ROWID;

-- An instrument a run on a list could not score alone: fewer than per_asset.MIN_MONTHS out-of-sample months (0: fewer
-- than 100 bars).
CREATE TABLE asset_too_short (
    strategy      TEXT NOT NULL REFERENCES strategy (name),
    list_id       TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    timeframe     TEXT NOT NULL CHECK (timeframe IN ('1h', '4h', '1d')),
    months        INTEGER NOT NULL,
    evaluated_at  TEXT NOT NULL,
    code_sha      TEXT NOT NULL REFERENCES code_stamp (sha),
    PRIMARY KEY (strategy, list_id, instrument_id, timeframe)
) STRICT, WITHOUT ROWID;

-- What a list's daily bars say about it (the UI's list sizes and costs): a fixed list's instruments with bars and those
-- trading in its last ten days; a ranked list's most names held at once and all it ever held; a trade's cost a side.
CREATE TABLE list_stats (
    list_id      TEXT PRIMARY KEY,
    size_fixed   INTEGER,
    size_trading INTEGER,
    size_held    INTEGER,
    size_ever    INTEGER,
    sources      TEXT NOT NULL CHECK (json_valid(sources) AND json_type(sources) = 'array'),
    cost_bp_min  REAL NOT NULL,
    cost_bp_max  REAL NOT NULL,
    updated_at   TEXT NOT NULL
) STRICT;

-- An instrument's median dollar volume over its last 60 daily bars: the order of the Asset row's quick picks.
CREATE TABLE instrument_stats (
    instrument_id TEXT PRIMARY KEY,
    liquidity_usd REAL,
    as_of         TEXT NOT NULL
) STRICT;

-- How many independent tries a family of results adds up to (board.luck_of) and the score luck gives the best of them
-- one time in twenty: lists — every list result, the luck bar of its checks; picks — every strategy_pick result;
-- assets — the strategies run on instruments alone, every (strategy, instrument, timeframe) pair scored once however
-- many lists scored it (board.refresh_asset_tries). covers: the family's results it was worked out on (how many, the
-- last id, the last evaluated_at): they changed since when it no longer matches. best_95_excess: the same one time in
-- twenty for the t of a Sharpe against holding, whose scores move as the records' excess over their buy & hold (NULL:
-- a count worked out before it was kept, or a family whose results are not judged against holding: picks).
CREATE TABLE tries (
    id          INTEGER PRIMARY KEY,
    family      TEXT NOT NULL CHECK (family IN ('lists', 'picks', 'assets')),
    computed_at TEXT NOT NULL,
    covers      TEXT NOT NULL,
    saved       INTEGER NOT NULL,
    independent REAL NOT NULL,
    best_95     REAL NOT NULL,
    code_sha    TEXT NOT NULL,
    best_95_excess REAL
) STRICT;

-- An instrument's own tries in a count of family 'assets': its pairs (its timeframes, a rule and its variants on it),
-- correlated as their records are. A result on an instrument is judged against the tries on that instrument, and its
-- luck is shown against those on every instrument too.
CREATE TABLE instrument_tries (
    tries_id      INTEGER NOT NULL REFERENCES tries (id) ON DELETE CASCADE,
    instrument_id TEXT NOT NULL,
    saved         INTEGER NOT NULL,
    independent   REAL NOT NULL,
    best_95       REAL NOT NULL,
    best_95_excess REAL,
    PRIMARY KEY (tries_id, instrument_id)
) STRICT, WITHOUT ROWID;

-- Every result saved or deleted, in order: what the server tells the pages to fetch again.
CREATE TABLE change (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    result_id INTEGER NOT NULL,
    kind      TEXT NOT NULL CHECK (kind IN ('list', 'asset', 'pick')),
    op        TEXT NOT NULL CHECK (op IN ('saved', 'deleted'))
) STRICT;
CREATE INDEX change_by_time ON change (at);

CREATE TRIGGER result_inserted AFTER INSERT ON result BEGIN
    INSERT INTO change (at, result_id, kind, op) VALUES (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), NEW.id,
        CASE WHEN NEW.instrument_id IS NULL THEN 'list' WHEN NEW.strategy = 'strategy_pick' THEN 'pick' ELSE 'asset' END,
        'saved');
END;
CREATE TRIGGER result_updated AFTER UPDATE ON result BEGIN
    INSERT INTO change (at, result_id, kind, op) VALUES (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), NEW.id,
        CASE WHEN NEW.instrument_id IS NULL THEN 'list' WHEN NEW.strategy = 'strategy_pick' THEN 'pick' ELSE 'asset' END,
        'saved');
END;
CREATE TRIGGER result_deleted AFTER DELETE ON result BEGIN
    INSERT INTO change (at, result_id, kind, op) VALUES (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), OLD.id,
        CASE WHEN OLD.instrument_id IS NULL THEN 'list' WHEN OLD.strategy = 'strategy_pick' THEN 'pick' ELSE 'asset' END,
        'deleted');
END;

-- Version 6: the tries of the strategies run on instruments alone (family 'assets': every (strategy, instrument,
-- timeframe) pair scored, once however many lists scored it), and each instrument's own beside them: a result on an
-- instrument is judged against the tries on that instrument, and its luck is shown against those on every instrument
-- too (board.refresh_asset_tries). SQLite changes a CHECK only by building the table again.
CREATE TABLE tries_6 (
    id          INTEGER PRIMARY KEY,
    family      TEXT NOT NULL CHECK (family IN ('lists', 'picks', 'assets')),
    computed_at TEXT NOT NULL,
    covers      TEXT NOT NULL,
    saved       INTEGER NOT NULL,
    independent REAL NOT NULL,
    best_95     REAL NOT NULL,
    code_sha    TEXT NOT NULL
) STRICT;

INSERT INTO tries_6 (id, family, computed_at, covers, saved, independent, best_95, code_sha)
SELECT id, family, computed_at, covers, saved, independent, best_95, code_sha FROM tries;

DROP TABLE tries;

ALTER TABLE tries_6 RENAME TO tries;

-- An instrument's own tries in a count of family 'assets': its pairs (its timeframes, a rule and its variants on it),
-- correlated as their records are.
CREATE TABLE instrument_tries (
    tries_id      INTEGER NOT NULL REFERENCES tries (id) ON DELETE CASCADE,
    instrument_id TEXT NOT NULL,
    saved         INTEGER NOT NULL,
    independent   REAL NOT NULL,
    best_95       REAL NOT NULL,
    PRIMARY KEY (tries_id, instrument_id)
) STRICT, WITHOUT ROWID;

-- Version 5: a buy & hold's average month at the target's drawdown (db.at_target_dd), with its multiple and the same
-- over its last five years, beside its K-ratio: what holding makes at the drawdown a result's own figure is sized to.
-- Buy & hold is fully invested (db.HELD_GROSS). The writer works it out from the series it saves; the buy & holds saved
-- before get it here from the series kept with them.
ALTER TABLE benchmark ADD COLUMN at_target_dd REAL;

ALTER TABLE benchmark ADD COLUMN at_target_dd_multiple REAL;

ALTER TABLE benchmark ADD COLUMN at_target_dd_5y REAL;

-- MATERIALIZED: worked out once a series, not once for each figure the UPDATE reads from it
WITH v AS MATERIALIZED (SELECT id, at_target_dd(first_day, days, series, 1.0, 1.0) AS figures FROM benchmark)
UPDATE benchmark SET at_target_dd = json_extract(v.figures, '$[0]'),
                     at_target_dd_multiple = json_extract(v.figures, '$[1]'),
                     at_target_dd_5y = json_extract(v.figures, '$[2]')
FROM v WHERE benchmark.id = v.id;

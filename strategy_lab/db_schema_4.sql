-- Version 4: a result's average month at the target's drawdown (db.at_target_dd) worked out again from the series kept
-- with it, now with no ceiling on the multiple: step 3 had held the equity in positions to at most twice the equity,
-- and a record that stayed above the target there had been taken at it. A scope without a kept series has none.
-- MATERIALIZED: worked out once a series, not once for each figure the UPDATE reads from it
WITH v AS MATERIALIZED (
    SELECT s.result_id, s.kind, at_target_dd(s.first_day, s.days, s.series, f.avg_gross, f.time_in_market) AS figures
    FROM result_series s JOIN result_figures f ON f.result_id = s.result_id AND f.scope = s.kind)
UPDATE result_figures SET at_target_dd = json_extract(v.figures, '$[0]'),
                          at_target_dd_multiple = json_extract(v.figures, '$[1]'),
                          at_target_dd_5y = json_extract(v.figures, '$[2]')
FROM v WHERE result_figures.result_id = v.result_id AND result_figures.scope = v.kind;

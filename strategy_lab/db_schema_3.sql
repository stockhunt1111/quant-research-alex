-- Version 3: a result's average month with its positions scaled so that its max drawdown is the firm's target
-- (db.at_target_dd), with the multiple and the same over the record's last five years, beside its other figures,
-- out-of-sample and in-sample. The writer works it out from the series it saves; the results saved before get it here
-- from the series kept with them (at_target_dd(...) is db.at_target_dd of a stored series, `db.migrate`): a scope
-- without a kept series (an instrument alone's in-sample record) has none.
ALTER TABLE result_figures ADD COLUMN at_target_dd REAL;

ALTER TABLE result_figures ADD COLUMN at_target_dd_multiple REAL;

ALTER TABLE result_figures ADD COLUMN at_target_dd_5y REAL;

-- MATERIALIZED: worked out once a series, not once for each figure the UPDATE reads from it
WITH v AS MATERIALIZED (
    SELECT s.result_id, s.kind, at_target_dd(s.first_day, s.days, s.series, f.avg_gross, f.time_in_market) AS figures
    FROM result_series s JOIN result_figures f ON f.result_id = s.result_id AND f.scope = s.kind)
UPDATE result_figures SET at_target_dd = json_extract(v.figures, '$[0]'),
                          at_target_dd_multiple = json_extract(v.figures, '$[1]'),
                          at_target_dd_5y = json_extract(v.figures, '$[2]')
FROM v WHERE result_figures.result_id = v.result_id AND result_figures.scope = v.kind;

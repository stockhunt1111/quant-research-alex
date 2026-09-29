-- Version 2: Kestner's K-ratio (db.k_ratio) beside a result's other figures, out-of-sample and in-sample, and beside a
-- buy & hold's. The writer works it out from the series it saves; the results saved before get it here from the series
-- kept with them (k_ratio(days, series) is db.k_ratio of a stored series, `db.migrate`): a scope without a kept series
-- (an instrument alone's in-sample record) has none.
ALTER TABLE result_figures ADD COLUMN k_ratio REAL;

UPDATE result_figures SET k_ratio = (SELECT k_ratio(s.days, s.series) FROM result_series s
                                     WHERE s.result_id = result_figures.result_id AND s.kind = result_figures.scope);

ALTER TABLE benchmark ADD COLUMN k_ratio REAL;

UPDATE benchmark SET k_ratio = k_ratio(days, series);

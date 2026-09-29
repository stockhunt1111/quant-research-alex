import math
import os

from strategy_lab import batch


def test_pieces_keep_a_group_together_up_to_a_share_of_the_run_and_start_longest_first():
    jobs = [("a", 30.0), ("a", 20.0), ("a", 10.0), ("b", 5.0), ("c", 1.0), ("c", 1.0)]
    out = batch.pieces(jobs, group=lambda j: j[0], seconds=lambda j: j[1], workers=1)
    # the run is 67 s: a piece holds at most a quarter of it, 16.75 s, or one job longer than that
    assert [[jobs[k] for k, _ in items] for _, items in out] == [
        [("a", 30.0)], [("a", 20.0)], [("a", 10.0)], [("b", 5.0)], [("c", 1.0), ("c", 1.0)]]
    assert [s for s, _ in out] == [30.0, 20.0, 10.0, 5.0, 2.0]


def test_pieces_without_times_on_record_count_every_job_the_same():
    jobs = list(range(8))
    out = batch.pieces(jobs, group=lambda j: j % 2, seconds=lambda j: 0.0, workers=1)
    assert sorted(len(items) for _, items in out) == [2, 2, 2, 2]


def _pid_and_root(x: float) -> tuple[int, float]:
    return os.getpid(), math.sqrt(x)


def test_run_returns_each_result_in_the_jobs_order_and_runs_the_jobs_it_is_told_to_in_the_main_process():
    out = batch.run(_pid_and_root, [16.0, 1.0, 9.0, 4.0], workers=2, group=lambda j: j > 5, seconds=lambda j: j,
                    in_main=lambda j: j == 9.0)
    assert [r[1] for r, _ in out] == [4.0, 1.0, 3.0, 2.0]
    assert all(s >= 0 for _, s in out)
    here = [j for j, (r, _) in zip([16.0, 1.0, 9.0, 4.0], out) if r[0] == os.getpid()]
    assert here == [9.0]


def test_expected_falls_back_to_the_median_of_the_list_and_timeframe():
    times = {("a", "l1", "1h"): 10.0, ("b", "l1", "1h"): 30.0, ("a", "l2", "1d"): 2.0}
    assert batch.expected(times, "a", "l1", "1h") == 10.0
    assert batch.expected(times, "c", "l1", "1h") == 20.0          # the list's median
    assert batch.expected(times, "c", "l9", "1d") == 2.0           # the timeframe's
    assert batch.expected(times, "c", "l9", "4h") == 0.0


def test_one_worker_runs_every_job_in_this_process_where_a_stand_in_reaches_it(monkeypatch):
    seen = []
    monkeypatch.setattr(math, "sqrt", lambda x: seen.append(x) or -x)
    out = batch.run(math.sqrt, [4.0, 9.0], workers=1, group=lambda j: 0, seconds=lambda j: j)
    assert [r for r, _ in out] == [-4.0, -9.0] and sorted(seen) == [4.0, 9.0]

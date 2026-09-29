from strategy_lab import provenance


def _tree(root):
    (root / "strategy_lab").mkdir()
    (root / "strategies").mkdir()
    (root / "strategy_lab" / "engine.py").write_text("RATE = 1\n")
    (root / "strategy_lab" / "board.py").write_text("TITLE = 'a'\n")
    (root / "strategies" / "trend.py").write_text("LOOKBACK = 20\n")
    return str(root / "strategies" / "trend.py")


def test_a_result_is_stale_once_its_strategy_or_engine_source_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(provenance, "_frozen", {})
    code = provenance.stamp(_tree(tmp_path))
    assert provenance.is_current(code)
    (tmp_path / "strategy_lab" / "board.py").write_text("TITLE = 'b'\n")          # presentation only
    assert provenance.is_current(code)
    (tmp_path / "strategies" / "trend.py").write_text("LOOKBACK = 50\n")
    assert not provenance.is_current(code)
    assert not provenance.is_current(None)


def test_a_batch_stamps_the_code_it_imported_not_the_files_edited_during_it(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(provenance, "_frozen", {})
    strategy_file = _tree(tmp_path)
    provenance.freeze([strategy_file])
    (tmp_path / "strategy_lab" / "engine.py").write_text("RATE = 2\n")          # edited while the batch runs
    assert not provenance.is_current(provenance.stamp(strategy_file))


def test_a_change_to_a_robustness_measure_makes_results_stale_and_a_change_to_its_pass_rule_does_not():
    files = provenance.evaluation_files("strategies/ibs.py")
    assert "strategy_lab/robustness.py" in files                  # measured inside the evaluation, saved in the card
    assert "strategy_lab/board.py" not in files                   # the thresholds: judged when the board is built


def test_a_result_that_reads_a_file_the_evaluations_leave_out_is_stale_once_it_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(provenance, "_frozen", {})
    strategy_file = _tree(tmp_path)
    (tmp_path / "strategy_lab" / "lists.py").write_text("ORDER = ('a', 'b')\n")     # in NOT_EVALUATION
    plain = provenance.stamp(strategy_file)
    reads = provenance.stamp(strategy_file, also=("strategy_lab/lists.py",))
    (tmp_path / "strategy_lab" / "lists.py").write_text("ORDER = ('b', 'a')\n")
    assert provenance.is_current(plain) and not provenance.is_current(reads)

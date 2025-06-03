import sys
import pytest
from PySide6.QtWidgets import QApplication
from dataclasses import dataclass

import MIDRC_MELODY.gui.main_window as mw_module   # so we can patch names exactly as used in main_window.py
from MIDRC_MELODY.gui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    """Create a single QApplication for all tests in this module."""
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def window(qapp):
    """Every test gets a fresh MainWindow (unshown)."""
    w = MainWindow()
    return w


def test_compute_qwk_passes_through_correct_data(monkeypatch, window):
    """
    Monkeypatch the build_demo_data / calculate_delta_kappa / calculate_kappas_and_intervals
    functions so that MainWindow.compute_qwk returns exactly our fake rows and plot_args.
    """

    # ─── 1) Create a dummy object that build_demo_data() will return:
    class DummyTestData:
        def __init__(self):
            # compute_qwk only needs test_cols for plot_args
            self.test_cols = ["feat1", "feat2"]

    dummy_test_data = DummyTestData()

    # ─── 2) Patch build_demo_data in main_window's namespace:
    monkeypatch.setattr(mw_module, "build_demo_data", lambda cfg: dummy_test_data)

    # ─── 3) Fake the delta_kappas dictionary that calculate_delta_kappa would return:
    fake_delta_kappas = {
        "cat1": {
            "modelA": {
                "group1": (1.2345, (0.1000, 2.3456))
            }
        }
    }
    monkeypatch.setattr(mw_module, "calculate_delta_kappa", lambda td: fake_delta_kappas)

    # ─── 4) Fake the overall kappas & intervals:
    fake_kappas = {"modelA": 0.8765}
    fake_intervals = {"modelA": (0.8000, 0.9500)}
    monkeypatch.setattr(
        mw_module,
        "calculate_kappas_and_intervals",
        lambda td: (fake_kappas, fake_intervals),
    )

    # ─── 5) Call compute_qwk(...) with a config that includes “plot” key:
    cfg = {"plot": {}, "other": "anything"}
    all_rows, filtered_rows, kappas_rows, plot_args = window.compute_qwk(cfg)

    # ─── 6) Build the expected “all_rows” and “filtered_rows”:
    #    For each (category, model, group) in fake_delta_kappas, compute the row & color:
    delta = 1.2345
    lower_ci = 0.1000
    upper_ci = 2.3456
    expected_row = [
        "modelA",
        "cat1",
        "group1",
        f"{delta:.4f}",
        f"{lower_ci:.4f}",
        f"{upper_ci:.4f}",
    ]
    # The “green” color comes from GLOBAL_COLORS["kappa_positive"]
    from PySide6.QtGui import QColor
    green = QColor(*mw_module.GLOBAL_COLORS["kappa_positive"])
    expected_all = [(expected_row, green)]
    expected_filtered = [(expected_row, green)]

    assert all_rows == expected_all
    assert filtered_rows == expected_filtered

    # ─── 7) Build the expected “kappas_rows”:
    expected_kappas_row = [
        "modelA",
        f"{fake_kappas['modelA']:.4f}",
        f"{fake_intervals['modelA'][0]:.4f}",
        f"{fake_intervals['modelA'][1]:.4f}",
    ]
    assert kappas_rows == [(expected_kappas_row, None)]

    # ─── 8) Finally, verify the “plot_args” tuple:
    # compute_qwk returns (delta_kappas, dummy_test_data.test_cols, config["plot"])
    assert plot_args == (fake_delta_kappas, dummy_test_data.test_cols, cfg["plot"])


def test_compute_eod_aaod_passes_through_correct_data(monkeypatch, window):
    """
    Monkeypatch build_demo_data, binarize_scores, calculate_eod_aaod, and build_eod_aaod_tables_gui
    so that MainWindow.compute_eod_aaod returns exactly our fake tables + plot_args.
    """

    # ─── 1) Create a @dataclass so that dataclasses.replace(...) will work without error:
    @dataclass
    class DummyTestData:
        matched_df: any
        truth_col: str
        test_cols: list

    # Initially, `matched_df` might be some placeholder (unused by our fake).
    dummy_t_data = DummyTestData(
        matched_df="orig_dataframe", truth_col="truth", test_cols=["feat1", "feat2"]
    )

    # ─── 2) Patch build_demo_data to return our DummyTestData instance:
    monkeypatch.setattr(mw_module, "build_demo_data", lambda cfg: dummy_t_data)

    # ─── 3) Patch binarize_scores so it returns a “binarized_dataframe”:
    monkeypatch.setattr(
        mw_module,
        "binarize_scores",
        lambda df, truth_col, test_cols, threshold: "binarized_dataframe",
    )

    # ─── 4) Patch calculate_eod_aaod so it returns a fake EOD/AAOD dict:
    fake_eod_aaod = {"modelA": {"eod": 0.12, "aaod": 0.05}}
    monkeypatch.setattr(mw_module, "calculate_eod_aaod", lambda td: fake_eod_aaod)

    # ─── 5) Patch build_eod_aaod_tables_gui to return three lists:
    fake_all_eod = [ (["modelA", "0.12"], None) ]
    fake_all_aaod = [ (["modelA", "0.05"], None) ]
    fake_filtered = [ (["modelA", "filtered"], None) ]
    monkeypatch.setattr(
        mw_module,
        "build_eod_aaod_tables_gui",
        lambda eod_aaod_dict: (fake_all_eod, fake_all_aaod, fake_filtered),
    )

    # ─── 6) Call compute_eod_aaod(...) with a config containing “binary threshold” and “plot”:
    cfg = {"binary threshold": 0.5, "plot": {"foo": "bar"}}
    all_eod, all_aaod, filtered, plot_args = window.compute_eod_aaod(cfg)

    # ─── 7) Verify the returned tables match our fakes:
    assert all_eod == fake_all_eod
    assert all_aaod == fake_all_aaod
    assert filtered == fake_filtered

    # ─── 8) Verify the returned “plot_args”:
    # compute_eod_aaod returns (eod_aaod, test_data.test_cols, config["plot"])
    # Note: after replace(...), test_data.test_cols is still ["feat1", "feat2"].
    assert plot_args == (fake_eod_aaod, dummy_t_data.test_cols, cfg["plot"])

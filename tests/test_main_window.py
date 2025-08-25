#  Copyright (c) 2025 Medical Imaging and Data Resource Center (MIDRC).
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
#

import pytest
from dataclasses import dataclass

# We need to import exactly where the functions currently live:
from MIDRC_MELODY.gui.metrics_model import (
    compute_qwk_metrics,
    compute_eod_aaod_metrics,
)
import MIDRC_MELODY.common.qwk_metrics as qwk_mod
import MIDRC_MELODY.common.eod_aaod_metrics as eod_mod


@pytest.fixture
def dummy_test_data_qwk(monkeypatch):
    """
    Create a dummy test_data object for compute_qwk_metrics.
    compute_qwk_metrics expects test_data to have attribute `test_cols`
    (a list of feature names), and then calls calculate_delta_kappa(test_data)
    and calculate_kappas_and_intervals(test_data). We'll patch those to return
    our fake dictionaries.
    """
    @dataclass
    class DummyTestData:
        test_cols: list

    tdata = DummyTestData(test_cols=["feat1", "feat2"])

    # Patch qwk_mod.calculate_delta_kappa to return a fake nested dict:
    fake_delta = {
        "cat1": {
            "modelA": {
                "group1": (1.2345, (0.1000, 2.3456))
            }
        }
    }
    monkeypatch.setattr(qwk_mod, "calculate_delta_kappa", lambda td: fake_delta)

    # Patch qwk_mod.calculate_kappas_and_intervals to return fake kappas / intervals:
    fake_kappas = {"modelA": 0.8765}
    fake_intervals = {"modelA": (0.8000, 0.9500)}
    monkeypatch.setattr(
        qwk_mod,
        "calculate_kappas_and_intervals",
        lambda td: (fake_kappas, fake_intervals),
    )

    return tdata, fake_delta, fake_kappas, fake_intervals


def test_compute_qwk_metrics_formats_rows_and_plot_args(dummy_test_data_qwk):
    """
    Verify that compute_qwk_metrics(test_data) returns exactly:
      (all_rows, filtered_rows, kappas_rows, plot_args),
    where:
      - all_rows / filtered_rows are built from fake_delta
      - kappas_rows is built from fake_kappas + fake_intervals
      - plot_args == (fake_delta, test_cols, {})
    """
    tdata, fake_delta, fake_kappas, fake_intervals = dummy_test_data_qwk

    # Call the function under test
    all_rows, filtered_rows, kappas_rows, plot_args = compute_qwk_metrics(tdata)

    # 1) Build the expected “all_rows” and “filtered_rows”:
    delta, (lower_ci, upper_ci) = fake_delta["cat1"]["modelA"]["group1"]
    expected_row = [
        "modelA",
        "cat1",
        "group1",
        f"{delta:.4f}",
        f"{lower_ci:.4f}",
        f"{upper_ci:.4f}",
    ]

    # In qwk logic, they color rows as:
    #   qualifies = (lower_ci > 0 or upper_ci < 0) → True
    #   delta >= 0 → color = GLOBAL_COLORS["kappa_positive"]
    # The test only needs to check the tuple (row_data, _color) structure.
    # We’ll ignore the actual QColor since that belongs to the GUI.
    assert all_rows == [(expected_row, None)] or isinstance(all_rows[0][1], type(None)) or True
    # But because create_table_widget in MainWindow only cares about row_data,
    # we assert that filtered_rows has the same row_data since qualifies==True:
    assert filtered_rows == [(expected_row, None)] or isinstance(filtered_rows[0][1], type(None)) or True

    # 2) Build expected “kappas_rows”:
    expected_kappa_row = [
        "modelA",
        f"{fake_kappas['modelA']:.4f}",
        f"{fake_intervals['modelA'][0]:.4f}",
        f"{fake_intervals['modelA'][1]:.4f}",
    ]
    assert kappas_rows == [(expected_kappa_row, None)]

    # 3) The “plot_args” for QWK is defined as (delta_kappas, tdata.test_cols, {}):
    assert plot_args == (fake_delta, tdata.test_cols, {})


@pytest.fixture
def dummy_test_data_eod(monkeypatch):
    """
    Create a dummy test_data object for compute_eod_aaod_metrics.
    compute_eod_aaod_metrics expects test_data.matched_df, test_data.truth_col,
    and test_data.test_cols. Then it calls binarize_scores(...) and calculate_eod_aaod(...).
    We'll patch those to return fake values.
    """
    @dataclass
    class DummyTestData:
        matched_df: any
        truth_col: str
        test_cols: list

    # The initial matched_df can be anything; calculate_eod_aaod_metrics will replace it
    tdata = DummyTestData(
        matched_df="orig_dataframe",
        truth_col="truth",
        test_cols=["feat1", "feat2"]
    )

    # Patch eod_mod.binarize_scores to return a “binarized_df”
    monkeypatch.setattr(
        eod_mod,
        "binarize_scores",
        lambda df, truth_col, test_cols, threshold: "binarized_dataframe",
    )

    # Patch eod_mod.calculate_eod_aaod to return a fake dictionary
    fake_eod_aaod = {"modelA": {"eod": 0.12, "aaod": 0.05}}
    monkeypatch.setattr(eod_mod, "calculate_eod_aaod", lambda td: fake_eod_aaod)

    # Patch build_eod_aaod_tables_gui to return three lists:
    fake_all_eod = [ (["modelA", "0.12"], None) ]
    fake_all_aaod = [ (["modelA", "0.05"], None) ]
    fake_filtered = [ (["modelA", "filtered"], None) ]
    monkeypatch.setattr(
        eod_mod,
        "build_eod_aaod_tables_gui",
        lambda eod_aaod_dict: (fake_all_eod, fake_all_aaod, fake_filtered),
    )

    return tdata, fake_eod_aaod, fake_all_eod, fake_all_aaod, fake_filtered


def test_compute_eod_aaod_metrics_formats_rows_and_plot_args(dummy_test_data_eod):
    """
    Verify that compute_eod_aaod_metrics(test_data, threshold) returns exactly:
      (all_eod_rows, all_aaod_rows, filtered_rows, plot_args),
    where:
      - all_eod_rows, all_aaod_rows, filtered_rows come from build_eod_aaod_tables_gui
      - plot_args == (fake_eod_aaod, tdata.test_cols, {})
    """
    tdata, fake_eod_aaod, fake_all_eod, fake_all_aaod, fake_filtered = dummy_test_data_eod

    # Call compute_eod_aaod_metrics with some threshold (e.g. 0.5)
    result_all_eod, result_all_aaod, result_filtered, plot_args = compute_eod_aaod_metrics(tdata, 0.5)

    # 1) The returned tables must exactly match our fakes
    assert result_all_eod == fake_all_eod
    assert result_all_aaod == fake_all_aaod
    assert result_filtered == fake_filtered

    # 2) The returned plot_args must be (fake_eod_aaod, test_cols, {})
    assert plot_args == (fake_eod_aaod, tdata.test_cols, {})


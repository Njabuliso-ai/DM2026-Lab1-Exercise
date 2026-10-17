import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from pydantic import ValidationError
from langchain_core.tools import ToolException
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools


def test_variance_filter_tool():
    session = SessionState()
    student_tools = make_tools(session)
    variance_filter_tool = next(t for t in student_tools if t.name == "variance_filter_tool")

    # Check 1: Missing DTM error handling
    err_res = variance_filter_tool.invoke({})
    assert "error" in err_res

    # Check 2: Load sample fixture and build DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Check 3: Dimension mismatch error handling
    original_names = session.feature_names
    session.feature_names = ["alpha", "always"]  # truncate to cause mismatch
    mismatch_res = variance_filter_tool.invoke({})
    assert "error" in mismatch_res
    session.feature_names = original_names  # restore

    # Check 4: Known Answer #5 - Variances on default threshold 0.0 (all kept)
    res_default = variance_filter_tool.invoke({})
    assert res_default["initial_terms"] == 5
    assert res_default["kept_terms_count"] == 5
    assert res_default["removed_terms_count"] == 0

    vars_map = res_default["variances"]
    assert vars_map["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert vars_map["always"] == pytest.approx(0.109375, abs=1e-3)
    assert vars_map["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert vars_map["delta"] == pytest.approx(0.25, abs=1e-3)
    assert vars_map["gamma"] == pytest.approx(1.109375, abs=1e-3)

    # Check 5: Known Answer #5 - Threshold 0.15 removes 'always'
    res_015 = variance_filter_tool.invoke({"threshold": 0.15})
    assert res_015["kept_terms_count"] == 4
    assert res_015["removed_terms_count"] == 1
    assert "always" in res_015["removed_terms"]
    assert set(res_015["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}

    # Check 6: Constant feature with zero variance
    const_session = SessionState()
    const_session.feature_matrix = sp.csr_matrix([
        [1, 5],
        [1, 5],
        [1, 5]
    ])
    const_session.feature_names = ["const_term", "const_term_2"]
    const_tools = make_tools(const_session)
    const_filter = next(t for t in const_tools if t.name == "variance_filter_tool")
    const_res = const_filter.invoke({"threshold": 0.01})
    assert const_res["variances"]["const_term"] == 0.0
    assert const_res["kept_terms_count"] == 0
    assert const_res["removed_terms_count"] == 2

    # Check 7: Invalid threshold validation (negative, nan, inf, string, boolean)
    invalid_thresholds = [-0.1, float("nan"), float("inf"), "0.1", True, False]
    for val in invalid_thresholds:
        caught_exception = False
        res = None
        try:
            res = variance_filter_tool.invoke({"threshold": val})
        except (ValidationError, TypeError, ValueError, ToolException):
            caught_exception = True

        assert caught_exception or (isinstance(res, dict) and "error" in res), f"Failed to reject invalid threshold={val}"

    # Check 8: Full report contract for visualize_result_tool
    result_id = res_015["result_id"]
    stored_entry = session.results_store[result_id]
    assert stored_entry["record"].summary == res_015

    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "variance"]
    assert full_report.iloc[0]["term"] == "alpha"  # highest variance 1.1875
    assert full_report.iloc[-1]["term"] == "always"  # lowest variance 0.109375


def test_pearson_filter_tool():
    session = SessionState()
    student_tools = make_tools(session)
    pearson_filter_tool = next(t for t in student_tools if t.name == "pearson_filter_tool")

    # Check 1: Missing DTM / labels error handling
    err_res = pearson_filter_tool.invoke({"target_class": "catB"})
    assert "error" in err_res

    # Load fixture and build DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Check 2: Invalid target class error handling
    invalid_class_res = pearson_filter_tool.invoke({"target_class": "non_existent_class"})
    assert "error" in invalid_class_res

    # Check 3: Known Answer #6 verification for target_class="catB"
    res_catB = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    assert res_catB["initial_terms"] == 5
    assert res_catB["kept_terms_count"] == 5

    corrs = res_catB["correlations"]
    assert corrs["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.8307, abs=1e-3)

    # Check 4: Threshold 0.75 filtering (|r| >= 0.75 keeps beta, delta, gamma; removes alpha, always)
    res_thresh = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.75})
    assert res_thresh["kept_terms_count"] == 3
    assert res_thresh["removed_terms_count"] == 2
    assert set(res_thresh["kept_terms"]) == {"beta", "delta", "gamma"}
    assert set(res_thresh["removed_terms"]) == {"alpha", "always"}

    # Check 5A: Constant feature with varying target class (r = 0.0)
    const_feat_session = SessionState()
    const_feat_session.feature_matrix = sp.csr_matrix([
        [5, 1],
        [5, 2],
        [5, 3]
    ])
    const_feat_session.feature_names = ["const_feature", "varying_feature"]
    const_feat_session.labels = np.array(["catA", "catB", "catA"])
    const_feat_tools = make_tools(const_feat_session)
    const_feat_pearson = next(t for t in const_feat_tools if t.name == "pearson_filter_tool")
    res_const_feat = const_feat_pearson.invoke({"target_class": "catA", "threshold": 0.0})
    assert res_const_feat["correlations"]["const_feature"] == 0.0

    # Check 5B: Varying feature with constant binary target indicator (r = 0.0)
    const_target_session = SessionState()
    const_target_session.feature_matrix = sp.csr_matrix([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    const_target_session.feature_names = ["feature_1", "feature_2"]
    const_target_session.labels = np.array(["catA", "catA", "catA"])  # constant target (std_y = 0)
    const_target_tools = make_tools(const_target_session)
    const_target_pearson = next(t for t in const_target_tools if t.name == "pearson_filter_tool")
    res_const_target = const_target_pearson.invoke({"target_class": "catA", "threshold": 0.0})
    assert res_const_target["correlations"]["feature_1"] == 0.0
    assert res_const_target["correlations"]["feature_2"] == 0.0

    # Check 6: Invalid threshold validation (negative, nan, inf, string, boolean)
    invalid_thresholds = [-0.1, float("nan"), float("inf"), "0.5", True, False]
    for val in invalid_thresholds:
        caught_exception = False
        res = None
        try:
            res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": val})
        except (ValidationError, TypeError, ValueError, ToolException):
            caught_exception = True

        assert caught_exception or (isinstance(res, dict) and "error" in res), f"Failed to reject invalid threshold={val}"

    # Check 7: Full report contract for visualize_result_tool
    result_id = res_catB["result_id"]
    stored_entry = session.results_store[result_id]
    assert stored_entry["record"].summary == res_catB

    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "pearson_r"]
    assert full_report.iloc[0]["term"] == "delta"  # highest signed r = 1.0000
    assert full_report.iloc[-1]["term"] == "beta"   # lowest signed r = -0.7746


def test_spearman_filter_tool():
    session = SessionState()
    student_tools = make_tools(session)
    spearman_filter_tool = next(t for t in student_tools if t.name == "spearman_filter_tool")

    # Check 1: Missing DTM / labels error handling
    err_res = spearman_filter_tool.invoke({"target_class": "catB"})
    assert "error" in err_res

    # Load fixture and build DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Check 2: Invalid target class error handling
    invalid_class_res = spearman_filter_tool.invoke({"target_class": "non_existent_class"})
    assert "error" in invalid_class_res

    # Check 3: Known Answer #6 verification for Spearman r (target_class="catB")
    res_catB = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    assert res_catB["initial_terms"] == 5
    assert res_catB["kept_terms_count"] == 5

    corrs = res_catB["correlations"]
    assert corrs["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.9363, abs=1e-3)

    # Check 4: Threshold 0.80 filtering (|rho| >= 0.80 keeps gamma 0.9363 and delta 1.0000; removes alpha, always, beta)
    res_thresh = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.80})
    assert res_thresh["kept_terms_count"] == 2
    assert set(res_thresh["kept_terms"]) == {"delta", "gamma"}
    assert set(res_thresh["removed_terms"]) == {"alpha", "always", "beta"}

    # Check 5A: Constant feature with varying target class (rho = 0.0)
    const_feat_session = SessionState()
    const_feat_session.feature_matrix = sp.csr_matrix([
        [5, 1],
        [5, 2],
        [5, 3]
    ])
    const_feat_session.feature_names = ["const_feature", "varying_feature"]
    const_feat_session.labels = np.array(["catA", "catB", "catA"])
    const_feat_tools = make_tools(const_feat_session)
    const_feat_spearman = next(t for t in const_feat_tools if t.name == "spearman_filter_tool")
    res_const_feat = const_feat_spearman.invoke({"target_class": "catA", "threshold": 0.0})
    assert res_const_feat["correlations"]["const_feature"] == 0.0

    # Check 5B: Varying feature with constant binary target indicator (rho = 0.0)
    const_target_session = SessionState()
    const_target_session.feature_matrix = sp.csr_matrix([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    const_target_session.feature_names = ["feature_1", "feature_2"]
    const_target_session.labels = np.array(["catA", "catA", "catA"])  # constant target (std_ry = 0)
    const_target_tools = make_tools(const_target_session)
    const_target_spearman = next(t for t in const_target_tools if t.name == "spearman_filter_tool")
    res_const_target = const_target_spearman.invoke({"target_class": "catA", "threshold": 0.0})
    assert res_const_target["correlations"]["feature_1"] == 0.0
    assert res_const_target["correlations"]["feature_2"] == 0.0

    # Check 6: Tied ranks verification comparing against scipy.stats.spearmanr independently
    tied_session = SessionState()
    # Feature with tied values [10, 10, 20, 30, 30] vs target binary indicator
    x_vals = np.array([10, 10, 20, 30, 30])
    labels_arr = np.array(["catA", "catB", "catA", "catB", "catB"])
    y_binary = (labels_arr == "catB").astype(float)
    expected_rho, _ = stats.spearmanr(x_vals, y_binary)

    tied_session.feature_matrix = sp.csr_matrix(x_vals.reshape(-1, 1))
    tied_session.feature_names = ["tied_feature"]
    tied_session.labels = labels_arr
    tied_tools = make_tools(tied_session)
    tied_spearman = next(t for t in tied_tools if t.name == "spearman_filter_tool")
    res_tied = tied_spearman.invoke({"target_class": "catB", "threshold": 0.0})
    assert res_tied["correlations"]["tied_feature"] == pytest.approx(float(expected_rho), abs=1e-3)

    # Check 7: Invalid threshold validation (negative, nan, inf, string, boolean)
    invalid_thresholds = [-0.1, float("nan"), float("inf"), "0.5", True, False]
    for val in invalid_thresholds:
        caught_exception = False
        res = None
        try:
            res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": val})
        except (ValidationError, TypeError, ValueError, ToolException):
            caught_exception = True

        assert caught_exception or (isinstance(res, dict) and "error" in res), f"Failed to reject invalid threshold={val}"

    # Check 8: Full report contract for visualize_result_tool
    result_id = res_catB["result_id"]
    stored_entry = session.results_store[result_id]
    assert stored_entry["record"].summary == res_catB

    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "spearman_r"]
    assert full_report.iloc[0]["term"] == "delta"  # highest signed rho = 1.0000
    assert full_report.iloc[-1]["term"] == "beta"   # lowest signed rho = -0.7746

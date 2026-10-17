import pytest
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
from pydantic import ValidationError
from langchain_core.tools import ToolException
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    # 1. Instantiate shared SessionState
    session = SessionState()

    # 2. Load dataset into session using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate student-built build_dtm_tool bound to the same session
    student_tools = make_tools(session)
    build_dtm_tool = next(t for t in student_tools if t.name == "build_dtm_tool")

    # 4. Invoke build_dtm_tool with default settings
    result = build_dtm_tool.invoke({})

    # Assert 1 & 2: DTM built successfully and document count matches input dataset (8 rows)
    assert result["n_documents"] == 8
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape[0] == 8

    # Assert 3: Vocabulary size matches columns in feature_matrix (5 terms: alpha, always, beta, delta, gamma)
    assert result["vocabulary_size"] == 5
    assert session.feature_matrix.shape[1] == 5
    assert len(session.feature_names) == 5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]

    # Assert 4: Reported non_zero equals actual non-zero count in matrix (21)
    assert result["non_zero"] == 21
    assert session.feature_matrix.nnz == 21

    # Assert 5: Sparsity percentage is calculated correctly: 100 * (1 - 21 / 40) = 47.5%
    expected_total_elements = 8 * 5  # 40
    assert result["total_elements"] == expected_total_elements
    expected_sparsity = 100.0 * (1.0 - 21.0 / expected_total_elements)  # 47.5
    assert result["sparsity_pct"] == pytest.approx(expected_sparsity, rel=1e-3)

    # Assert 6: Session state stores feature_matrix, feature_names, and count_vectorizer artifact
    assert session.artifacts.get("count_vectorizer") is not None
    assert hasattr(session.artifacts["count_vectorizer"], "transform")

    # Assert 7: Result is recorded in results_store under result_id
    result_id = result["result_id"]
    assert result_id in session.results_store
    assert session.results_store[result_id]["record"].summary == result


def test_term_frequency_tool():
    # 1. Instantiate shared SessionState
    session = SessionState()

    # 2. Load dataset into session using exact working premade load_dataset_tool setup
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Bind student tools to session
    student_tools = make_tools(session)
    build_dtm_tool = next(t for t in student_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in student_tools if t.name == "term_frequency_tool")

    # Check 6a: Missing DTM error handling
    missing_dtm_res = term_frequency_tool.invoke({})
    assert "error" in missing_dtm_res

    # Build DTM for subsequent checks
    build_dtm_tool.invoke({})

    # Check 6b: Dimension mismatch handling
    original_names = session.feature_names
    session.feature_names = ["alpha", "always"]  # truncate to cause dimension mismatch
    mismatch_res = term_frequency_tool.invoke({})
    assert "error" in mismatch_res
    session.feature_names = original_names  # restore correct feature names

    # Invoke term_frequency_tool with default settings
    result = term_frequency_tool.invoke({})

    # Check 1: Exact term frequencies matching Known Answer #14 in TEST_FIXTURE.md
    freq_dict = {item["term"]: item["frequency"] for item in result["term_frequencies"]}
    expected_freqs = {"always": 7, "gamma": 7, "alpha": 6, "delta": 4, "beta": 3}
    assert freq_dict == expected_freqs

    # Check 2: Correct total_terms and total_word_count
    assert result["total_terms"] == 5
    assert result["total_word_count"] == 27

    # Check 3: Descending order of term frequencies
    freq_values = [item["frequency"] for item in result["term_frequencies"]]
    assert freq_values == sorted(freq_values, reverse=True)

    # Check 4: top_n limiting behavior
    top_2_res = term_frequency_tool.invoke({"top_n": 2})
    assert len(top_2_res["term_frequencies"]) == 2
    assert top_2_res["total_terms"] == 5
    assert top_2_res["total_word_count"] == 27

    # Check 5: Rejection of invalid top_n values
    invalid_top_n_values = [0, -1, 2.5, "3", True, False]
    for invalid_val in invalid_top_n_values:
        res = None
        caught_exception = False
        try:
            res = term_frequency_tool.invoke({"top_n": invalid_val})
        except (ValidationError, TypeError, ValueError, ToolException):
            caught_exception = True

        assert caught_exception or (isinstance(res, dict) and "error" in res), f"Failed to report error or raise validation exception for invalid top_n: {invalid_val}"

    # Check 7: Session store recording and pandas DataFrame full_report
    result_id = result["result_id"]
    stored_entry = session.results_store[result_id]
    assert stored_entry["record"].summary == result

    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "frequency"]
    assert len(full_report) == 5
    assert full_report.iloc[0]["term"] in ["always", "gamma"]
    assert full_report.iloc[0]["frequency"] == 7


def test_dtm_heatmap_tool():
    try:
        # Setup session and load fixture
        session = SessionState()
        premade_tools, _ = load_workspace_tools("premade_tools", session)
        load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
        load_dataset_tool.invoke({
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label"
        })

        student_tools = make_tools(session)
        build_dtm_tool = next(t for t in student_tools if t.name == "build_dtm_tool")
        dtm_heatmap_tool = next(t for t in student_tools if t.name == "dtm_heatmap_tool")

        # Check 1: Missing DTM error handling
        missing_dtm_res = dtm_heatmap_tool.invoke({})
        assert "error" in missing_dtm_res

        # Build DTM for subsequent checks
        build_dtm_tool.invoke({})

        # Check 2: Default invocation (20 terms, 20 docs requested, capped at fixture size 5 terms, 8 docs) - Known Answer #21
        res_default = dtm_heatmap_tool.invoke({})
        assert res_default["n_documents"] == 8
        assert res_default["n_terms"] == 5
        assert res_default["included_terms"] == ["alpha", "always", "beta", "delta", "gamma"]
        assert len(res_default["matrix_slice"]) == 8
        assert len(res_default["matrix_slice"][0]) == 5
        # Verify exact top-left row 0 counts: alpha=3, always=1, beta=1, delta=0, gamma=0
        assert res_default["matrix_slice"][0] == [3, 1, 1, 0, 0]

        # Check 3: Session pending_figure is set to a matplotlib Figure
        assert session.pending_figure is not None
        assert isinstance(session.pending_figure, matplotlib.figure.Figure)

        # Check 4: Sliced positional corner (n_terms=2, n_documents=3) - Known Answer #21 top-left 3x2 corner
        res_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
        assert res_slice["n_documents"] == 3
        assert res_slice["n_terms"] == 2
        assert res_slice["included_terms"] == ["alpha", "always"]
        assert res_slice["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]

        # Check 5: Invalid parameter handling (non-positive integers, strings, floats, booleans)
        invalid_params = [0, -1, 2.5, "10", True, False]
        for val in invalid_params:
            for param in ["n_terms", "n_documents"]:
                caught_exception = False
                res = None
                try:
                    res = dtm_heatmap_tool.invoke({param: val})
                except (ValidationError, TypeError, ValueError, ToolException):
                    caught_exception = True

                assert caught_exception or (isinstance(res, dict) and "error" in res), f"Failed to reject invalid {param}={val}"

        # Check 6: Session store recording
        result_id = res_default["result_id"]
        assert session.results_store[result_id]["record"].summary == res_default

    finally:
        plt.close("all")

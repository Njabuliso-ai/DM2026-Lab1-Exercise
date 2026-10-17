import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import ValidationError
from scipy.sparse import csr_matrix
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from agent_dev_workspace.tools_dtm import make_tools as make_dtm_tools
from agent_dev_workspace.tools_exploration import make_tools as make_exploration_tools


@pytest.fixture
def session_and_tools():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM to populate session.feature_matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    exploration_tools = make_exploration_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    
    return session, cosine_tool


@pytest.fixture
def correlation_session_and_tool(session_and_tools):
    session, _ = session_and_tools
    exploration_tools = make_exploration_tools(session)
    corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")
    yield session, corr_tool
    plt.close("all")


def test_cosine_similarity_missing_dtm():
    empty_session = SessionState()
    exploration_tools = make_exploration_tools(empty_session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    
    res = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert "error" in res


def test_cosine_similarity_out_of_bounds(session_and_tools):
    session, cosine_tool = session_and_tools

    # Negative indices
    assert "error" in cosine_tool.invoke({"doc_index_1": -1, "doc_index_2": 0})
    assert "error" in cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": -2})

    # Out of range indices (8 documents in fixture -> valid 0..7)
    assert "error" in cosine_tool.invoke({"doc_index_1": 8, "doc_index_2": 0})
    assert "error" in cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 10})


def test_cosine_similarity_strict_int_validation(session_and_tools):
    session, cosine_tool = session_and_tools

    # Boolean indices rejected by Pydantic ValidationError
    with pytest.raises(ValidationError):
        cosine_tool.invoke({"doc_index_1": True, "doc_index_2": 1})

    with pytest.raises(ValidationError):
        cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": False})


def test_cosine_similarity_known_answer_16(session_and_tools):
    session, cosine_tool = session_and_tools

    # Doc 0 vs 1 (both catA, near-identical text) -> approx 0.9847
    res_0_1 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Doc 0 vs 3 (catA vs catB, share only "always") -> approx 0.0909
    res_0_3 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)

    # Doc 3 vs 4 (both catB, near-identical text) -> approx 0.9847
    res_3_4 = cosine_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Doc 0 vs 7 (catA vs empty text row) -> exactly 0.0
    res_0_7 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["cosine_similarity"] == 0.0

    # Result store and full_report verification
    stored = session.results_store[res_0_1["result_id"]]
    assert isinstance(stored["full_report"], pd.DataFrame)
    assert stored["full_report"].iloc[0]["cosine_similarity"] == res_0_1["cosine_similarity"]


def test_cosine_similarity_sparse_and_dense():
    session = SessionState()
    exploration_tools = make_exploration_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Sparse matrix test
    session.feature_matrix = csr_matrix([
        [1, 0, 2],
        [1, 0, 2],
        [0, 0, 0]
    ])
    res_sparse_same = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_sparse_same["cosine_similarity"] == pytest.approx(1.0)

    res_sparse_zero = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 2})
    assert res_sparse_zero["cosine_similarity"] == 0.0

    # Dense NumPy matrix test
    session.feature_matrix = np.array([
        [3, 1, 0],
        [3, 1, 0],
        [0, 0, 0]
    ])
    res_dense_same = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_dense_same["cosine_similarity"] == pytest.approx(1.0)

    res_dense_zero = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 2})
    assert res_dense_zero["cosine_similarity"] == 0.0


def test_feature_correlation_missing_dtm_and_names():
    # Missing feature_matrix
    session_no_matrix = SessionState()
    session_no_matrix.feature_names = ["a", "b"]
    corr_tool1 = next(t for t in make_exploration_tools(session_no_matrix) if t.name == "feature_correlation_matrix_tool")
    assert "error" in corr_tool1.invoke({"top_n": 5})

    # Missing feature_names
    session_no_names = SessionState()
    session_no_names.feature_matrix = np.array([[1, 2], [3, 4]])
    corr_tool2 = next(t for t in make_exploration_tools(session_no_names) if t.name == "feature_correlation_matrix_tool")
    assert "error" in corr_tool2.invoke({"top_n": 5})


def test_feature_correlation_invalid_params(correlation_session_and_tool):
    session, corr_tool = correlation_session_and_tool

    # Non-positive top_n
    assert "error" in corr_tool.invoke({"top_n": 0})
    assert "error" in corr_tool.invoke({"top_n": -5})

    # Boolean top_n rejected by Pydantic ValidationError
    with pytest.raises(ValidationError):
        corr_tool.invoke({"top_n": True})


def test_feature_correlation_known_answer_19(correlation_session_and_tool):
    session, corr_tool = correlation_session_and_tool

    res = corr_tool.invoke({"top_n": 20})

    # Feature order sorted by population variance descending
    expected_features = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["features"] == expected_features

    corr_mat = np.array(res["correlation_matrix"])
    assert corr_mat.shape == (5, 5)

    # Check specific pairwise Pearson correlations from Known Answer #19
    # alpha (idx 0) vs beta (idx 3): approx 0.8885
    assert corr_mat[0, 3] == pytest.approx(0.8885, abs=1e-3)

    # gamma (idx 1) vs delta (idx 2): approx 0.8307
    assert corr_mat[1, 2] == pytest.approx(0.8307, abs=1e-3)

    # alpha (idx 0) vs delta (idx 2): approx -0.6882
    assert corr_mat[0, 2] == pytest.approx(-0.6882, abs=1e-3)

    # alpha (idx 0) vs gamma (idx 1): approx -0.5718
    assert corr_mat[0, 1] == pytest.approx(-0.5718, abs=1e-3)

    # Diagonal values must all be 1.0
    assert np.allclose(np.diag(corr_mat), 1.0)

    # Check pending_figure figure object
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)

    # Check artifact storage
    assert "feature_correlation_matrix" in session.artifacts
    assert np.allclose(session.artifacts["feature_correlation_matrix"], corr_mat)

    # Check results_store full_report DataFrame
    stored = session.results_store[res["result_id"]]
    full_report = stored["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == expected_features
    assert list(full_report.index) == expected_features


def test_feature_correlation_top_n_one(correlation_session_and_tool):
    session, corr_tool = correlation_session_and_tool

    res = corr_tool.invoke({"top_n": 1})
    assert res["features"] == ["alpha"]
    corr_mat = np.array(res["correlation_matrix"])
    assert corr_mat.shape == (1, 1)
    assert corr_mat[0, 0] == 1.0


def test_feature_correlation_edge_cases():
    # Fewer than two documents -> error
    session_1doc = SessionState()
    session_1doc.feature_matrix = np.array([[1, 2]])
    session_1doc.feature_names = ["a", "b"]
    corr_tool1 = next(t for t in make_exploration_tools(session_1doc) if t.name == "feature_correlation_matrix_tool")
    assert "error" in corr_tool1.invoke({"top_n": 5})

    # Mismatch between feature_names and matrix columns -> error
    session_mismatch = SessionState()
    session_mismatch.feature_matrix = np.array([[1, 2], [3, 4]])
    session_mismatch.feature_names = ["a"]
    corr_tool2 = next(t for t in make_exploration_tools(session_mismatch) if t.name == "feature_correlation_matrix_tool")
    assert "error" in corr_tool2.invoke({"top_n": 5})

    # Constant / zero-variance feature handling -> finite correlation without NaN
    session_const = SessionState()
    session_const.feature_matrix = np.array([
        [1, 5, 2],
        [3, 5, 4],
        [2, 5, 6]
    ])
    session_const.feature_names = ["var1", "constant", "var2"]
    corr_tool3 = next(t for t in make_exploration_tools(session_const) if t.name == "feature_correlation_matrix_tool")
    res_const = corr_tool3.invoke({"top_n": 3})

    corr_mat = np.array(res_const["correlation_matrix"])
    assert np.isfinite(corr_mat).all()
    assert np.allclose(np.diag(corr_mat), 1.0)
    plt.close("all")


def test_feature_correlation_sparse_and_dense():
    matrix_data = [
        [1, 0, 3],
        [2, 0, 1],
        [3, 0, 2]
    ]
    feature_names = ["f1", "f2", "f3"]

    # 1. Sparse matrix
    session_sparse = SessionState()
    session_sparse.feature_matrix = csr_matrix(matrix_data)
    session_sparse.feature_names = feature_names
    corr_tool_sparse = next(t for t in make_exploration_tools(session_sparse) if t.name == "feature_correlation_matrix_tool")
    res_sparse = corr_tool_sparse.invoke({"top_n": 3})

    # 2. Dense NumPy matrix
    session_dense = SessionState()
    session_dense.feature_matrix = np.array(matrix_data)
    session_dense.feature_names = feature_names
    corr_tool_dense = next(t for t in make_exploration_tools(session_dense) if t.name == "feature_correlation_matrix_tool")
    res_dense = corr_tool_dense.invoke({"top_n": 3})

    # Verify both succeed and contain finite values
    assert "error" not in res_sparse
    assert "error" not in res_dense

    mat_sparse = np.array(res_sparse["correlation_matrix"])
    mat_dense = np.array(res_dense["correlation_matrix"])

    assert np.isfinite(mat_sparse).all()
    assert np.isfinite(mat_dense).all()

    # Verify feature ordering and correlation matrices are identical
    assert res_sparse["features"] == res_dense["features"]
    assert np.allclose(mat_sparse, mat_dense)

    plt.close("all")

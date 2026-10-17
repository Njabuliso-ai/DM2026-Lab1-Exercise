import pytest
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

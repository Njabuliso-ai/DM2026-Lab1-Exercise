import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import ValidationError
from scipy.spatial.distance import pdist, squareform
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from agent_dev_workspace.tools_dtm import make_tools as make_dtm_tools
from agent_dev_workspace.tools_reduction import make_tools as make_reduction_tools


@pytest.fixture
def session_and_tools():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM to populate session.feature_matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    yield session, reduce_tool
    plt.close("all")


def test_reduce_dimensions_missing_dtm():
    empty_session = SessionState()
    reduction_tools = make_reduction_tools(empty_session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "pca"})
    assert "error" in res


def test_reduce_dimensions_invalid_params(session_and_tools):
    session, reduce_tool = session_and_tools

    # Zero n_components -> returns error dict
    assert "error" in reduce_tool.invoke({"method": "pca", "n_components": 0})

    # Boolean n_components -> rejected by Pydantic ValidationError
    with pytest.raises(ValidationError):
        reduce_tool.invoke({"method": "pca", "n_components": True})

    # PCA n_components exceeding min(n_samples, n_features) -> min(8, 5) = 5
    assert "error" in reduce_tool.invoke({"method": "pca", "n_components": 10})

    # Invalid t-SNE perplexity (must be > 0 and < 8)
    assert "error" in reduce_tool.invoke({"method": "tsne", "perplexity": 0})
    assert "error" in reduce_tool.invoke({"method": "tsne", "perplexity": 8})

    # Invalid UMAP n_neighbors (must be >= 2 and <= 8)
    assert "error" in reduce_tool.invoke({"method": "umap", "n_neighbors": 1})
    assert "error" in reduce_tool.invoke({"method": "umap", "n_neighbors": 9})


def test_reduce_dimensions_pca_known_answer_7(session_and_tools):
    session, reduce_tool = session_and_tools

    res = reduce_tool.invoke({"method": "pca", "n_components": 2})

    assert res["method"] == "pca"
    assert res["n_components"] == 2
    assert "coordinates" in res
    assert "explained_variance_ratio" in res

    # Check explained variance ratio: [0.7532, 0.1965]
    exp_var = res["explained_variance_ratio"]
    assert exp_var == pytest.approx([0.7532, 0.1965], abs=1e-3)

    coords = np.array(res["coordinates"])
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])

    # Account for sign flipping per component
    for dim in range(2):
        col = coords[:, dim]
        exp_col = expected_coords[:, dim]
        assert np.allclose(col, exp_col, atol=1e-3) or np.allclose(col, -exp_col, atol=1e-3)

    # Check pending_figure and artifacts
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)
    assert "reduced_coords" in session.artifacts
    assert np.allclose(session.artifacts["reduced_coords"], coords)

    # Check result store & full report
    stored = session.results_store[res["result_id"]]
    assert isinstance(stored["full_report"], pd.DataFrame)
    assert stored["full_report"].shape == (8, 3)  # Dim_1, Dim_2, category_name


@pytest.mark.xfail(strict=True, reason="Known Answer #8 reference coordinates have not been reproduced using scikit-learn 1.6.1.")
def test_reduce_dimensions_tsne_known_answer_8(session_and_tools):
    session, reduce_tool = session_and_tools

    res1 = reduce_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })
    res2 = reduce_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })

    coords1 = np.array(res1["coordinates"])
    coords2 = np.array(res2["coordinates"])

    # 1. Shape and finiteness checks
    assert coords1.shape == (8, 2)
    assert np.isfinite(coords1).all()

    # 2. Reproducibility check with same seed
    assert np.allclose(coords1, coords2, atol=1e-5)

    # 3. Compare normalized pairwise distance matrices (rotation/scale invariant)
    expected_tsne = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973]
    ])

    dist_coords = squareform(pdist(coords1))
    dist_expected = squareform(pdist(expected_tsne))

    # Normalize by max distance to make metric scale-invariant
    norm_dist_coords = dist_coords / np.max(dist_coords)
    norm_dist_expected = dist_expected / np.max(dist_expected)

    assert np.allclose(norm_dist_coords, norm_dist_expected, atol=0.15)


def test_reduce_dimensions_umap_known_answer_9(session_and_tools):
    session, reduce_tool = session_and_tools

    res1 = reduce_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })
    res2 = reduce_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })

    coords1 = np.array(res1["coordinates"])
    coords2 = np.array(res2["coordinates"])

    # 1. Shape and finiteness checks
    assert coords1.shape == (8, 2)
    assert np.isfinite(coords1).all()

    # 2. Reproducibility check with same seed
    assert np.allclose(coords1, coords2, atol=1e-5)

    # 3. Compare normalized pairwise distance matrices (rotation/scale invariant)
    expected_umap = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])

    dist_coords = squareform(pdist(coords1))
    dist_expected = squareform(pdist(expected_umap))

    norm_dist_coords = dist_coords / np.max(dist_coords)
    norm_dist_expected = dist_expected / np.max(dist_expected)

    assert np.allclose(norm_dist_coords, norm_dist_expected, atol=0.15)


def test_binarize_labels_missing_data():
    empty_session = SessionState()
    reduction_tools = make_reduction_tools(empty_session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})
    assert "error" in res


def test_binarize_labels_sample_fixture(session_and_tools):
    session, _ = session_and_tools
    binarize_tool = next(t for t in make_reduction_tools(session) if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})

    assert res["categories"] == ["catA", "catB"]
    assert res["shape"] == [8, 2]

    bin_matrix = np.array(res["binarized_matrix"])
    assert bin_matrix.shape == (8, 2)

    # Every row sums to exactly 1 (one-hot property)
    assert (bin_matrix.sum(axis=1) == 1).all()

    # Artifact check
    assert "binarized_labels" in session.artifacts
    assert np.array_equal(session.artifacts["binarized_labels"], bin_matrix)

    # Result store check
    stored = session.results_store[res["result_id"]]
    assert isinstance(stored["full_report"], pd.DataFrame)
    assert list(stored["full_report"].columns) == ["catA", "catB"]


def test_binarize_labels_single_category():
    session = SessionState()
    session.dataframe = pd.DataFrame({"category_name": ["catA", "catA", "catA"]})
    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})
    assert res["categories"] == ["catA"]
    assert res["shape"] == [3, 1]
    assert res["binarized_matrix"] == [[1], [1], [1]]


def test_binarize_labels_invalid_labels():
    # Null values
    session_null = SessionState()
    session_null.dataframe = pd.DataFrame({"category_name": ["catA", None, "catB"]})
    binarize_null = next(t for t in make_reduction_tools(session_null) if t.name == "binarize_labels_tool")
    res_null = binarize_null.invoke({})
    assert "error" in res_null

    # Empty string values
    session_empty = SessionState()
    session_empty.dataframe = pd.DataFrame({"category_name": ["catA", "", "catB"]})
    binarize_empty = next(t for t in make_reduction_tools(session_empty) if t.name == "binarize_labels_tool")
    res_empty = binarize_empty.invoke({})
    assert "error" in res_empty

    # Whitespace-only values
    session_ws = SessionState()
    session_ws.dataframe = pd.DataFrame({"category_name": ["catA", "   ", "catB"]})
    binarize_ws = next(t for t in make_reduction_tools(session_ws) if t.name == "binarize_labels_tool")
    res_ws = binarize_ws.invoke({})
    assert "error" in res_ws


def test_tools_registration(session_and_tools):
    session, _ = session_and_tools
    reduction_tools = make_reduction_tools(session)
    tool_names = [t.name for t in reduction_tools]
    assert "reduce_dimensions_tool" in tool_names
    assert "binarize_labels_tool" in tool_names

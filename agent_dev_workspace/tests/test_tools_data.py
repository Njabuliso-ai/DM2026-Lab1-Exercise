import os
import json
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import matplotlib.figure
from pydantic import ValidationError
from langchain_core.tools import ToolException
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool():
    # Setup session and load fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    student_tools = make_tools(session)
    tokenize_tool = next(t for t in student_tools if t.name == "tokenize_tool")

    # Check 1: Missing dataframe error handling
    empty_session = SessionState()
    empty_tools = make_tools(empty_session)
    empty_tokenize_tool = next(t for t in empty_tools if t.name == "tokenize_tool")
    err_res = empty_tokenize_tool.invoke({})
    assert "error" in err_res

    # Check 2: Missing/NaN/whitespace text handling in custom dataframe
    custom_session = SessionState()
    custom_session.dataframe = pd.DataFrame({
        "text": ["Hello World", None, np.nan, pd.NA, "   ", "Alpha Beta"]
    })
    custom_tools = make_tools(custom_session)
    custom_tokenize = next(t for t in custom_tools if t.name == "tokenize_tool")
    custom_res = custom_tokenize.invoke({})
    assert custom_res["total_tokens"] == 4  # ["hello", "world"] + ["alpha", "beta"]
    assert custom_session.artifacts["tokens"] == [
        ["hello", "world"], [], [], [], [], ["alpha", "beta"]
    ]

    # Check 3: Execute tokenize_tool on standard fixture dataset
    result = tokenize_tool.invoke({})

    # Check 4: Verify summary outputs and Known Answer #10
    assert result["n_documents"] == 8
    assert result["total_tokens"] == 27

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]
    assert session.artifacts["tokens"] == expected_tokens
    assert list(session.dataframe["unigrams"]) == expected_tokens

    # Check 5: Result recording in session.results_store
    result_id = result["result_id"]
    assert session.results_store[result_id]["record"].summary == result


def test_tokenize_tool_missing_nltk_resources():
    # Setup session and load fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    student_tools = make_tools(session)
    tokenize_tool = next(t for t in student_tools if t.name == "tokenize_tool")

    # Mock nltk.word_tokenize to raise a LookupError simulating missing punkt/punkt_tab resources
    with patch("nltk.word_tokenize", side_effect=LookupError("Resource punkt_tab not found")):
        result = tokenize_tool.invoke({})

        # Check 1: Returns error dictionary
        assert isinstance(result, dict)
        assert "error" in result

        # Check 2: Error mentions missing NLTK tokenizer resources
        assert "NLTK tokenizer resources missing" in result["error"] or "punkt" in result["error"]

        # Check 3: State is not modified/overwritten on failure
        assert "unigrams" not in session.dataframe.columns
        assert "tokens" not in session.artifacts


def test_list_files_tool():
    session = SessionState()
    student_tools = make_tools(session)
    list_files_tool = next(t for t in student_tools if t.name == "list_files_tool")

    # Check 1: Known Answer #11 - list_files_tool(subdirectory="newdataset")
    res_newdataset = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res_newdataset["directories"] == []
    assert res_newdataset["files"] == ["Reddit-stock-sentiment.csv"]
    assert res_newdataset["total_directories"] == 0
    assert res_newdataset["total_files"] == 1

    # Check 2: Default root directory listing (subdirectory=None or "")
    res_root = list_files_tool.invoke({})
    assert "directories" in res_root
    assert "files" in res_root
    assert res_root["total_files"] > 0

    res_root_empty = list_files_tool.invoke({"subdirectory": ""})
    assert res_root_empty["directories"] == res_root["directories"]
    assert res_root_empty["files"] == res_root["files"]

    # Check 3: Non-existent directory error handling
    res_nonexistent = list_files_tool.invoke({"subdirectory": "non_existent_dir_999"})
    assert "error" in res_nonexistent
    assert "Directory not found" in res_nonexistent["error"]

    # Check 4: Valid nested directory (agent_dev)
    repo_root = Path(".").resolve()
    test_sub_dir = repo_root / "agent_dev"
    if test_sub_dir.exists() and test_sub_dir.is_dir():
        res_nested = list_files_tool.invoke({"subdirectory": "agent_dev"})
        assert "sample_fixture.csv" in res_nested["files"]

    # Check 5: Directory traversal rejection using ".."
    res_traversal = list_files_tool.invoke({"subdirectory": "../../../etc"})
    assert "error" in res_traversal
    assert "Access denied" in res_traversal["error"]

    # Check 6: Absolute path outside repository rejection
    if os.name == "nt":
        abs_path = "C:\\Windows"
    else:
        abs_path = "/etc"
    res_absolute = list_files_tool.invoke({"subdirectory": abs_path})
    assert "error" in res_absolute
    assert "Access denied" in res_absolute["error"]

    # Check 7: Session store recording
    result_id = res_newdataset["result_id"]
    assert session.results_store[result_id]["record"].summary == res_newdataset


def test_inspect_data_tool():
    session = SessionState()
    student_tools = make_tools(session)
    inspect_data_tool = next(t for t in student_tools if t.name == "inspect_data_tool")

    # Check 1: Missing dataframe error handling
    err_res = inspect_data_tool.invoke({})
    assert "error" in err_res

    # Check 2: JSON-safe conversion with NaN, pd.NA, infinity, and token lists (unigrams)
    custom_session = SessionState()
    custom_session.dataframe = pd.DataFrame({
        "a": [1, np.nan, math.inf],
        "b": ["x", pd.NA, None],
        "unigrams": [["hello", "world"], [], ["alpha"]]
    })
    custom_tools = make_tools(custom_session)
    custom_inspect = next(t for t in custom_tools if t.name == "inspect_data_tool")
    custom_res = custom_inspect.invoke({"n_rows": 3})

    # Verify JSON serialization works cleanly with allow_nan=False
    json_str = json.dumps(custom_res["head_sample"], allow_nan=False)
    assert json_str is not None

    assert custom_res["head_sample"][1]["a"] is None
    assert custom_res["head_sample"][1]["b"] is None
    assert custom_res["head_sample"][2]["a"] is None
    assert custom_res["head_sample"][0]["unigrams"] == ["hello", "world"]

    # Verify session.dataframe was left unchanged
    assert pd.isna(custom_session.dataframe.iloc[1]["a"])

    # Load sample fixture
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Check 3: Default inspection (head 5 rows) against dynamic session.dataframe shape
    res_default = inspect_data_tool.invoke({})
    assert res_default["shape"] == list(session.dataframe.shape)
    assert "text" in res_default["columns"]
    assert "category" in res_default["columns"]
    assert res_default["n_rows_inspected"] == 5
    assert len(res_default["head_sample"]) == 5

    # Verify json.dumps(allow_nan=False) succeeds on sample fixture head inspection
    fixture_json = json.dumps(res_default["head_sample"], allow_nan=False)
    assert fixture_json is not None

    # Check 4: Custom n_rows (2 rows)
    res_2 = inspect_data_tool.invoke({"n_rows": 2})
    assert res_2["n_rows_inspected"] == 2
    assert len(res_2["head_sample"]) == 2

    # Check 5: Invalid n_rows rejection
    invalid_rows = [0, -1, 2.5, "5", True, False]
    for val in invalid_rows:
        caught_exception = False
        res = None
        try:
            res = inspect_data_tool.invoke({"n_rows": val})
        except (ValidationError, TypeError, ValueError, ToolException):
            caught_exception = True

        assert caught_exception or (isinstance(res, dict) and "error" in res), f"Failed to reject invalid n_rows={val}"

    # Check 6: Session store recording
    result_id = res_default["result_id"]
    assert session.results_store[result_id]["record"].summary == res_default


def test_check_missing_tool():
    session = SessionState()
    student_tools = make_tools(session)
    check_missing_tool = next(t for t in student_tools if t.name == "check_missing_tool")

    # Requirement 1: Missing dataframe error handling
    err_res = check_missing_tool.invoke({})
    assert "error" in err_res

    # Requirement 2: Missing text column error handling
    no_text_session = SessionState()
    no_text_session.dataframe = pd.DataFrame({"other": [1, 2, 3]})
    no_text_tools = make_tools(no_text_session)
    no_text_check = next(t for t in no_text_tools if t.name == "check_missing_tool")
    assert "error" in no_text_check.invoke({})

    # Requirement 3 & 7 & 8: Known Answer #2 on sample_fixture.csv (missing_count=1 at row 7)
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    fixture_res = check_missing_tool.invoke({})
    assert fixture_res["total_rows"] == 8
    assert fixture_res["missing_count"] == 1
    assert fixture_res["missing_indices"] == [7]
    assert fixture_res["missing_pct"] == pytest.approx(100.0 * 1 / 8, rel=1e-3)

    # Requirement 4: Custom dataframe with None, np.nan, pd.NA, empty string "", and whitespace "   "
    custom_session = SessionState()
    custom_session.dataframe = pd.DataFrame({
        "text": ["valid text", None, np.nan, pd.NA, "", "   ", "another valid"]
    }, index=["a", "b", "c", "d", "e", "f", "g"])  # Non-integer index labels to test enumerate()
    custom_tools = make_tools(custom_session)
    custom_check = next(t for t in custom_tools if t.name == "check_missing_tool")
    custom_res = custom_check.invoke({})
    assert custom_res["total_rows"] == 7
    assert custom_res["missing_count"] == 5
    assert custom_res["missing_indices"] == [1, 2, 3, 4, 5]  # Positional 0-based row indices
    assert custom_res["missing_pct"] == pytest.approx(100.0 * 5 / 7, rel=1e-3)

    # Requirement 5: Dataframe with NO missing text (missing_count=0)
    clean_session = SessionState()
    clean_session.dataframe = pd.DataFrame({"text": ["hello", "world", "python"]})
    clean_tools = make_tools(clean_session)
    clean_check = next(t for t in clean_tools if t.name == "check_missing_tool")
    clean_res = clean_check.invoke({})
    assert clean_res["total_rows"] == 3
    assert clean_res["missing_count"] == 0
    assert clean_res["missing_indices"] == []
    assert clean_res["missing_pct"] == 0.0

    # Requirement 6: Empty dataframe (0 rows)
    empty_df_session = SessionState()
    empty_df_session.dataframe = pd.DataFrame({"text": []})
    empty_df_tools = make_tools(empty_df_session)
    empty_df_check = next(t for t in empty_df_tools if t.name == "check_missing_tool")
    empty_df_res = empty_df_check.invoke({})
    assert empty_df_res["total_rows"] == 0
    assert empty_df_res["missing_count"] == 0
    assert empty_df_res["missing_indices"] == []
    assert empty_df_res["missing_pct"] == 0.0

    # Requirement 9: Session result store recording
    result_id = fixture_res["result_id"]
    assert session.results_store[result_id]["record"].summary == fixture_res


def test_check_duplicates_tool():
    session = SessionState()
    student_tools = make_tools(session)
    check_duplicates_tool = next(t for t in student_tools if t.name == "check_duplicates_tool")

    # Check 1: Missing dataframe error handling
    err_res = check_duplicates_tool.invoke({})
    assert "error" in err_res

    # Check 2: Load sample fixture
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Check 3: Known Answer #3 - duplicate check without dropping
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check["initial_rows"] == 8
    assert res_check["duplicate_count"] == 1
    assert res_check["dropped"] is False
    assert res_check["remaining_rows"] == 8

    # Check 4: Known Answer #3 - dropping duplicates (keep=False leaves 6 rows)
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["initial_rows"] == 8
    assert res_drop["duplicate_count"] == 1
    assert res_drop["dropped"] is True
    assert res_drop["remaining_rows"] == 6
    assert len(session.dataframe) == 6

    # Check 5: Dataframe with NO duplicates
    clean_session = SessionState()
    clean_session.dataframe = pd.DataFrame({"text": ["doc A", "doc B", "doc C"]})
    clean_tools = make_tools(clean_session)
    clean_dup = next(t for t in clean_tools if t.name == "check_duplicates_tool")
    clean_res = clean_dup.invoke({"drop": True})
    assert clean_res["duplicate_count"] == 0
    assert clean_res["remaining_rows"] == 3

    # Check 6: Fallback duplicate detection without 'text' column & unhashable columns (unigrams)
    # 6a: Same ID but DIFFERENT token lists must NOT count as duplicates
    diff_tokens_session = SessionState()
    diff_tokens_session.dataframe = pd.DataFrame({
        "id": [1, 1],
        "unigrams": [["a", "b"], ["a", "c"]]
    })
    diff_tools = make_tools(diff_tokens_session)
    diff_dup = next(t for t in diff_tools if t.name == "check_duplicates_tool")
    diff_res = diff_dup.invoke({"drop": False})
    assert diff_res["duplicate_count"] == 0

    # 6b: Identical ID and IDENTICAL token lists MUST count as duplicates
    same_tokens_session = SessionState()
    same_tokens_session.dataframe = pd.DataFrame({
        "id": [1, 2, 1],
        "unigrams": [["a", "b"], ["c"], ["a", "b"]]
    })
    same_tools = make_tools(same_tokens_session)
    same_dup = next(t for t in same_tools if t.name == "check_duplicates_tool")
    same_res = same_dup.invoke({"drop": False})
    assert same_res["duplicate_count"] == 1

    # 6c: Non-default indices with list-valued columns
    non_default_idx_session = SessionState()
    non_default_idx_session.dataframe = pd.DataFrame({
        "id": [10, 20, 10],
        "tags": [["x", "y"], ["z"], ["x", "y"]]
    }, index=["idx_a", "idx_b", "idx_c"])
    non_default_tools = make_tools(non_default_idx_session)
    non_default_dup = next(t for t in non_default_tools if t.name == "check_duplicates_tool")
    non_default_res = non_default_dup.invoke({"drop": True})
    assert non_default_res["duplicate_count"] == 1
    assert non_default_res["remaining_rows"] == 1
    assert non_default_idx_session.dataframe.iloc[0]["id"] == 20

    # 6d: Set-valued cells including mixed-type sets {"apple", 10}
    set_session = SessionState()
    set_session.dataframe = pd.DataFrame({
        "id": [1, 2, 1],
        "set_col": [{"apple", 10}, {"cherry"}, {10, "apple"}]
    })
    set_tools = make_tools(set_session)
    set_dup = next(t for t in set_tools if t.name == "check_duplicates_tool")
    set_res = set_dup.invoke({"drop": False})
    assert set_res["duplicate_count"] == 1

    # Check 7: Invalid drop parameter types (StrictBool rejection of strings/numbers)
    invalid_drop_vals = ["False", "True", "0", "1", 0, 1, 2.5]
    for val in invalid_drop_vals:
        caught_exception = False
        res = None
        try:
            res = check_duplicates_tool.invoke({"drop": val})
        except (ValidationError, TypeError, ValueError, ToolException):
            caught_exception = True

        assert caught_exception or (isinstance(res, dict) and "error" in res), f"Failed to reject invalid drop={val}"

    # Check 8: Session store recording
    result_id = res_drop["result_id"]
    assert session.results_store[result_id]["record"].summary == res_drop


def test_sample_data_tool():
    session = SessionState()
    student_tools = make_tools(session)
    sample_data_tool = next(t for t in student_tools if t.name == "sample_data_tool")

    # Check 1: Missing dataframe error handling
    err_res = sample_data_tool.invoke({})
    assert "error" in err_res

    # Check 2: Load sample fixture
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Check 3: Known Answer #13 - n=4, random_state=42 returns positional rows [1, 5, 0, 7]
    res_fixture = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res_fixture["n_requested"] == 4
    assert res_fixture["n_sampled"] == 4
    assert res_fixture["random_state"] == 42
    assert res_fixture["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4

    # Check 4: Correct session binding when requesting n greater than total rows
    cap_session = SessionState()
    cap_premade, _ = load_workspace_tools("premade_tools", cap_session)
    cap_load_dataset = next(t for t in cap_premade if t.name == "load_dataset_tool")
    cap_load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    cap_tools = make_tools(cap_session)
    cap_sample = next(t for t in cap_tools if t.name == "sample_data_tool")
    res_cap = cap_sample.invoke({"n": 20, "random_state": 42})
    assert res_cap["n_requested"] == 20
    assert res_cap["n_sampled"] == 8
    assert len(cap_session.dataframe) == 8

    # Check 5: Duplicate index labels dataframe sampling
    dup_idx_session = SessionState()
    dup_idx_session.dataframe = pd.DataFrame({
        "text": ["A", "B", "C"]
    }, index=["row_a", "row_a", "row_b"])
    dup_tools = make_tools(dup_idx_session)
    dup_sample = next(t for t in dup_tools if t.name == "sample_data_tool")
    res_dup = dup_sample.invoke({"n": 2, "random_state": 42})
    assert res_dup["n_sampled"] == 2
    assert isinstance(res_dup["sampled_indices"][0], int)
    assert len(dup_idx_session.dataframe) == 2

    # Check 6: Empty dataframe sampling (total_rows = 0)
    empty_session = SessionState()
    empty_session.dataframe = pd.DataFrame({"text": []})
    empty_tools = make_tools(empty_session)
    empty_sample = next(t for t in empty_tools if t.name == "sample_data_tool")
    res_empty = empty_sample.invoke({"n": 5})
    assert res_empty["n_sampled"] == 0
    assert res_empty["sampled_indices"] == []
    assert res_empty["sampled_records"] == []

    # Check 7: Invalid n parameter types (StrictInt rejection of 0, negative numbers, floats, strings, booleans)
    invalid_n_vals = [0, -1, 2.5, "4", True, False]
    for val in invalid_n_vals:
        caught_exception = False
        res = None
        try:
            res = sample_data_tool.invoke({"n": val})
        except (ValidationError, TypeError, ValueError, ToolException):
            caught_exception = True

        assert caught_exception or (isinstance(res, dict) and "error" in res), f"Failed to reject invalid n={val}"

    # Check 8: Session store recording
    result_id = res_fixture["result_id"]
    assert session.results_store[result_id]["record"].summary == res_fixture


def test_describe_data_tool():
    try:
        session = SessionState()
        student_tools = make_tools(session)
        describe_data_tool = next(t for t in student_tools if t.name == "describe_data_tool")

        # Check 1: Missing dataframe error handling
        err_res = describe_data_tool.invoke({})
        assert "error" in err_res

        # Check 2: Load sample fixture
        premade_tools, _ = load_workspace_tools("premade_tools", session)
        load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
        load_dataset_tool.invoke({
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label"
        })

        original_cols = list(session.dataframe.columns)

        # Check 3: Known Answer #18 verification (all overall and per-category stats)
        result = describe_data_tool.invoke({})

        # Verify session.dataframe was not mutated (no new columns added)
        assert list(session.dataframe.columns) == original_cols

        overall = result["overall_stats"]
        assert overall["count"] == 8
        assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
        assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
        assert overall["min"] == 0
        assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
        assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
        assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
        assert overall["max"] == 30

        cat_stats = result["per_category_stats"]
        assert "catA" in cat_stats and "catB" in cat_stats

        # Full catA verification (Known Answer #18)
        cat_a = cat_stats["catA"]
        assert cat_a["count"] == 4
        assert cat_a["mean"] == pytest.approx(17.25, abs=1e-3)
        assert cat_a["std"] == pytest.approx(12.5, abs=1e-3)
        assert cat_a["min"] == 0
        assert cat_a["25%"] == pytest.approx(12.75, abs=1e-3)
        assert cat_a["50%"] == pytest.approx(20.0, abs=1e-3)
        assert cat_a["75%"] == pytest.approx(24.5, abs=1e-3)
        assert cat_a["max"] == 29

        # Full catB verification (Known Answer #18)
        cat_b = cat_stats["catB"]
        assert cat_b["count"] == 4
        assert cat_b["mean"] == pytest.approx(22.5, abs=1e-3)
        assert cat_b["std"] == pytest.approx(5.7446, abs=1e-3)
        assert cat_b["min"] == 18
        assert cat_b["25%"] == pytest.approx(18.0, abs=1e-3)
        assert cat_b["50%"] == pytest.approx(21.0, abs=1e-3)
        assert cat_b["75%"] == pytest.approx(25.5, abs=1e-3)
        assert cat_b["max"] == 30

        # Check 4: Figure attached to session.pending_figure
        assert session.pending_figure is not None
        assert isinstance(session.pending_figure, matplotlib.figure.Figure)

        # Check 5: Dataframe without category columns returns overall_stats and empty per_category_stats
        no_cat_session = SessionState()
        no_cat_session.dataframe = pd.DataFrame({
            "text": ["Short", "Longer text entry", None]
        })
        no_cat_tools = make_tools(no_cat_session)
        no_cat_describe = next(t for t in no_cat_tools if t.name == "describe_data_tool")
        no_cat_res = no_cat_describe.invoke({})
        assert "overall_stats" in no_cat_res
        assert no_cat_res["per_category_stats"] == {}
        assert no_cat_res["overall_stats"]["count"] == 3

        # Check 6: Duplicate index labels dataframe handling
        dup_idx_session = SessionState()
        dup_idx_session.dataframe = pd.DataFrame({
            "text": ["Alpha text", "Beta text", "Gamma text"],
            "category_name": ["catA", "catA", "catB"]
        }, index=["idx_1", "idx_1", "idx_2"])
        dup_idx_tools = make_tools(dup_idx_session)
        dup_idx_describe = next(t for t in dup_idx_tools if t.name == "describe_data_tool")
        dup_idx_res = dup_idx_describe.invoke({})
        assert dup_idx_res["per_category_stats"]["catA"]["count"] == 2
        assert dup_idx_res["per_category_stats"]["catB"]["count"] == 1

        # Check 7: Session store recording
        result_id = result["result_id"]
        assert session.results_store[result_id]["record"].summary == result

    finally:
        plt.close("all")

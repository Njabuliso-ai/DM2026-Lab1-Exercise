import pytest
import pandas as pd
from pydantic import ValidationError
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


@pytest.fixture
def session_and_tools():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    return session, mine_tool


def test_mine_patterns_invalid_params(session_and_tools):
    session, mine_tool = session_and_tools

    # Missing min_sup for fpgrowth
    res1 = mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth"})
    assert "error" in res1

    # Non-positive min_sup
    res1_zero = mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 0})
    assert "error" in res1_zero

    res1_neg = mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": -2})
    assert "error" in res1_neg

    # Boolean min_sup rejected via ValidationError
    with pytest.raises(ValidationError):
        mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": True})

    # Missing / bad k for topk
    res2_missing = mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk"})
    assert "error" in res2_missing

    res2_zero = mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": 0})
    assert "error" in res2_zero

    # Boolean k rejected via ValidationError
    with pytest.raises(ValidationError):
        mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": True})

    # Non-existent category
    res3 = mine_tool.invoke({"category_name": "non_existent", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1})
    assert "error" in res3


def test_mine_patterns_zero_surviving_terms(session_and_tools):
    session, mine_tool = session_and_tools

    # Known answer 17: variance filtering removes all 2 terms in catA
    res_var = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["n_patterns"] == 0
    assert res_var["patterns"] == []
    assert res_var["note"] == "no terms survived filtering"

    # Known answer 17: term_frequency filtering removes all 2 terms in catB
    res_tf = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tf["n_patterns"] == 0
    assert res_tf["patterns"] == []
    assert res_tf["note"] == "no terms survived filtering"

    # Artifact & full_report storage check for empty results
    art_key = "patterns_catA_variance_fpgrowth"
    assert art_key in session.artifacts
    assert session.artifacts[art_key] == []

    stored_res = session.results_store[res_var["result_id"]]
    assert isinstance(stored_res["full_report"], pd.DataFrame)
    assert stored_res["full_report"].empty


def test_mine_patterns_known_answer_17_fpgrowth(session_and_tools):
    session, mine_tool = session_and_tools

    # Known answer 17: catA + tfidf + fpgrowth (min_sup=1..3 -> alpha support 3)
    res_catA = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA["n_patterns"] == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    # Known answer 17: catA at min_sup=4 -> 0 patterns
    res_catA_sup4 = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catA_sup4["n_patterns"] == 0

    # Known answer 17: catB + tfidf + fpgrowth (min_sup=1..4 -> gamma support 4)
    res_catB = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catB["n_patterns"] == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_known_answer_20_topk_and_maxfpgrowth(session_and_tools):
    session, mine_tool = session_and_tools

    # Known answer 20: topk (k=1)
    res_topk_catA = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_catA["n_patterns"] == 1
    assert res_topk_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk_catA["patterns"][0]["support"] == 3

    # Known answer 20: maxfpgrowth returns 0 patterns for single-item data
    res_max_catA = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catA["n_patterns"] == 0
    assert res_max_catA["patterns"] == []

    # Check stored artifact
    art_key = "patterns_catA_tfidf_topk"
    assert art_key in session.artifacts
    assert session.artifacts[art_key] == [{"pattern": ["alpha"], "support": 3}]

    # Check full_report in stored result
    stored_res = session.results_store[res_topk_catA["result_id"]]
    assert isinstance(stored_res["full_report"], pd.DataFrame)
    assert not stored_res["full_report"].empty


def test_mine_patterns_empty_vocabulary():
    session = SessionState()
    # Create DataFrame with only stopwords for category 'catC'
    session.dataframe = pd.DataFrame({
        "text": ["the a an", "in on at"],
        "category_name": ["catC", "catC"]
    })
    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res = mine_tool.invoke({
        "category_name": "catC",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res["n_patterns"] == 0
    assert "note" in res

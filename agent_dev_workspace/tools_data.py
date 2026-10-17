import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from pydantic import StrictInt, StrictBool
from langchain_core.tools import tool


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert values into JSON-serializable Python types."""

    if obj is None or obj is pd.NA:
        return None

    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)

    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)

    if isinstance(obj, (int, np.integer)):
        return int(obj)

    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        return {
            str(key): _make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple, np.ndarray, pd.Series, set, frozenset)):
        return [_make_json_safe(item) for item in obj]

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return str(obj)


def _normalize_cell_for_hash(val: Any) -> Any:
    """Recursively converts unhashable objects into consistently ordered, hashable tuples."""
    if isinstance(val, (set, frozenset)):
        return tuple(sorted(
            (_normalize_cell_for_hash(item) for item in val),
            key=lambda item: (type(item).__name__, repr(item))
        ))
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        return tuple(_normalize_cell_for_hash(item) for item in val)
    if isinstance(val, dict):
        return tuple(sorted((str(k), _normalize_cell_for_hash(v)) for k, v in val.items()))
    if pd.isna(val) or val is None or val is pd.NA:
        return None
    return val



def _safe_text_len(val: Any) -> int:
    """Safely calculate text length, including missing or unusual values."""
    if val is None or val is pd.NA:
        return 0
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return 0
    if isinstance(val, (list, tuple, dict, set, np.ndarray)):
        return 0
    return len(str(val))


def make_tools(session: Any) -> List[Any]:
    """Create data management tools bound to the current session."""

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenize document text into lowercase unigrams using NLTK.

        Stores token lists in the dataframe and session artifacts.
        Returns tokenization summary statistics.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {
                "error": "No dataframe loaded or 'text' column missing in session. Load a dataset first."
            }

        tokens_list = []

        for text in session.dataframe["text"]:
            if text is None or pd.isna(text) or str(text).strip() == "":
                tokens_list.append([])
            else:
                try:
                    tokens = nltk.word_tokenize(str(text).lower())
                    tokens_list.append(tokens)
                except LookupError as e:
                    return {
                        "error": (
                            f"NLTK tokenizer resources missing ({str(e)}). "
                            "Please download NLTK 'punkt' or 'punkt_tab'."
                        )
                    }

        session.dataframe["unigrams"] = tokens_list
        session.artifacts["tokens"] = tokens_list

        total_tokens = sum(len(tokens) for tokens in tokens_list)
        n_docs = len(tokens_list)

        result_id = session.next_result_id("tokenize")

        summary = {
            "result_id": result_id,
            "n_documents": n_docs,
            "total_tokens": total_tokens,
            "sample_tokens": tokens_list[:3]
        }

        session.store_result("tokenize_tool", {}, summary)

        return summary

    @tool
    def list_files_tool(subdirectory: Optional[str] = None) -> Dict[str, Any]:
        """List files and directories within the repository.

        Args:
            subdirectory: Optional relative directory path.

        Returns:
            Dictionary containing files, directories, and summary counts.
        """
        repo_root = Path(".").resolve()

        if subdirectory is not None and str(subdirectory).strip() != "":
            raw_sub = str(subdirectory).strip()

            if Path(raw_sub).is_absolute():
                return {
                    "error": "Access denied: Absolute paths are not permitted."
                }

            target_path = (repo_root / raw_sub).resolve()
        else:
            raw_sub = ""
            target_path = repo_root

        # Prevent access outside the repository
        try:
            target_path.relative_to(repo_root)
        except ValueError:
            return {
                "error": "Access denied: Path is outside the repository root."
            }

        if not target_path.exists() or not target_path.is_dir():
            return {
                "error": f"Directory not found: '{subdirectory}'"
            }

        try:
            entries = list(target_path.iterdir())
        except Exception as e:
            return {
                "error": f"Failed to list directory '{subdirectory}': {str(e)}"
            }

        dirs = sorted([e.name for e in entries if e.is_dir()])
        files = sorted([e.name for e in entries if e.is_file()])

        result_id = session.next_result_id("list_files")

        summary = {
            "result_id": result_id,
            "subdirectory": raw_sub,
            "directories": dirs,
            "files": files,
            "total_directories": len(dirs),
            "total_files": len(files)
        }

        session.store_result(
            "list_files_tool",
            {"subdirectory": subdirectory},
            summary
        )

        return summary

    @tool
    def inspect_data_tool(n_rows: StrictInt = 5) -> Dict[str, Any]:
        """Inspect the first n_rows of the loaded dataframe.

        Args:
            n_rows: Positive integer number of rows to inspect.

        Returns:
            Dataset shape, columns, and JSON-safe preview records.
        """

        if session.dataframe is None:
            return {
                "error": "No dataframe loaded in session. Load a dataset first."
            }

        if n_rows <= 0:
            return {
                "error": "n_rows must be a positive integer."
            }

        df = session.dataframe

        # Read the first n_rows without changing the dataframe
        raw_records = df.head(n_rows).to_dict(orient="records")

        # Convert missing values and NumPy objects safely
        head_sample = _make_json_safe(raw_records)

        result_id = session.next_result_id("inspect")

        summary = {
            "result_id": result_id,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "n_rows_inspected": len(head_sample),
            "head_sample": head_sample
        }

        session.store_result(
            "inspect_data_tool",
            {"n_rows": n_rows},
            summary
        )

        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Check for missing or empty text values in the loaded dataframe.

        Detects None, NaN, pd.NA, empty strings, and whitespace-only
        strings in session.dataframe["text"].

        Returns:
            Dictionary containing result_id, total_rows,
            missing_count, missing_pct, and missing_indices.
        """

        # Check whether the dataset and text column exist
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {
                "error": (
                    "No dataframe loaded or 'text' column missing "
                    "in session. Load a dataset first."
                )
            }

        text_series = session.dataframe["text"]
        total_rows = len(text_series)

        # Find missing values using zero-based row positions
        missing_indices = []

        for idx, val in enumerate(text_series):
            if val is None or pd.isna(val) or str(val).strip() == "":
                missing_indices.append(idx)

        # Calculate the number and percentage of missing rows
        missing_count = len(missing_indices)

        if total_rows > 0:
            missing_pct = 100.0 * missing_count / total_rows
        else:
            missing_pct = 0.0

        # Generate a result identifier
        result_id = session.next_result_id("check_missing")

        # Prepare the summary
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4),
            "missing_indices": missing_indices
        }

        # Store the result in the session
        session.store_result(
            "check_missing_tool",
            {},
            summary
        )

        return summary

    @tool
    def check_duplicates_tool(drop: StrictBool = False) -> Dict[str, Any]:
        """Checks for duplicate documents in session.dataframe and optionally drops all duplicate copies.

        Args:
            drop: If True, drops every copy of duplicate documents using keep=False (default False).

        Returns:
            Dict containing result_id, initial_rows, duplicate_count, dropped, and remaining_rows.
        """
        if session.dataframe is None:
            return {
                "error": "No dataframe loaded in session. Load a dataset first."
            }

        df = session.dataframe
        initial_rows = len(df)

        if "text" in df.columns:
            eval_series = df["text"]
            dup_first_mask = eval_series.duplicated(keep="first")
            dup_all_mask = eval_series.duplicated(keep=False)
        else:
            norm_records = []
            for record in df.to_dict(orient="records"):
                norm_record = tuple((col, _normalize_cell_for_hash(record[col])) for col in df.columns)
                norm_records.append(norm_record)

            norm_series = pd.Series(norm_records, index=df.index)
            dup_first_mask = norm_series.duplicated(keep="first")
            dup_all_mask = norm_series.duplicated(keep=False)

        duplicate_count = int(dup_first_mask.sum())

        if drop and duplicate_count > 0:
            session.dataframe = session.dataframe[~dup_all_mask].reset_index(drop=True)
            remaining_rows = len(session.dataframe)
            dropped = True
        else:
            remaining_rows = len(session.dataframe)
            dropped = False

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "initial_rows": initial_rows,
            "duplicate_count": duplicate_count,
            "dropped": dropped,
            "remaining_rows": remaining_rows
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: StrictInt = 5,
        random_state: Optional[StrictInt] = None
    ) -> Dict[str, Any]:
        """Subsamples the working dataset in session.dataframe.

        Updates session.dataframe in place with n randomly sampled rows.

        Args:
            n: Positive integer number of rows to sample (default 5).
            random_state: Optional random seed integer for reproducible sampling.

        Returns:
            Dict containing result_id, n_requested, n_sampled, random_state, sampled_indices, and sampled_records.
        """
        if session.dataframe is None:
            return {
                "error": "No dataframe loaded in session. Load a dataset first."
            }

        if n <= 0:
            return {
                "error": "n must be a positive integer."
            }

        df = session.dataframe
        total_rows = len(df)

        if total_rows == 0:
            result_id = session.next_result_id("sample")
            summary = {
                "result_id": result_id,
                "n_requested": n,
                "n_sampled": 0,
                "random_state": random_state,
                "sampled_indices": [],
                "sampled_records": []
            }
            session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
            return summary

        sample_size = min(n, total_rows)

        # Sample zero-based integer row positions directly to safely handle duplicate index labels
        position_df = pd.DataFrame({"position": range(total_rows)})
        sampled_indices = position_df.sample(
            n=sample_size,
            random_state=random_state
        )["position"].tolist()

        sampled_df = df.iloc[sampled_indices]

        # Update session.dataframe in place with the subsampled dataset
        session.dataframe = sampled_df.reset_index(drop=True)

        sampled_records = _make_json_safe(sampled_df.to_dict(orient="records"))

        result_id = session.next_result_id("sample")
        summary = {
            "result_id": result_id,
            "n_requested": n,
            "n_sampled": sample_size,
            "random_state": random_state,
            "sampled_indices": [int(idx) for idx in sampled_indices],
            "sampled_records": sampled_records
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary


    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Calculate text-length statistics and generate a category box plot."""
        if (
            session.dataframe is None
            or len(session.dataframe) == 0
            or "text" not in session.dataframe.columns
        ):
            return {
                "error": "No dataframe or 'text' column found in session. Load a dataset first."
            }

        df = session.dataframe
        text_lengths = df["text"].apply(_safe_text_len)

        overall_stats = _make_json_safe(
            text_lengths.describe().to_dict()
        )

        category_col = (
            "category_name" if "category_name" in df.columns
            else "category" if "category" in df.columns
            else None
        )

        per_category_stats = {}

        if category_col:
            plot_df = pd.DataFrame({
                "category": df[category_col].to_numpy(),
                "text_length": text_lengths.to_numpy()
            })

            for cat, group in plot_df.groupby("category"):
                per_category_stats[str(cat)] = _make_json_safe(
                    group["text_length"].describe().to_dict()
                )

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                x="category",
                y="text_length",
                data=plot_df,
                ax=ax
            )

            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
            ax.set_ylabel("Text Length (Characters)")
            plt.tight_layout()

            session.pending_figure = fig

        result_id = session.next_result_id("describe")

        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats
        }

        session.store_result("describe_data_tool", {}, summary)

        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]

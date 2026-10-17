
import math
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from pydantic import StrictFloat
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Create feature filtering tools bound to the current session."""

    @tool
    def variance_filter_tool(threshold: StrictFloat = 0.0) -> Dict[str, Any]:
        """Filter DTM terms by variance and produce a sorted variance report.

        Args:
            threshold: Finite, non-negative minimum variance for keeping terms.

        Returns:
            Summary containing term variances and filtering results.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix found in session. Run build_dtm_tool first."
            }

        # Validate threshold
        if (
            not isinstance(threshold, (int, float, np.number))
            or isinstance(threshold, bool)
        ):
            return {"error": "threshold must be a non-negative float."}

        if not math.isfinite(threshold) or threshold < 0.0:
            return {"error": "threshold must be a finite non-negative float."}

        dtm = session.feature_matrix

        # Validate DTM shape
        if not hasattr(dtm, "shape") or len(dtm.shape) != 2:
            return {
                "error": "Document-Term Matrix must be a two-dimensional matrix."
            }

        n_rows, n_cols = dtm.shape

        if n_rows < 1:
            return {
                "error": "Document-Term Matrix must contain at least one document."
            }

        if n_cols != len(session.feature_names):
            return {
                "error": (
                    f"Dimension mismatch: feature_matrix has {n_cols} columns "
                    f"but feature_names has {len(session.feature_names)} terms."
                )
            }

        # Population variance: E[X²] - E[X]²
        # Avoid converting sparse matrices to dense arrays.
        if sp.issparse(dtm):
            mean_x = np.asarray(dtm.mean(axis=0)).flatten()
            mean_x2 = np.asarray(
                dtm.multiply(dtm).mean(axis=0)
            ).flatten()

            variances = mean_x2 - (mean_x ** 2)
        else:
            variances = np.var(
                np.asarray(dtm),
                axis=0,
                ddof=0
            )

        # Prevent negative values caused by floating-point rounding.
        variances = np.maximum(variances, 0.0)

        feature_names = session.feature_names

        report_data = []
        kept_terms = []
        removed_terms = []
        term_variance_map = {}

        for term, var in zip(feature_names, variances):
            term_str = str(term)
            var_float = float(var)

            term_variance_map[term_str] = round(var_float, 6)

            report_data.append({
                "term": term_str,
                "variance": var_float
            })

            if var_float >= threshold:
                kept_terms.append(term_str)
            else:
                removed_terms.append(term_str)

        # Build the report required by visualize_result_tool.
        if report_data:
            full_report_df = (
                pd.DataFrame(report_data)
                .sort_values(by="variance", ascending=False)
                .reset_index(drop=True)
            )
        else:
            full_report_df = pd.DataFrame(
                columns=["term", "variance"]
            )

        result_id = session.next_result_id("variance")

        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "initial_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "variances": term_variance_map
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=full_report_df
        )

        return summary


    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: StrictFloat = 0.0
    ) -> Dict[str, Any]:
        """Calculate Pearson correlations between DTM terms and a target class.

        Uses one-vs-rest binary labels and keeps terms whose absolute
        correlation is greater than or equal to the threshold.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix found in session. Run build_dtm_tool first."
            }

        if session.labels is None:
            return {
                "error": "No category labels found in session. Ensure dataset is loaded with labels."
            }

        if not isinstance(target_class, str) or not target_class.strip():
            return {"error": "target_class must be a non-empty string."}

        if not math.isfinite(threshold) or threshold < 0.0:
            return {"error": "threshold must be a finite non-negative float."}

        dtm = session.feature_matrix

        if not hasattr(dtm, "shape") or len(dtm.shape) != 2:
            return {
                "error": "Document-Term Matrix must be a two-dimensional matrix."
            }

        n_rows, n_cols = dtm.shape

        if n_rows < 1:
            return {
                "error": "Document-Term Matrix must contain at least one document."
            }

        if n_cols != len(session.feature_names):
            return {
                "error": (
                    f"Dimension mismatch: feature_matrix has {n_cols} columns "
                    f"but feature_names has {len(session.feature_names)} terms."
                )
            }

        labels = np.asarray(session.labels)

        if len(labels) != n_rows:
            return {
                "error": (
                    f"Label count mismatch: session.labels has {len(labels)} items "
                    f"but feature_matrix has {n_rows} rows."
                )
            }

        if target_class not in labels:
            return {
                "error": f"Target class '{target_class}' not found in session labels."
            }

        # Create one-vs-rest binary target
        y = (labels == target_class).astype(float)

        mu_y = float(y.mean())
        var_y = float(y.var(ddof=0))
        std_y = math.sqrt(var_y) if var_y > 0 else 0.0

        # Calculate means, variances, and covariance
        # without converting sparse matrices to dense arrays.
        if sp.issparse(dtm):
            mu_x = np.asarray(dtm.mean(axis=0)).flatten()
            mu_x2 = np.asarray(
                dtm.multiply(dtm).mean(axis=0)
            ).flatten()

            var_x = np.maximum(mu_x2 - mu_x ** 2, 0.0)
            std_x = np.sqrt(var_x)

            xy_dot = np.asarray(dtm.T.dot(y)).flatten()
            mean_xy = xy_dot / n_rows
            cov_xy = mean_xy - mu_x * mu_y

        else:
            dtm_arr = np.asarray(dtm, dtype=float)

            mu_x = dtm_arr.mean(axis=0)
            var_x = np.maximum(
                dtm_arr.var(axis=0, ddof=0),
                0.0
            )
            std_x = np.sqrt(var_x)

            mean_xy = np.mean(dtm_arr * y[:, None], axis=0)
            cov_xy = mean_xy - mu_x * mu_y

        report_data = []
        kept_terms = []
        removed_terms = []
        correlations_map = {}

        for term, cov, sx in zip(
            session.feature_names, cov_xy, std_x
        ):
            term_str = str(term)

            # Correlation is undefined for constant variables.
            # Return 0.0 instead of dividing by zero.
            if sx == 0.0 or std_y == 0.0:
                r = 0.0
            else:
                r = float(cov / (sx * std_y))
                r = max(-1.0, min(1.0, r))

            correlations_map[term_str] = round(r, 6)

            report_data.append({
                "term": term_str,
                "pearson_r": r
            })

            if abs(r) >= threshold:
                kept_terms.append(term_str)
            else:
                removed_terms.append(term_str)

        # Build the report required by visualize_result_tool
        if report_data:
            full_report_df = (
                pd.DataFrame(report_data)
                .sort_values(by="pearson_r", ascending=False)
                .reset_index(drop=True)
            )
        else:
            full_report_df = pd.DataFrame(
                columns=["term", "pearson_r"]
            )

        result_id = session.next_result_id("pearson")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "initial_terms": len(session.feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "correlations": correlations_map
        }

        session.store_result(
            "pearson_filter_tool",
            {
                "target_class": target_class,
                "threshold": threshold
            },
            summary,
            full_report=full_report_df
        )

        return summary


    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: StrictFloat = 0.0
    ) -> Dict[str, Any]:
        """Calculate Spearman rank correlations between DTM terms and a target class."""

        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No Document-Term Matrix found. Run build_dtm_tool first."}

        if session.labels is None:
            return {"error": "No category labels found in session."}

        if not isinstance(target_class, str) or not target_class.strip():
            return {"error": "target_class must be a non-empty string."}

        if not math.isfinite(threshold) or threshold < 0.0:
            return {"error": "threshold must be a finite non-negative float."}

        dtm = session.feature_matrix

        if not hasattr(dtm, "shape") or len(dtm.shape) != 2:
            return {"error": "Document-Term Matrix must be two-dimensional."}

        n_rows, n_cols = dtm.shape

        if n_rows < 1:
            return {"error": "Document-Term Matrix must contain at least one document."}

        if n_cols != len(session.feature_names):
            return {"error": "Feature matrix and feature names have different dimensions."}

        labels = np.asarray(session.labels)

        if len(labels) != n_rows:
            return {"error": "Label count does not match document count."}

        if target_class not in labels:
            return {"error": f"Target class '{target_class}' not found in session labels."}

        # One-vs-rest binary target
        y = (labels == target_class).astype(float)

        # Rank the binary target, averaging tied ranks
        rank_y = stats.rankdata(y, method="average")
        std_ry = float(np.std(rank_y))

        report_data = []
        kept_terms = []
        removed_terms = []
        correlations_map = {}

        # Process one feature column at a time
        for j, term in enumerate(session.feature_names):

            if sp.issparse(dtm):
                col_x = np.asarray(
                    dtm[:, j].toarray()
                ).flatten().astype(float)
            else:
                col_x = np.asarray(
                    dtm[:, j]
                ).flatten().astype(float)

            # Spearman correlation = Pearson correlation of ranks
            rank_x = stats.rankdata(col_x, method="average")
            std_rx = float(np.std(rank_x))

            # Constant variables have undefined correlation.
            # Use zero to avoid NaN results.
            if std_rx == 0.0 or std_ry == 0.0:
                rho = 0.0
            else:
                rho = float(np.corrcoef(rank_x, rank_y)[0, 1])

                if not math.isfinite(rho):
                    rho = 0.0
                else:
                    rho = max(-1.0, min(1.0, rho))

            term_str = str(term)
            correlations_map[term_str] = round(rho, 6)

            report_data.append({
                "term": term_str,
                "spearman_r": rho
            })

            if abs(rho) >= threshold:
                kept_terms.append(term_str)
            else:
                removed_terms.append(term_str)

        # Full report for visualize_result_tool
        if report_data:
            full_report_df = (
                pd.DataFrame(report_data)
                .sort_values(by="spearman_r", ascending=False)
                .reset_index(drop=True)
            )
        else:
            full_report_df = pd.DataFrame(
                columns=["term", "spearman_r"]
            )

        result_id = session.next_result_id("spearman")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "initial_terms": len(session.feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "correlations": correlations_map
        }

        session.store_result(
            "spearman_filter_tool",
            {
                "target_class": target_class,
                "threshold": threshold
            },
            summary,
            full_report=full_report_df
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]



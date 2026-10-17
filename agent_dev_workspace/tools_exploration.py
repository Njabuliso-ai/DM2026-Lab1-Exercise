from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel, Field, StrictInt
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


class CosineSimilarityInput(BaseModel):
    doc_index_1: StrictInt = Field(description="Zero-based index of the first document in the DTM.")
    doc_index_2: StrictInt = Field(description="Zero-based index of the second document in the DTM.")


class FeatureCorrelationInput(BaseModel):
    top_n: StrictInt = Field(default=20, description="Number of top-variance features to include.")


def make_tools(session):
    @tool(args_schema=CosineSimilarityInput)
    def cosine_similarity_tool(doc_index_1: int, doc_index_2: int) -> Dict[str, Any]:
        """Compute cosine similarity between two documents in the document-term matrix.

        Args:
            doc_index_1: Zero-based row index of the first document.
            doc_index_2: Zero-based row index of the second document.

        Returns:
            Dict containing result_id, doc_index_1, doc_index_2, and full-precision cosine_similarity.
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix (DTM) available in session. Run build_dtm_tool first."}

        n_samples = session.feature_matrix.shape[0]

        if doc_index_1 < 0 or doc_index_1 >= n_samples:
            return {"error": f"doc_index_1 ({doc_index_1}) is out of bounds for matrix with {n_samples} documents."}

        if doc_index_2 < 0 or doc_index_2 >= n_samples:
            return {"error": f"doc_index_2 ({doc_index_2}) is out of bounds for matrix with {n_samples} documents."}

        vec1 = session.feature_matrix[doc_index_1]
        vec2 = session.feature_matrix[doc_index_2]

        # Handle zero vectors and 2D reshaping for dense matrices
        if hasattr(vec1, "nnz"):
            is_vec1_zero = vec1.nnz == 0
            is_vec2_zero = vec2.nnz == 0
            v1_input, v2_input = vec1, vec2
        else:
            arr1 = np.asarray(vec1)
            arr2 = np.asarray(vec2)
            is_vec1_zero = np.all(arr1 == 0)
            is_vec2_zero = np.all(arr2 == 0)
            v1_input = arr1.reshape(1, -1) if arr1.ndim == 1 else arr1
            v2_input = arr2.reshape(1, -1) if arr2.ndim == 1 else arr2

        if is_vec1_zero or is_vec2_zero:
            sim_score = 0.0
        else:
            sim_mat = cosine_similarity(v1_input, v2_input)
            sim_score = float(sim_mat[0, 0])

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "cosine_similarity": sim_score,
        }

        full_report = pd.DataFrame([{
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "cosine_similarity": sim_score
        }])

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary,
            full_report=full_report
        )

        return summary

    @tool(args_schema=FeatureCorrelationInput)
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """Compute Pearson feature-vs-feature correlation matrix over top-variance terms and plot heatmap.

        Args:
            top_n: Number of top variance terms to include (default 20).

        Returns:
            Dict containing result_id, features, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix (DTM) or feature names available in session. Run build_dtm_tool first."}

        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            return {"error": "top_n must be a positive integer."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        feature_names = list(session.feature_names)

        n_samples, n_features = X.shape

        if n_samples < 2:
            return {"error": f"Feature correlation requires at least 2 documents, but got {n_samples}."}

        if len(feature_names) != n_features:
            return {"error": f"Mismatch between feature_names count ({len(feature_names)}) and feature_matrix columns ({n_features})."}

        # Calculate population variance (ddof=0) per column to match Known Answer #19
        variances = np.var(X, axis=0, ddof=0)

        # Sort deterministically by descending variance, then feature name
        feature_sort_keys = [(-v, name) for name, v in zip(feature_names, variances)]
        sorted_indices = [idx for idx, _ in sorted(enumerate(feature_sort_keys), key=lambda x: x[1])]

        actual_k = min(top_n, n_features)
        top_indices = sorted_indices[:actual_k]

        selected_features = [feature_names[i] for i in top_indices]
        X_selected = X[:, top_indices]

        # Compute Pearson correlation matrix
        if actual_k == 1:
            corr_mat = np.array([[1.0]])
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_mat = np.corrcoef(X_selected.T)
            corr_mat = np.nan_to_num(corr_mat, nan=0.0)
            np.fill_diagonal(corr_mat, 1.0)

        # Save artifacts
        session.artifacts["feature_correlation_matrix"] = corr_mat

        # Create heatmap plot
        corr_df = pd.DataFrame(corr_mat, index=selected_features, columns=selected_features)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1.0, vmax=1.0)
        ax.set_title(f"Top-{actual_k} Feature Correlation Matrix")
        fig.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "features": selected_features,
            "correlation_matrix": corr_mat.tolist(),
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_n": top_n},
            summary,
            full_report=corr_df
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]

from typing import Optional, Literal, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel, Field, StrictInt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import umap
from langchain_core.tools import tool


class ReduceDimensionsInput(BaseModel):
    method: Literal["pca", "tsne", "umap"] = Field(default="pca", description="Dimensionality reduction technique.")
    n_components: StrictInt = Field(default=2, description="Number of dimensions to reduce to.")
    perplexity: Optional[float] = Field(default=None, description="Perplexity parameter for t-SNE.")
    n_neighbors: Optional[int] = Field(default=None, description="Number of neighbors for UMAP.")
    random_state: Optional[int] = Field(default=None, description="Random seed for reproducible output.")


class BinarizeLabelsInput(BaseModel):
    pass


def make_tools(session):
    @tool(args_schema=ReduceDimensionsInput)
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: Optional[float] = None,
        n_neighbors: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reduce dimensionality of the document-term matrix (PCA, t-SNE, or UMAP) and plot projections.

        Args:
            method: Dimensionality reduction technique ('pca', 'tsne', or 'umap').
            n_components: Number of dimensions to reduce to (default 2).
            perplexity: Perplexity parameter for t-SNE (optional; must be > 0 and < n_samples).
            n_neighbors: Number of neighbors for UMAP (optional; must be >= 2 and <= n_samples).
            random_state: Random seed for reproducible output.

        Returns:
            Dict containing result_id, method, n_components, coordinates, and explained_variance_ratio (if PCA).
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix (DTM) available in session. Run build_dtm_tool first."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        n_samples, n_features = X.shape

        if n_samples == 0 or n_features == 0:
            return {"error": "Feature matrix is empty."}

        if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
            return {"error": "n_components must be a positive integer."}

        explained_variance_ratio = None

        if method == "pca":
            min_allowed = min(n_samples, n_features)
            if n_components > min_allowed:
                return {"error": f"n_components ({n_components}) cannot exceed min(n_samples, n_features) = {min_allowed} for PCA."}

            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X)
            explained_variance_ratio = model.explained_variance_ratio_.tolist()

        elif method == "tsne":
            if n_samples < 2:
                return {"error": f"t-SNE requires at least 2 samples, but got {n_samples}."}

            if perplexity is not None:
                if perplexity <= 0 or perplexity >= n_samples:
                    return {"error": f"perplexity ({perplexity}) must be > 0 and < n_samples ({n_samples})."}
                perp = float(perplexity)
            else:
                perp = max(1.0, min(30.0, float(n_samples - 1) / 3.0))
                if perp >= n_samples:
                    perp = max(1.0, float(n_samples - 1))

            model = TSNE(n_components=n_components, perplexity=perp, random_state=random_state, init="random", learning_rate=200.0)
            coords = model.fit_transform(X)

        elif method == "umap":
            if n_samples < 2:
                return {"error": f"UMAP requires at least 2 samples, but got {n_samples}."}

            if n_neighbors is not None:
                if n_neighbors < 2 or n_neighbors > n_samples:
                    return {"error": f"n_neighbors ({n_neighbors}) must be >= 2 and <= n_samples ({n_samples})."}
                n_neigh = int(n_neighbors)
            else:
                n_neigh = max(2, min(15, n_samples))

            model = umap.UMAP(n_components=n_components, n_neighbors=n_neigh, random_state=random_state)
            coords = model.fit_transform(X)

        else:
            return {"error": f"Unsupported reduction method: '{method}'"}

        # Store coords artifact
        session.artifacts["reduced_coords"] = coords

        # Category labels for scatter plot
        labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].tolist()
        elif session.labels is not None:
            labels = list(session.labels)

        # Build scatter plot figure
        fig, ax = plt.subplots(figsize=(8, 6))

        if n_components >= 2:
            x_vals = coords[:, 0]
            y_vals = coords[:, 1]
            x_label, y_label = "Dim 1", "Dim 2"
        else:
            x_vals = coords[:, 0]
            y_vals = np.zeros_like(x_vals)
            x_label, y_label = "Dim 1", "Projection"

        if labels is not None and len(labels) == n_samples:
            df_plot = pd.DataFrame({
                x_label: x_vals,
                y_label: y_vals,
                "Category": labels
            })
            sns.scatterplot(data=df_plot, x=x_label, y=y_label, hue="Category", style="Category", s=100, ax=ax)
        else:
            ax.scatter(x_vals, y_vals, s=100)

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()

        session.pending_figure = fig

        # Store result summary
        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "coordinates": coords.tolist(),
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        full_report = pd.DataFrame(coords, columns=[f"Dim_{i+1}" for i in range(n_components)])
        if labels is not None and len(labels) == n_samples:
            full_report["category_name"] = labels

        session.store_result(
            "reduce_dimensions_tool",
            {"method": method, "n_components": n_components, "perplexity": perplexity, "n_neighbors": n_neighbors, "random_state": random_state},
            summary,
            full_report=full_report
        )

        return summary

    @tool(args_schema=BinarizeLabelsInput)
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encode categorical labels into a dense binary indicator matrix.

        Reads category labels from session.dataframe ('category_name') or session.labels,
        applies LabelBinarizer, stores result artifacts, and returns the one-hot matrix.

        Returns:
            Dict containing result_id, categories, shape, and binarized_matrix.
        """
        if session.dataframe is None and session.labels is None:
            return {"error": "No dataframe or category labels available in session."}

        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels_series = session.dataframe["category_name"]
        elif session.labels is not None:
            labels_series = pd.Series(session.labels)
        else:
            return {"error": "No category labels found in session."}

        if labels_series.empty:
            return {"error": "Category label collection is empty."}

        if labels_series.isnull().any():
            return {"error": "Category labels contain null or missing values."}

        labels_list = labels_series.astype(str).tolist()

        if any(not label.strip() for label in labels_list):
            return {"error": "Category labels contain empty or whitespace-only values."}

        lb = LabelBinarizer()
        bin_matrix = lb.fit_transform(labels_list)
        classes = [str(c) for c in lb.classes_]

        # Handle binary classification (2 categories -> hstack to 2 dense columns)
        if len(classes) == 2 and bin_matrix.shape[1] == 1:
            bin_matrix = np.hstack([1 - bin_matrix, bin_matrix])
        elif len(classes) == 1 and bin_matrix.shape[1] == 1:
            bin_matrix = np.ones((len(labels_list), 1), dtype=int)

        bin_list = bin_matrix.tolist()

        # Store artifacts
        session.artifacts["binarized_labels"] = bin_matrix
        session.artifacts["label_binarizer"] = lb

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "categories": classes,
            "shape": list(bin_matrix.shape),
            "binarized_matrix": bin_list,
        }

        full_report = pd.DataFrame(bin_matrix, columns=classes)
        session.store_result(
            "binarize_labels_tool",
            {},
            summary,
            full_report=full_report
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]

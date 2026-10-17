from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import StrictInt
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session: Any) -> List[Any]:
    """Factory function that creates and returns the DTM tools bound to session."""

    @tool
    def build_dtm_tool(
        min_df: int = 1,
        max_df: float = 1.0,
        ngram_range: List[int] = [1, 1],
        max_features: Optional[int] = None,
        stop_words: Optional[str] = None
    ) -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from document text in the working session.

        Converts the text column in session.dataframe into a sparse term-frequency
        matrix using scikit-learn's CountVectorizer, and computes matrix sparsity.

        Args:
            min_df: Ignore terms that have a document frequency strictly lower than this threshold.
            max_df: Ignore terms that have a document frequency strictly higher than this threshold.
            ngram_range: Lower and upper boundary of the range of n-gram values to be extracted, as [min_n, max_n].
            max_features: Build a vocabulary that only considers the top max_features ordered by term frequency.
            stop_words: String like 'english' to remove standard stop words, or None to keep all words.

        Returns:
            Dict containing summary statistics of the built DTM (result_id, n_documents,
            vocabulary_size, non_zero, total_elements, sparsity_pct).
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {
                "error": "No dataframe loaded or 'text' column missing in session. Load a dataset first."
            }

        text_series = session.dataframe["text"].fillna("")

        # Convert List[int] schema type to tuple required by CountVectorizer
        ngram_tuple = tuple(ngram_range)

        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
            max_features=max_features,
            stop_words=stop_words
        )

        dtm = vectorizer.fit_transform(text_series)
        feature_names = list(vectorizer.get_feature_names_out())

        # Update session state
        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        # Calculate sparsity
        n_rows, n_cols = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = (100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": list(ngram_tuple),
            "max_features": max_features,
            "stop_words": stop_words
        }

        # Stored args should record all parameters used to generate this DTM result
        stored_args = {
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": ngram_range,
            "max_features": max_features,
            "stop_words": stop_words
        }

        session.store_result("build_dtm_tool", stored_args, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: Optional[StrictInt] = None) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the Document-Term Matrix.

        Calculates the total count of each term across all documents in session.feature_matrix
        and returns terms sorted by total frequency in descending order.

        Args:
            top_n: Optional positive integer limit on the number of top terms to return in the summary. If None, returns all terms.

        Returns:
            Dict containing result_id, total_terms, total_word_count, and term_frequencies.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix found in session. Run build_dtm_tool first."
            }

        # Check dimension alignment
        n_rows, n_cols = session.feature_matrix.shape
        if n_cols != len(session.feature_names):
            return {
                "error": f"Dimension mismatch: feature_matrix has {n_cols} columns but feature_names has {len(session.feature_names)} terms."
            }

        # Validate top_n argument (must be an integer > 0)
        if top_n is not None and top_n <= 0:
            return {
                "error": "top_n must be a positive integer."
            }

        # Calculate total frequency per term (sum across document rows)
        term_counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()

        # Sort terms by frequency descending
        sorted_indices = np.argsort(-term_counts)

        frequencies = []
        for idx in sorted_indices:
            frequencies.append({
                "term": str(session.feature_names[idx]),
                "frequency": int(term_counts[idx])
            })

        # DataFrame for visualization full_report (term as 1st col, frequency as 2nd col)
        full_report_df = pd.DataFrame(frequencies)

        display_frequencies = frequencies[:top_n] if top_n is not None else frequencies

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "total_word_count": int(term_counts.sum()),
            "term_frequencies": display_frequencies
        }

        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=full_report_df
        )
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: StrictInt = 20,
        n_documents: StrictInt = 20
    ) -> Dict[str, Any]:
        """Generates a heatmap visualization of a slice of the raw Document-Term Matrix.

        Slices the top n_documents rows and n_terms columns of session.feature_matrix,
        creates a seaborn heatmap plot on session.pending_figure, and returns the matrix slice.

        Args:
            n_terms: Positive integer number of term columns to include (default 20).
            n_documents: Positive integer number of document rows to include (default 20).

        Returns:
            Dict containing result_id, n_documents, n_terms, included_terms, included_documents, and matrix_slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix found in session. Run build_dtm_tool first."
            }

        if n_terms <= 0 or n_documents <= 0:
            return {
                "error": "n_terms and n_documents must be positive integers."
            }

        # Slice matrix up to n_documents and n_terms
        raw_slice = session.feature_matrix[:n_documents, :n_terms]
        dense_slice = raw_slice.toarray() if hasattr(raw_slice, "toarray") else np.asarray(raw_slice)

        actual_docs, actual_terms = dense_slice.shape
        included_terms = list(session.feature_names[:actual_terms])
        included_documents = [f"Doc {i}" for i in range(actual_docs)]

        # Generate heatmap plot and attach to session.pending_figure
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            dense_slice,
            xticklabels=included_terms,
            yticklabels=included_documents,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            ax=ax
        )
        ax.set_title(f"DTM Heatmap ({actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "included_terms": included_terms,
            "included_documents": included_documents,
            "matrix_slice": dense_slice.tolist()
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]

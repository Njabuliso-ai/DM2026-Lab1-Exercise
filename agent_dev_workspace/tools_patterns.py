
import os
import locale
import tempfile
from typing import Optional, Literal, Dict, Any, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, StrictInt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


class MinePatternsInput(BaseModel):
    category_name: str = Field(
        description="Category label string to filter documents by."
    )
    filtering_method: Literal[
        "variance", "tfidf", "term_frequency"
    ] = Field(description="Vocabulary filtering method.")
    algorithm: Literal[
        "fpgrowth", "topk", "maxfpgrowth"
    ] = Field(
        default="fpgrowth",
        description="Pattern mining algorithm."
    )
    min_sup: Optional[StrictInt] = Field(
        default=None,
        description="Minimum support count (positive integer)."
    )
    k: Optional[StrictInt] = Field(
        default=None,
        description="Top-k pattern count (positive integer)."
    )


def make_tools(session):

    @tool(args_schema=MinePatternsInput)
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal[
            "variance", "tfidf", "term_frequency"
        ],
        algorithm: Literal[
            "fpgrowth", "topk", "maxfpgrowth"
        ] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mine frequent patterns from documents in a selected category."""

        # Check that a dataset is loaded.
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        # Validate the mining parameters.
        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if (
                min_sup is None
                or isinstance(min_sup, bool)
                or not isinstance(min_sup, int)
                or min_sup < 1
            ):
                return {
                    "error": (
                        f"min_sup must be a positive integer "
                        f"for algorithm '{algorithm}'."
                    )
                }

        if algorithm == "topk":
            if (
                k is None
                or isinstance(k, bool)
                or not isinstance(k, int)
                or k < 1
            ):
                return {
                    "error": "k must be a positive integer for algorithm 'topk'."
                }

        # Filter documents by category.
        cat_df = session.dataframe[
            session.dataframe["category_name"] == category_name
        ]

        if cat_df.empty:
            return {
                "error": f"No documents found for category '{category_name}'."
            }

        texts = cat_df["text"].fillna("").tolist()

        artifact_key = (
            f"patterns_{category_name}_{filtering_method}_{algorithm}"
        )

        # Store an empty result when no terms survive.
        def return_empty_result(note: str) -> Dict[str, Any]:
            session.artifacts[artifact_key] = []

            result_id = session.next_result_id("mine_patterns")

            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": note,
            }

            empty_report = pd.DataFrame(
                columns=["pattern", "support"]
            )

            session.store_result(
                "mine_patterns_tool",
                {
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "min_sup": min_sup,
                    "k": k,
                },
                summary,
                full_report=empty_report,
            )

            return summary

        # Step 1: Build category-specific document-term matrix.
        try:
            vectorizer = CountVectorizer(stop_words="english")
            dtm_sparse = vectorizer.fit_transform(texts)
            feature_names = np.array(
                vectorizer.get_feature_names_out()
            )
        except ValueError:
            return return_empty_result(
                "empty vocabulary after text processing and stopword removal"
            )

        if len(feature_names) == 0:
            return return_empty_result(
                "no terms in category vocabulary after stopword removal"
            )

        dtm_dense = dtm_sparse.toarray()

        # Step 2: Apply feature filtering.
        if filtering_method == "variance":
            scores = (
                np.var(dtm_dense, axis=0, ddof=1)
                if dtm_dense.shape[0] > 1
                else np.var(dtm_dense, axis=0)
            )

            low_p = np.percentile(scores, 5)
            high_p = np.percentile(scores, 95)

            keep_mask = (scores > low_p) & (scores < high_p)

        elif filtering_method == "term_frequency":
            scores = np.sum(dtm_dense, axis=0)

            low_p = np.percentile(scores, 5)
            high_p = np.percentile(scores, 95)

            keep_mask = (scores > low_p) & (scores < high_p)

        elif filtering_method == "tfidf":
            tfidf = (
                TfidfTransformer()
                .fit_transform(dtm_sparse)
                .toarray()
            )

            scores = np.mean(tfidf, axis=0)
            p20 = np.percentile(scores, 20)

            keep_mask = scores >= p20

        else:
            return {
                "error": f"Unsupported filtering_method: {filtering_method}"
            }

        surviving_terms = feature_names[keep_mask]
        surviving_dtm = dtm_dense[:, keep_mask]

        if len(surviving_terms) == 0:
            return return_empty_result("no terms survived filtering")

        # Step 3: Convert to transactional database and mine with PAMI.
        bin_df = pd.DataFrame(
            surviving_dtm,
            columns=surviving_terms
        )

        patterns_list: List[Dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(
                tmp_dir,
                "transactional_db.txt"
            )

            from PAMI.extras.convert import DF2DB

            converter = DF2DB.DF2DB(
                bin_df,
                DFtype="dense"
            )

            converter.convert2TransactionalDatabase(
                db_path,
                condition=">",
                thresholdValue=0.5
            )

            # =====================================================
            # UTF-8 ENCODING FIX
            # =====================================================
            # PAMI DF2DB writes using the default text encoding
            # on Windows. The mining algorithms read UTF-8.
            # Convert the transaction file before mining.

            source_encoding = locale.getencoding()

            with open(
                db_path,
                "r",
                encoding=source_encoding
            ) as f:
                transaction_text = f.read()

            with open(
                db_path,
                "w",
                encoding="utf-8"
            ) as f:
                f.write(transaction_text)

            print(
                "Converted transaction file from",
                source_encoding,
                "to UTF-8"
            )

            # =====================================================
            # UTF-8 DIAGNOSTIC
            # =====================================================
            # Confirm that the transaction file is now valid UTF-8.

            with open(db_path, "rb") as f:
                raw_bytes = f.read()

            print(
                "Transaction file size:",
                len(raw_bytes),
                "bytes"
            )

            try:
                raw_bytes.decode("utf-8")
                print("Transaction file is valid UTF-8.")

            except UnicodeDecodeError as e:
                print("Transaction file is NOT valid UTF-8.")
                print("Decode error:", e)
                raise

            # Run the selected mining algorithm.
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth

                obj = FPGrowth.FPGrowth(
                    db_path,
                    minSup=min_sup
                )

                if hasattr(obj, "startMine"):
                    obj.startMine()
                elif hasattr(obj, "mine"):
                    obj.mine()

                df_res = obj.getPatternsAsDataFrame()

            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE

                obj = FAE.FAE(
                    db_path,
                    k=k
                )

                if hasattr(obj, "startMine"):
                    obj.startMine()
                elif hasattr(obj, "mine"):
                    obj.mine()

                df_res = obj.getPatternsAsDataFrame()

            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth

                obj = MaxFPGrowth.MaxFPGrowth(
                    db_path,
                    minSup=min_sup
                )

                if hasattr(obj, "startMine"):
                    obj.startMine()
                elif hasattr(obj, "mine"):
                    obj.mine()

                df_res = obj.getPatternsAsDataFrame()

            # Extract patterns and their support counts.
            if df_res is not None and not df_res.empty:
                for _, row in df_res.iterrows():
                    raw_pat = row["Patterns"]

                    if isinstance(raw_pat, str):
                        pat_items = raw_pat.strip().split()
                    else:
                        pat_items = list(raw_pat)

                    sup = int(row["Support"])

                    patterns_list.append({
                        "pattern": pat_items,
                        "support": sup
                    })

        # Step 4: Store results and artifacts.
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id(
            "mine_patterns"
        )

        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns_list,
            "n_patterns": len(patterns_list),
        }

        full_report = (
            pd.DataFrame(patterns_list)
            if patterns_list
            else pd.DataFrame(
                columns=["pattern", "support"]
            )
        )

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
            full_report=full_report,
        )

        return summary

    return [mine_patterns_tool]

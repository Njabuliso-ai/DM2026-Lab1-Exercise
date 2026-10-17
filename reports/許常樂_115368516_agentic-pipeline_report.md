# Agentic Pipeline Analysis Report

**Author:** 許常樂 (115368516)  
**Dataset:** 20 Newsgroups (Four Categories)

## 1. Executive Summary

This report documents an agent-assisted data mining analysis of 2,257 documents from four categories of the 20 Newsgroups dataset. The pipeline included dataset inspection, data quality checks, tokenization, document-term matrix construction, term frequency analysis, cosine similarity, feature correlations, feature filtering, frequent pattern mining, and dimensionality reduction.

## 2. Dataset Overview

**Results:** `load_dataset_1`, `inspect_2`

The dataset contains 2,257 documents and three columns: `text`, `category`, and `category_name`.

| Category | Documents |
|---|---:|
| soc.religion.christian | 599 |
| sci.med | 594 |
| comp.graphics | 584 |
| alt.atheism | 480 |
| **Total** | **2,257** |

The text includes original Usenet post content, including message headers and other metadata.

## 3. Data Quality Checks

**Missing values:** `check_missing_3`

No missing values were found in the three dataset columns.

**Duplicate records:** `check_duplicates_4`

No fully identical rows were detected. This does not necessarily exclude documents with identical text but different labels.

## 4. Text Preprocessing and Document-Term Matrix

### Tokenization

**Result:** `tokenize_5`

The tokenization operation produced 903,209 tokens. Raw text contained ordinary words, punctuation, and Usenet metadata.

### Document-Term Matrix

**Result:** `dtm_6`

The CountVectorizer configuration was:

- `stop_words='english'`
- `max_features=2000`
- `min_df=5`
- `max_df=0.90`

The resulting document-term matrix had:

- 2,257 documents
- 2,000 features
- 144,498 nonzero entries
- Approximately 96.80% sparsity

The matrix represents raw term counts.

## 5. Exploratory Analysis

### Term Frequency

**Result:** `term_freq_7`

Total term occurrences: 229,045.

| Term | Frequency |
|---|---:|
| edu | 4,675 |
| god | 2,216 |
| com | 2,172 |
| writes | 1,645 |
| people | 1,604 |
| article | 1,453 |
| don | 1,400 |
| know | 1,230 |
| think | 1,213 |
| like | 1,177 |

Several high-frequency terms originate from Usenet message metadata, suggesting that header removal could improve topic-focused analysis.

**Visualization:** `plots/許常樂_agentic-pipeline_plot_turn52.png`

### Cosine Similarity

**Results:** `cosine_similarity_8`, `cosine_similarity_9`

| Document pair | Categories | Cosine similarity |
|---|---|---:|
| Documents 0 and 1 | Both comp.graphics | 0.074928 |
| Documents 0 and 2 | comp.graphics and soc.religion.christian | 0.076129 |

The between-category pair was slightly more similar than the selected within-category pair. Two comparisons are insufficient to establish general category similarity or separation.

## 6. Feature Correlation Analysis

**Result:** `feature_correlation_10`

Selected correlations included:

| Term pair | Correlation |
|---|---:|
| jpeg / gif | +0.9615 |
| graphics / pub | +0.9229 |
| jpeg / file | +0.9044 |
| image / images | +0.8861 |
| edu / com | +0.6417 |
| god / graphics | -0.0286 |

High correlations may reflect repeated terminology, document-specific word counts, or metadata. Correlation does not establish causation.

**Visualization:** `plots/許常樂_agentic-pipeline_plot_feature_correlation_10.png`

## 7. Feature Filtering

### Variance Filtering

**Results:** `variance_11`, `variance_12`

| Variance threshold | Features retained | Features removed |
|---|---:|---:|
| 0.01 | 2,000 | 0 |
| 0.05 | 1,269 | 731 |

The 0.05 threshold removed 36.55% of the original features.

### Pearson Correlation Filtering

**Result:** `pearson_13`

- Target category: `comp.graphics`
- Selection criterion: absolute Pearson correlation >= 0.10
- Retained features: 189
- Removed features: 1,811 (90.55%)

Selected positive associations:

- `thanks`: +0.2778
- `keywords`: +0.2329
- `vga`: +0.2111
- `video`: +0.2095
- `graphics`: +0.1779

Selected negative associations:

- `writes`: -0.2337
- `people`: -0.2089
- `god`: -0.1643
- `christian`: -0.1475

### Spearman Correlation Filtering

**Result:** `spearman_14`

- Target category: `comp.graphics`
- Selection criterion: absolute Spearman correlation >= 0.10
- Retained features: 313
- Removed features: 1,687 (84.35%)

Selected positive associations:

- `graphics`: +0.5211
- `3d`: +0.3060
- `program`: +0.2911
- `file`: +0.2837
- `gif`: +0.2582

Selected negative associations:

- `people`: -0.3193
- `god`: -0.3073
- `writes`: -0.2562

Pearson and Spearman selected different feature sets because Pearson measures linear relationships in the original values, whereas Spearman measures rank-based associations. These results do not establish that either method is universally better.

Because these methods use class labels, feature selection should be performed within training folds when used in predictive modeling to avoid data leakage.

## 8. Frequent Pattern Mining

**Category:** `comp.graphics`  
**Filtering:** Term frequency  
**Algorithm:** FP-Growth

| Result | Minimum support | Patterns discovered |
|---|---:|---:|
| mine_patterns_16 | 30 | 0 |
| mine_patterns_17 | 5 | 198,719 |

At minimum support 5, the most frequent single-term patterns included:

| Term | Support |
|---|---:|
| rumours | 21 |
| wondering | 21 |
| hard | 20 |
| pascal | 19 |
| ram | 18 |

Reducing the support threshold from 30 to 5 caused a substantial increase in discovered patterns.

The results illustrate the sensitivity of frequent-pattern mining to the support parameter. Many discovered patterns may reflect general language or Usenet metadata rather than meaningful topic associations.

The PAMI transaction-file encoding issue was corrected by converting the generated file from the Windows default encoding to UTF-8 before mining.

## 9. Dimensionality Reduction

### Principal Component Analysis (PCA)

**Result:** `reduce_dimensions_18`

- Input dimensions: 2,257 x 2,000
- Output dimensions: 2,257 x 2
- Random state: 42
- PC1 explained variance: 18.31%
- PC2 explained variance: 8.18%
- Total explained variance: 26.49%

The first two principal components captured approximately 26.49% of the variance in the document-term matrix. The remaining variance lies outside this two-dimensional representation.

**Visualization:** `plots/許常樂_agentic-pipeline_plot_reduce_dimensions_18.png`

### t-SNE

**Result:** `reduce_dimensions_19`

- Output dimensions: 2,257 x 2
- Perplexity: 30
- Random state: 42

The agent reported localized document groupings with partial overlap between categories. This is an exploratory visual observation, not a quantitative clustering evaluation.

Unlike PCA, t-SNE does not provide explained variance ratios. PCA emphasizes global linear variance, while t-SNE emphasizes local neighborhood relationships. The methods serve different purposes.

**Visualization:** `plots/許常樂_agentic-pipeline_plot_reduce_dimensions_19.png`

## 10. Key Findings and Limitations

1. The document-term matrix was highly sparse, with approximately 96.80% zero entries.
2. Usenet metadata contributed substantially to common term frequencies.
3. Variance, Pearson, and Spearman filtering reduced the feature space by different amounts.
4. Frequent-pattern mining was highly sensitive to the minimum-support threshold.
5. PCA captured 26.49% of variance in two dimensions, while t-SNE provided a different exploratory view of local document relationships.
6. Pairwise similarity examples and visual cluster impressions should not be generalized without further evaluation.
7. Feature correlations may be influenced by outlier documents and repeated metadata.

## 11. Conclusion

The agent-assisted pipeline successfully demonstrated the main stages of text data mining, from initial data inspection through feature selection, frequent-pattern discovery, and dimensionality reduction.

The results emphasize the importance of text preprocessing, careful parameter selection, and cautious interpretation of exploratory visualizations.

Future improvements could include removing Usenet headers, evaluating TF-IDF representations, testing alternative feature-selection settings, and quantitatively assessing category separability.

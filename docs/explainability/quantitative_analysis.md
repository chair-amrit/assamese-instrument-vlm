# Quantitative Analysis — Cross-Modal Explainability

## Overview

This document reports the numerical results of the cross-modal attention and
attribution analysis across 35 curated samples (5 per instrument, spanning all 7
failure categories), following the protocol described in `protocol.md`. All
attribution figures use the validated `n_steps=50` configuration (see
Stability Validation below).

## Sample Coverage

| | Count |
|---|---|
| Total curated samples | 35 |
| Attention results successfully processed | 35/35 |
| Attribution results successfully processed | 35/35 |
| Instruments covered | 7 (5 samples each) |
| Failure categories covered | 7 |

## Attribution Stability Validation

Before adopting final attribution results, a stability check compared
`n_steps=10` against `n_steps=50` on 5 samples (one per distinct failure
category, subject to availability), using cosine similarity between the
resulting patch-level attribution maps.

| Instrument | Failure Category | Cosine Similarity (n=10 vs n=50) | Stable (>0.9)? |
|---|---|---|---|
| Bahi | Correct | 0.6334 | No |
| Bahi | Hallucination | 0.7405 | No |
| Toka | Mixed Attribute | 0.1833 | No |
| Toka | Partial Answer / Incomplete Answer | 0.2690 | No |
| Bahi | Question Misunderstanding | 0.5775 | No |
| **Average** | | **0.4807** | **No** |

**Conclusion:** `n_steps=10` produces unstable, unreliable attribution maps.
All 35 samples were reprocessed at `n_steps=50` before final aggregation, which
is the configuration reported in all results below.

## Per-Failure-Category Results

| Category | n Samples | Avg. Attention Coverage | Avg. Attribution Coverage (n_steps=50) |
|---|---|---|---|
| Correct | 7 | 0.0099 | 0.0025 |
| Question Misunderstanding | 7 | 0.0048 | 0.0027 |
| Hallucination | 5 | 0.0122 | 0.0020 |
| Partial Answer / Incomplete Answer | 4 | 0.0068 | 0.0020 |
| Truncation | 6 | 0.0062 | 0.0025 |
| Repetition | 2 | 0.0097 | 0.0019 |
| Mixed Attribute | 4 | 0.0147 | 0.0021 |

**Coverage ratio** is defined as the proportion of the upsampled heatmap with
normalized intensity above 0.5, i.e. the fraction of the image area receiving
strong attention/attribution.

### Key patterns

- **Highest attention coverage:** Mixed Attribute (0.0147), followed by
  Hallucination (0.0122)
- **Lowest attention coverage:** Question Misunderstanding (0.0048)
- **Attribution coverage** is largely flat across categories (range: 0.0019 to
  0.0027), showing no strong category-dependent pattern at this sample size
- Attention coverage varies by roughly **3x** between the highest and lowest
  categories, while attribution coverage varies by less than **1.5x** —
  attention is more sensitive to failure type than attribution is

## Per-Instrument Results

| Instrument | n Samples | Avg. Attention Coverage | Failure Category Distribution |
|---|---|---|---|
| Bahi | 5 | 0.0063 | Correct: 1, Question Misunderstanding: 1, Truncation: 1, Hallucination: 1, Repetition: 1 |
| Bihu Dhol | 5 | 0.0154 | Correct: 1, Mixed Attribute: 1, Partial Answer: 1, Repetition: 1, Hallucination: 1 |
| Gogona | 5 | 0.0057 | Correct: 1, Mixed Attribute: 1, Question Misunderstanding: 1, Hallucination: 1, Truncation: 1 |
| Khutitaal | 5 | 0.0039 | Correct: 1, Question Misunderstanding: 1, Partial Answer: 1, Truncation: 1, Hallucination: 1 |
| Pepa | 5 | 0.0107 | Correct: 1, Mixed Attribute: 1, Question Misunderstanding: 2, Truncation: 1 |
| Toka | 5 | 0.0086 | Correct: 1, Question Misunderstanding: 1, Partial Answer: 1, Truncation: 1, Mixed Attribute: 1 |
| Xutuli | 5 | 0.0106 | Correct: 1, Question Misunderstanding: 1, Partial Answer: 1, Truncation: 1, Hallucination: 1 |

**Range:** attention coverage varies roughly 4x between the lowest (Khutitaal,
0.0039) and highest (Bihu Dhol, 0.0154) instruments.

## Figures

| File | Description |
|---|---|
| `category_grid_attention.png` | 7-panel grid, one representative attention heatmap per failure category |
| `category_coverage_comparison_n50.png` | Bar chart comparing attention vs. attribution coverage across all 7 categories |

## Interpretation

The overall low magnitude of both attention and attribution coverage (all values
under 0.015) indicates the model relies on a small, concentrated subset of image
patches when generating answers, rather than integrating information broadly
across the visual scene — true even for correct predictions. Combined with the
qualitative observation that hotspots frequently fall on background rather than
instrument regions, this raises the question of how much of the model's accuracy
on this task stems from genuine visual-cultural grounding versus dataset-level
regularities exploitable without full scene understanding.

The divergence between attention (category-sensitive) and attribution
(category-flat) patterns suggests these two methods are capturing different
aspects of model behavior: attention reflects where computational focus is
allocated, while attribution reflects what causally determines the output
token. Their weak correlation here is itself a finding — it indicates that
attention weights alone would be a misleading proxy for genuine visual
grounding in this model, reinforcing the choice to use both methods jointly
rather than relying on attention (or GradCAM-style) analysis alone.

## Limitations

- Category sample sizes are small (2–7), and results should be read as
  indicative rather than statistically definitive
- Coverage ratio is a single summary statistic; it does not capture the spatial
  location or semantic relevance of hotspots (addressed qualitatively in
  `qualitative_analysis.md`)
- All results are specific to this fine-tuned checkpoint and this 35-sample
  curated subset; broader validation across the full test set was outside the
  scope of this explainability pass
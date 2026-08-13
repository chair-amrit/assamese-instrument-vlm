# Qualitative Analysis — Cross-Modal Explainability

## Overview

This document describes the visual patterns observed in the cross-modal attention
and attribution heatmaps across the 35 curated samples, organized by failure
category. All heatmaps referenced here are available in
`results/explainability/attention/` and `results/explainability/attribution/`,
with the summary grid at `results/explainability/category_grid_attention.png`.

## General Observations

Across nearly all samples, regardless of failure category, attention is **sparse
and localized** rather than broadly distributed across the instrument. Heatmaps
consistently show one or two small hotspot regions rather than diffuse coverage
of the instrument's visible extent. This pattern holds even for **Correct**
predictions — the model reaching the right answer does not require, and does not
appear to involve, attending broadly to the instrument's visual structure.

A recurring and notable pattern: hotspots frequently fall on **background or
edge regions** (wall texture, floor tiles, image borders) rather than on the
instrument body itself. This is visible in the representative "Correct" example
(Bahi, material question) shown in the protocol validation, where the strongest
attention region lands on the wall behind the instrument rather than on the
bamboo body.

## Per-Category Patterns

**Correct** — Attention is present but weak and often off-object. The model
produces the right answer with minimal apparent reliance on instrument-specific
visual features, suggesting partial dependence on language priors or dataset-level
regularities (e.g. material questions defaulting to commonly correct answers for
a given instrument class) rather than purely visual grounding.

**Question Misunderstanding** — This category shows the **lowest** average
attention coverage of all seven categories. The model appears to engage with the
image only minimally when it misreads the question, consistent with the failure
originating in language comprehension rather than visual processing — the model
isn't looking at the wrong thing, it's largely not looking at all.

**Hallucination** — Attention coverage here is among the **highest** of all
categories, which is counterintuitive at first glance. Rather than showing
under-attention (as might be expected if hallucination meant "ignoring the
image"), hallucinated answers are associated with broader, more scattered
attention across multiple unrelated regions. This suggests hallucination may
arise less from ignoring the image outright and more from unfocused or diffuse
visual engagement that fails to anchor on task-relevant features.

**Partial Answer / Incomplete Answer** — Attention patterns are moderate and
somewhat scattered, with hotspots appearing in multiple small clusters rather
than one dominant region. This is broadly consistent with an answer that
captures some but not all relevant attributes — the visual evidence gathered
is real but incomplete.

**Truncation** — Attention here resembles the "Correct" pattern in intensity but
with hotspots sometimes appearing near the edges of the token sequence region,
plausibly related to the generation being cut short rather than a distinct
visual failure.

**Repetition** — The lowest sample count of any category (n=2) limits confidence,
but observed heatmaps show diffuse, low-intensity attention with no clear single
hotspot — consistent with the model cycling through similar internal states
during repeated generation.

**Mixed Attribute** — This category shows the **highest** average attention
coverage among all seven categories, with multiple distinct hotspots often
visible simultaneously in a single heatmap. This is visually consistent with the
underlying failure mode: the model appears to be drawing on visual evidence from
more than one attribute or region at once, and conflating them in the generated
answer.

## Attention vs. Attribution Divergence

A consistent finding across all categories: **attention coverage varies
meaningfully between failure types** (ranging roughly 3x between the lowest and
highest categories), while **attribution coverage remains nearly flat** across
all categories at a much lower absolute level. This suggests that where the
model *looks* (attention) shifts depending on the type of failure it is about to
make, but *what actually drives* the output token (attribution) stays weakly and
uniformly distributed regardless of outcome. In other words, the model's stated
focus and its causal reliance on visual evidence are not well aligned — a finding
with direct relevance to the project's core question of whether the model
semantically understands what it is looking at, rather than merely producing
plausible-sounding text conditioned loosely on an image.

## Instrument-Level Observations

Attention coverage also varies by instrument independent of failure category.
Khutitaal shows the lowest average attention coverage (0.0039) across its 5
samples, while Bihu Dhol shows the highest (0.0154) — roughly a 4x difference.
Given the small per-instrument sample size (n=5), this is noted as an
observation for further investigation rather than a robust conclusion, but may
reflect differences in visual complexity or background clutter across the
instrument image sets.

## Limitations

- Sample sizes per failure category are small and uneven (2 to 7 samples),
  limiting the statistical strength of category-level comparisons
- Representative heatmaps shown in the summary grid are single examples per
  category, not averages, and may not fully capture within-category variation
- Attribution results reflect the validated `n_steps=50` configuration; earlier
  exploratory runs at `n_steps=10` were found to be unstable (see
  `quantitative_analysis.md`) and are not used in this analysis
# Taxonomy Methodology — Overview

This directory contains the revised VQA failure-taxonomy methodology for the Assamese Musical Instrument VLM project. This README is a compact orientation guide; the four linked documents are the authoritative source for all definitions, equations, and decision rules.

---

## 1. Purpose and Motivation

The prior taxonomy defined seven categories directly, with category-specific decision variables (e.g. an attribute-consistency function designed around Question Misunderstanding). This caused conflation: a non-answer or refusal could be misclassified as Question Misunderstanding merely because it failed to address the requested attribute, even though the model may have understood the question and simply declined to answer.

The revised methodology replaces this category-first design with an **axis-first design**: independent behavioral properties of a prediction are measured first, and the seven final categories are derived afterward from combinations of those measurements. This removes the specific conflation identified above and is intended to produce a taxonomy that is precise, deterministic, semantically grounded, mutually exclusive, collectively covering, and reproducible.

---

## 2. Axis-First Methodology

Every prediction is represented as a tuple $T = (I, Q, G, P)$ (image, question, ground truth, prediction). Rather than asking "which of seven categories does this belong to," the methodology first measures a fixed set of independent behavioral signals, then derives the category from the resulting signal combination.

The framework measures **eight signals organized into seven conceptual axes**:

| Axis | Signal | Domain |
|---|---|---|
| 1 | Question/Semantic Alignment | aligned, misaligned, indeterminate |
| 2a | Substantive Content Present | yes, no |
| 2b | Uncertainty/Refusal Marker | yes, no |
| 3 | Semantic Correctness | correct, incorrect, mixed, not applicable, indeterminate |
| 4 | Completeness | complete, partial, not applicable, indeterminate |
| 5 | Termination Integrity | intact, truncated |
| 6 | Repetition | absent, present |
| 7 | Unsupported Content | none, present |

Axis 2 comprises two independent sub-signals (2a, 2b); all eight are measurable signals within the seven conceptual axes. Full domains, preconditions, and measurement order are defined in `02_decision_functions.md`.

---

## 3. Claim Routing and Required Attributes

A prediction is decomposed into minimal, checkable claims and partitioned according to whether each claim addresses the attribute the question actually requires:

$$
P \rightarrow (P_K, P_{\bar K})
$$

$P_K$ (on-topic claims) feeds both Axis 3 (Semantic Correctness) and Axis 7 (Unsupported Content). $P_{\bar K}$ (off-topic claims) feeds Axis 7 only. This ensures off-topic digressions cannot be scored as correct or incorrect against a ground truth that doesn't cover them, while still catching fabricated content anywhere in the response.

The required attribute set for each question template, $A_{\text{set}}(K)$, is fixed by the predefined question-template specification — not inferred from the ground truth or the model's prediction. Templates q1–q8 each require a single attribute; q9 (description) is a composite template requiring two attributes: cultural significance and role in Assamese music. Full definitions are in `01_mathematical_foundation.md`.

---

## 4. Core Categories and Cross-Cutting Flags

The taxonomy defines seven core categories:

- Non-Answer / Abstention
- Question Misunderstanding
- Incoherent Response
- Hallucination (redefined: topically-aligned but semantically incorrect content — not the retired unsupported-content indicator)
- Correct
- Partial Answer
- Mixed Attribute

Under **Design A**, final classification is a core category paired with three independent flags, rather than a single flat label:

$$
\tau(T) = \left(C_r,\ \mathrm{Ax}_5,\ \mathrm{Ax}_6,\ \mathrm{Ax}_7\right)
$$

Truncation, Repetition, and Unsupported Content are cross-cutting flags, not standalone categories — a response can be, for example, "Correct + truncated" or "Partial Answer + truncated + unsupported." This preserves the semantic state of a response even when a structural issue (truncation, repetition) or content issue (unsupported detail) also occurs. Category and flag definitions are in `01_mathematical_foundation.md` and `03_quantitative_formulation.md`.

---

## 5. Classification Pipeline

The four documents form a linear derivation:


```
01 → Mathematical foundation
02 → Decision functions
03 → Quantitative formulation
04 → Taxonomy algorithm
```

`04` is the implementation-facing document: it specifies the exact sequence of measurement and derivation steps applied to a single prediction, culminating in a final $\tau(T)$.

---

## 6. Manual Review ($C_{\text{review}}$)

Two signal states cannot be automatically resolved to one of the seven categories: Axis 3 = indeterminate (no comparable evidence, or comparable evidence that cannot be reliably verified), and Axis 4 = indeterminate (coverage cannot be reliably determined). Rather than forcing these into an arbitrary category, they are routed to an internal outcome, $C_{\text{review}}$, which is **not** a member of the final category space.

$C_{\text{review}}$ instances require manual resolution: a reviewer re-examines the prediction, resolves the indeterminate signal, and recomputes any downstream dependent signal. Every $C_{\text{review}}$ instance must be resolved to one of the seven core categories before final dataset statistics are computed. Full resolution procedure is in `04_taxonomy_algorithm.md`.

---

## 7. Documented Reference-Data Limitation (Toka, q9)

The authored ground truth for the instrument Toka's q9 (description) response does not independently realize the `cultural significance` attribute, separate from `role_in_assamese_music` content. The required attribute set for q9 remains fixed at two attributes for every instrument, including Toka — this is treated as a **reference-data authoring gap**, not a taxonomy exception. As a direct consequence, Toka's q9 completeness is structurally capped at `partial`, and $C_{\text{correct}}$ cannot be reached through the `complete` branch for this instrument–template pair under the current reference annotation. Any reported completeness statistic involving Toka q9 must explicitly disclose this limitation. Full details are in `01_mathematical_foundation.md` (Section 6) and `03_quantitative_formulation.md` (Section 10).

---

## 8. Methodology Documents

- [`01_mathematical_foundation.md`](./01_mathematical_foundation.md) — objects, spaces, axes, categories, notation
- [`02_decision_functions.md`](./02_decision_functions.md) — signal measurement rules, preconditions, branches
- [`03_quantitative_formulation.md`](./03_quantitative_formulation.md) — category derivation, flag attachment, coverage and exclusivity
- [`04_taxonomy_algorithm.md`](./04_taxonomy_algorithm.md) — ordered classification procedure, review resolution, scope

```
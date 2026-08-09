# Methodology

This directory documents the mathematical framework used to analyze and classify prediction failures for the Assamese Musical Instrument Visual Question Answering (VQA) model. It defines the formal objects, evaluation functions, quantitative criteria, and decision algorithm that together convert raw model predictions into a structured, reproducible failure taxonomy.

## 1. Overview

Manually reviewing model predictions and labeling failures by eye does not scale and is not reproducible across reviewers or across runs. This methodology replaces that manual process with an explicit, ordered evaluation pipeline: every prediction is evaluated against a fixed set of mathematical criteria and assigned either to one of the seven final failure categories or, when the available evidence is insufficient for automatic classification, to an internal Review outcome. All Review cases must be resolved before final category statistics are reported.

The pipeline proceeds as follows:

\`\`\`
VQA Sample → Model Prediction → Evaluation Functions → Quantitative Conditions → Taxonomy Algorithm → Category Statistics
\`\`\`

Each stage is defined precisely in one of the four methodology documents listed below. The fixed evaluation definitions, thresholds, decision priority, and documented Review procedure are intended to ensure consistent categorization across runs and reviewers.
---

## 2. Methodology Documents

The framework is developed incrementally across four documents. Each document defines the inputs required by the next.

| Document | Defines | Feeds into |
|---|---|---|
| [`01_mathematical_foundation.md`](./01_mathematical_foundation.md) | The core object spaces — images, questions, ground truth, predictions — and the formal representation of a VQA sample and its prediction. | The objects that all later evaluation functions operate on. |
| [`02_decision_functions.md`](./02_decision_functions.md) | The evaluation functions that compare a ground truth and a prediction — question alignment, hallucination, truncation, repetition, and mixed-attribute checks. | The raw signals used to quantify prediction quality. |
| [`03_quantitative_formulation.md`](./03_quantitative_formulation.md) | The measurable indicators and thresholds derived from the decision functions, including the completeness score κ. | The concrete values the taxonomy algorithm branches on. |
| [`04_taxonomy_algorithm.md`](./04_taxonomy_algorithm.md) | The deterministic, priority-ordered algorithm that maps each prediction to one of the seven final failure categories or, when necessary, to the internal Review outcome. | The final per-prediction category used for dataset-level statistics after Review resolution. |

---

## 3. Mathematical Framework

All later documents build on a small set of core mathematical objects, formally defined in [`01_mathematical_foundation.md`](./01_mathematical_foundation.md). This section summarizes them at a high level, without reproducing the full definitions.

**Object spaces**

| Symbol | Space |
|---|---|
| 𝕀 | Image space |
| ℚ | Question space |
| 𝔾 | Ground-truth answer space |
| ℙ | Prediction space |
| 𝕋 | Prediction-tuple space |
| ℂ | Failure-category space |

Two further spaces support semantic evaluation: a **semantic concept space**, representing the underlying concept a question targets (e.g. material, origin, sound), and an **attribute space**, representing the specific value expected in a correct answer (e.g. a particular material or instrument name).

**Core functions**

- **fθ** — the fine-tuned VQA model, mapping an image and question to a predicted answer.
- **Concept function** — maps a question to the semantic concept being evaluated.
- **Attribute interpretation functions** — identify the relevant semantic attribute values in the ground-truth and predicted answers.
- **Evaluation and taxonomy functions** — evaluate the relationship between the ground truth and prediction and ultimately assign an operational taxonomy outcome.

This is consistent with the notation fixed in the foundation document: a sample is x = (I, Q, G), a prediction is P = f_θ(I, Q), and a complete prediction instance is T = (I, Q, G, P).



## 4. Decision-Function Layer

A single binary correctness check — whether a prediction exactly matches the ground truth — is not sufficient to characterize *how* a prediction fails. Two incorrect predictions can fail for entirely different reasons: one might introduce unsupported content, another might be truncated or repetitive, and a third might simply misinterpret the question. Treating all of these as a single "incorrect" bucket would discard information needed for meaningful failure analysis.

To address this, the methodology defines a set of evaluation mechanisms in [`02_decision_functions.md`](./02_decision_functions.md). Each evaluation mechanism inspects a specific aspect of the relationship between the ground truth G and the prediction P and produces evidence that is used by the quantitative formulation and taxonomy algorithm.

| Function | Symbol | Checks for |
|---|---|---|
| Question alignment | m_Q | Whether the prediction addresses the question that was actually asked. |
| Ground-truth/prediction match | m_G | Whether the prediction's content semantically matches the ground truth. |
| Completeness | κ | Measures the degree to which the prediction covers the required information in the ground truth. |
| Hallucination | H | Whether the prediction introduces unsupported factual content relative to the available ground-truth evidence. |
| Truncation | R_tr | Whether the prediction is cut off or incomplete due to generation length. |
| Repetition | R_rep | Whether the prediction contains unnecessary or degenerate repeated content. |
| Mixed attribute | MA | Whether the prediction contains both correct and incorrect attribute-level information for the evaluated concept. |

Together, these mechanisms provide the evidentiary signals — not the final decision — that the taxonomy algorithm uses to classify each prediction.

---

## 5. Quantitative Formulation

The evaluation mechanisms above are qualitative checks; the quantitative formulation in [`03_quantitative_formulation.md`](./03_quantitative_formulation.md) converts them into concrete, measurable values and fixed thresholds so that classification is reproducible rather than subjective.

| Indicator | Description |
|---|---|
| Semantic consistency | Measures whether the prediction's meaning aligns with the ground truth's meaning, beyond exact string matching. |
| Question alignment score | Quantifies whether the prediction addresses the concept the question is asking about. |
| Completeness κ | A continuous score representing how much of the expected ground-truth content is present in the prediction. |
| Hallucination indicator | A binary flag derived from the hallucination check, indicating unsupported content in the prediction. |
| Truncation / repetition indicators | Binary structural flags identifying truncation or unnecessary repeated content; these conditions are evaluated before the semantic failure categories. |
| Mixed-attribute indicator | A binary flag identifying predictions that contain both correct and incorrect attribute-level information for the evaluated concept. |
| Completeness threshold τ_complete | The fixed cutoff value of κ above which a prediction is considered complete rather than partial. |

These indicators and the predefined completeness threshold provide the quantitative evidence used by the priority-ordered decision rules in the taxonomy algorithm.
---

## 6. Taxonomy Decision Algorithm

The taxonomy algorithm operates over the extended outcome space consisting of the seven final taxonomy categories and the internal Review outcome. Conditions are evaluated top to bottom, and the first condition that holds determines the operational outcome:

1. Truncation
2. Repetition
3. Question Misunderstanding
4. Mixed Attribute
5. Hallucination
6. Partial Answer / Incomplete Answer
7. Correct
8. Review (fallback)

The ordering matters: for example, Mixed Attribute is evaluated before Hallucination because it represents a distinct attribute-level failure mode. Hallucination is then used for unsupported factual content that does not satisfy the higher-priority Mixed Attribute condition. The **Review** outcome is an internal operational fallback only — it is not one of the seven final taxonomy categories, and every Review case must be resolved into one of the seven before final statistics are reported.


## 7. Final Taxonomy Categories

Once all Review cases have been resolved, every prediction falls into exactly one of the following seven categories:

| Symbol | Category | Description |
|---|---|---|
| C_correct | Correct | The prediction is semantically correct and complete relative to the ground truth. |
| C_QM | Question Misunderstanding | The prediction does not address the semantic concept the question is asking about. |
| C_HA | Hallucination | The prediction introduces unsupported factual content relative to the available ground-truth evidence. |
| C_PA | Partial Answer / Incomplete Answer | The prediction is on-topic and supported, but does not fully cover the expected content. |
| C_TR | Truncation | The prediction is cut off before completion. |
| C_REP | Repetition | The prediction contains degenerate or repeated tokens or phrases. |
| C_MA | Mixed Attribute | The prediction contains both correct and incorrect attribute-level information for the evaluated concept. |

These seven categories are mutually exclusive and are the only labels used in final reporting; the internal Review outcome described in Section 6 never appears in final statistics.

---

## 8. Dataset-Level Statistics

Beyond classifying individual predictions, the methodology defines how per-instance categories are aggregated into dataset-level statistics:

- **Per-instance classification** — each prediction tuple T is assigned one operational outcome via τ(T); any Review outcomes are resolved into one of the seven final categories before aggregation.
- **Category counts (N_r)** — the number of predictions assigned to each category C_r.
- **Category proportions (ρ_r)** — the fraction of the dataset assigned to each category, computed as N_r divided by the total number of predictions.
- **Percentage representation** — proportions expressed as percentages for reporting and comparison.
- **Partition property** — after all Review cases have been resolved, the seven final categories are mutually exclusive and exhaustive; therefore, their counts sum to the total number of evaluated predictions and their proportions sum to 1.
- **Zero-count categories** — a category with zero observed instances is a valid and expected outcome, not an error in the methodology; it simply indicates that failure mode did not occur in the evaluated dataset.
---

## 9. Evaluation Workflow

Applying the methodology to a set of predictions follows a fixed operational sequence:

1. Draw a sample from the dataset.
2. Generate the model prediction P.
3. Construct the complete prediction tuple T = (I, Q, G, P).
4. Evaluate all decision-function variables (Section 4) and quantitative indicators (Section 5) for T.
5. Apply the priority-ordered decision rules (Section 6) to obtain a category, resolving to Review if unresolved.
6. Manually resolve any Review outcomes into one of the seven final categories.
7. Record the final category for the instance.
8. Aggregate recorded categories into dataset-level statistics (Section 8).

This sequence is applied identically to every prediction in the dataset, ensuring that category assignment does not depend on the order in which predictions are processed.



## 10. Reproducibility and Operational Requirements

For the taxonomy to produce consistent, comparable results across runs and reviewers, the methodology enforces the following operational requirements:

- **Fixed evaluation definitions** — the decision functions (Section 4) are defined once and applied identically to every prediction; they are not adjusted case by case.
- **Fixed thresholds** — quantitative thresholds such as τ_complete (Section 5) are set in advance and held constant across the full dataset.
- **Fixed decision priority** — the ordering of conditions in the taxonomy algorithm (Section 6) does not change between evaluation runs.
- **Consistent application** — every prediction in the dataset is passed through the same pipeline, with no exceptions or manual shortcuts outside the defined Review process.
- **Review procedure for unresolved cases** — predictions that do not resolve cleanly under the fixed rules are routed to Review and resolved through a documented process, rather than being classified ad hoc.
- **Separation between model inference and taxonomy evaluation** — the model that generates predictions and the methodology that evaluates them are independent; the taxonomy is applied strictly after inference and does not influence how predictions are generated.

Together, these requirements ensure that re-running the methodology on the same predictions produces the same automated outcomes and, after applying the documented Review procedure consistently, the same final category assignments and dataset-level statistics.

---

## 11. Relationship to the Paper

Each methodology document corresponds to a stage of the eventual research paper's presentation:

| Methodology Document | Paper Section |
|---|---|
| Mathematical foundation | Formal notation and problem setup |
| Decision functions | Evaluation mechanisms |
| Quantitative formulation | Measurable criteria and thresholds |
| Taxonomy algorithm | Deterministic classification procedure |
| *(planned)* Final paper formulation | Condensed, paper-ready presentation of the above |

The methodology documents in this directory are intentionally more detailed and exploratory than the final paper will be; the planned final paper formulation will distill them into a compact, self-contained presentation suitable for publication.

---

## 12. Methodology Scope and Limitations

This framework is a **VQA failure-analysis taxonomy**: its purpose is to classify the *observed behavior* of model predictions against ground-truth answers, using fixed, reproducible criteria.

It is important to be explicit about what it does not claim:

- It does not assert that every failure category must occur in every dataset or model — zero-count categories are valid (Section 8).
- It does not evaluate *why* a model produces a given failure mode at a mechanistic level; it classifies observed output behavior, not internal model causes.
- Its results are only as reliable as the operational definitions and thresholds established in Sections 4–6; changing those definitions changes the resulting statistics.
- Review cases (Section 6) must be fully resolved before any final category statistics are considered valid — unresolved Review counts should never be reported as final results.


## 13. Directory Structure

```text
docs/
└── methodology/
    ├── README.md
    ├── 01_mathematical_foundation.md
    ├── 02_decision_functions.md
    ├── 03_quantitative_formulation.md
    └── 04_taxonomy_algorithm.md
```

---

## 14. Document Development Status

| Document | Status |
|---|---|
| Mathematical Foundation | ✅ Methodology version complete |
| Decision Functions | ✅ Methodology version complete |
| Quantitative Formulation | ✅ Methodology version complete |
| Taxonomy Algorithm | ✅ Methodology version complete |
| Paper Formulation | 🔲 Planned / Next |
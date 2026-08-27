# 01 — Mathematical Foundation

## 1. Purpose

This document establishes the mathematical foundation of the revised VQA failure-taxonomy framework for the Assamese Musical Instrument VLM project. It supersedes the prior version of this document. The framework represents each VQA prediction as a structured mathematical object, defines the axis-based measurement system used to characterize model behavior, and establishes the objects required by the subsequent decision-function, quantitative-formulation, and taxonomy-algorithm documents.

This revision replaces the previous category-first design (Question Misunderstanding, Hallucination, Partial Answer, Truncation, Repetition, Mixed Attribute, Correct) with an axis-first design. Category-specific variables in the prior version were found to conflate distinct behavioral properties — most notably, the prior attribute-consistency function was designed around Question Misunderstanding and consequently absorbed non-answer/refusal cases that do not reflect question misunderstanding. The revised framework measures independent behavioral axes first and derives categories from axis combinations afterward.

---

## 2. VQA Dataset

$$
\mathbb{D} = \{x_i\}_{i=1}^{N}, \qquad x_i = (I_i, Q_i, G_i)
$$

- $I_i$: input image
- $Q_i$: question
- $G_i$: ground-truth reference answer
- $N = 2016$ total VQA samples, from $N_I = 7 \times 32 = 224$ unique images across 7 instruments, each with 9 question concepts ($224 \times 9 = 2016$).

---

## 3. Image, Question, and Ground-Truth Spaces

$$
I \in \mathbb{I}, \qquad Q \in \mathbb{Q}, \qquad G \in \mathbb{G}
$$

$\mathbb{I}$, $\mathbb{Q}$, $\mathbb{G}$ denote the image, question, and ground-truth answer spaces respectively. Multiple linguistic phrasings of a question may express the same underlying semantic concept; the concept, not the wording, is the object of evaluation (Section 6).

---

## 4. Prediction Space and Model

$$
P = f_\theta(I,Q), \qquad f_\theta : \mathbb{I} \times \mathbb{Q} \rightarrow \mathbb{P}
$$

$f_\theta$ is the fine-tuned Qwen2.5-VL-3B-Instruct model (QLoRA fine-tuning), $\theta$ its learned parameters, $P \in \mathbb{P}$ the generated answer.

---

## 5. Prediction Tuple

$$
T = (I, Q, G, P) = \left(I, Q, G, f_\theta(I,Q)\right), \qquad T \in \mathbb{T}
$$

$T$ is the complete evidentiary unit for classification.

---

## 6. Semantic Concept and Required Attribute Set

$$
K = h(Q), \qquad h : \mathbb{Q} \rightarrow \mathbb{K}
$$

$K$ is the semantic concept targeted by $Q$ (e.g. material, origin, festival, traditional player, playing method, instrument type, sound, description).

Each concept has an associated **required attribute set**:

$$
A_{set}(K) \subseteq \mathbb{A}
$$

where $\mathbb{A}$ is the semantic attribute space (e.g., material, sound, festival, or playing method), while $\mathcal{V}$ denotes the corresponding space of attribute values (e.g., bamboo, soft/twangy, Rongali Bihu, or striking).

**Fixed specification for the current dataset (9 templates):**

| Template | Concept $K$ | $A_{set}(K)$ |
|---|---|---|
| q1 | festival | {festival} |
| q2 | origin | {origin} |
| q3 | material | {material} |
| q4 | parts | {parts} |
| q5 | sound | {sound} |
| q6 | traditional_player | {traditional_player} |
| q7 | playing_method | {playing_method} |
| q8 | instrument_type | {instrument_type} |
| q9 | description | {cultural_significance, role_in_assamese_music} |

q1–q8 are single-attribute templates: $|A_{set}| = 1$. q9 is a composite template testing two attributes that have no dedicated single-attribute template of their own; it is not intended to require the union of q1–q8's attributes, and its two required attributes are fixed regardless of additional contextual content that may appear in any given $G$.

**Documented dataset limitation:** For the instrument *Toka*, the authored $G$
for q9 does not contain an independently identifiable realization of the
`cultural_significance` attribute, separate from its `role_in_assamese_music`
content.

The required attribute set remains fixed as:

$$
A_{\text{set}}(q9) = \{ \text{cultural significance}, \text{role in Assamese music} \}
$$


for all seven instruments, including Toka, without exception.
This is recorded as a reference-data authoring gap, not a taxonomy design
branch. Any completeness result for Toka q9 instances must be reported
alongside this documented limitation (see `04_taxonomy_algorithm.md`,
Scope).

---

## 7. Claim Space, Decomposition, and Routing

Let $\mathbb{S}$ denote the **claim space** — the space of minimal, checkable propositions extractable from a natural-language response. An individual claim is $c \in \mathbb{S}$.

A prediction $P$ is decomposed into its constituent claims and routed according to whether each claim addresses an attribute in $A_{set}(K)$:

$$
P\rightarrow\left(P_K,P_{\bar K}\right),
\qquad
P_K,P_{\bar K}\subseteq\mathbb S,
$$

with

$$
P_K\cap P_{\bar K}=\varnothing,
\qquad
P_K\cup P_{\bar K}=\mathrm{Claims}(P).
$$

- $P_K = \{c \in \mathbb{S} : c \text{ is extracted from } P \text{ and bears on some } A \in A_{set}(K)\}$ — claims directly relevant to the required attribute(s).
- $P_{\bar K} = \{c \in \mathbb{S} : c \text{ is extracted from } P \text{ and } c \notin P_K\}$ — claims about concepts outside $A_{set}(K)$.

$P_K$ and $P_{\bar K}$ are sets of claims (elements of $\mathbb{S}$), not elements of the prediction space $\mathbb{P}$; $P$ itself remains the single generated-text object in $\mathbb{P}$ from which these claim sets are extracted.

This partition is a precondition for the decision functions defined in `02_decision_functions.md`:

$$
P_K \rightarrow \text{Axis 3 and Axis 7}, \qquad P_{\bar K} \rightarrow \text{Axis 7 only}
$$

Axis 3 (Semantic Correctness) evaluates only $P_K$, so that content addressing concepts outside the asked attribute cannot be scored as correct or incorrect relative to a $G$ that does not cover it. Axis 7 (Unsupported Content) evaluates the full response ($P_K \cup P_{\bar K}$), so that fabricated or unsupported detail is caught regardless of whether it appears inside or outside the requested attribute.

---

## 8. Attribute Extraction Functions

Because $A_{set}(K)$ may contain more than one required attribute (e.g. q9), extraction from $G$ and from $P_K$ must return a realization for **each** attribute in $A_{set}(K)$, not a single scalar value.

Define:

$$
G_A=g_G(G,A_{set}(K)),\qquad
P_A=g_P(P_K,A_{set}(K))
$$

where:

$$
g_G:\mathbb G\times2^{\mathbb A}\rightarrow
\left(A_{set}(K)\rightarrow\mathcal V\right),
$$

$$
g_P:2^{\mathbb S}\times2^{\mathbb A}\rightarrow
\left(A_{set}(K)\rightarrow\mathcal V\right).
$$

$G_A$ and $P_A$ are indexed mappings from each required attribute to its
corresponding ground-truth or predicted value:

$$
G_A:A_{set}(K)\rightarrow\mathcal V,
\qquad
P_A:A_{set}(K)\rightarrow\mathcal V.
$$

For single-attribute templates (q1–q8), $|A_{set}(K)| = 1$ and $G_A$, $P_A$ reduce to a single-element mapping, equivalent in effect to the scalar case. For q9, $G_A$ and $P_A$ each contain two entries, one per required attribute, enabling per-attribute correctness (Axis 3) and per-attribute coverage (Axis 4) to be evaluated independently before being combined.

Here, $\mathcal{V}$ denotes the space of possible values associated with
semantic attributes in $\mathbb{A}$ (e.g., `bamboo` for `material`).

---

## 9. Behavioral Axes and Measurable Signals

The framework measures eight signals organized into seven conceptual axes. Axis 2 is a single conceptual axis comprising two independent sub-signals (2a, 2b); all eight are referred to as **signals** to avoid ambiguity between the seven-axis conceptual grouping and the eight-signal measurement set.

| Axis | Signal | Domain |
|---|---|---|
| 1 | Question/Semantic Alignment | {aligned, misaligned, indeterminate} |
| 2a | Substantive Content Present | {yes, no} |
| 2b | Uncertainty/Refusal Marker | {yes, no} |
| 3 | Semantic Correctness | {correct, incorrect, mixed, not_applicable, indeterminate} |
| 4 | Completeness | {complete, partial, not_applicable, indeterminate} |
| 5 | Termination Integrity | {intact, truncated} |
| 6 | Repetition | {absent, present} |
| 7 | Unsupported Content | {none, present} |

Each signal is a function of $T$ (and, where applicable, of $G_A$, $P_A$, or the $P_K/P_{\bar K}$ partition):

$$
\mathrm{Ax}_1, \mathrm{Ax}_{2a}, \mathrm{Ax}_{2b}, \mathrm{Ax}_3, \mathrm{Ax}_4, \mathrm{Ax}_5, \mathrm{Ax}_6, \mathrm{Ax}_7 : \mathbb{T} \rightarrow (\text{respective domain})
$$

Full measurement definitions, preconditions, and dependency order are given in `02_decision_functions.md`. This document fixes only the domains and their role in the overall structure.

---

## 10. Failure-Category Space

$$
\mathbb{C} = \{C_{NA}, C_{QM}, C_{IC}, C_{HA}, C_{correct}, C_{PA}, C_{MA}\}
$$

- $C_{NA}$ — Non-Answer / Abstention
- $C_{QM}$ — Question Misunderstanding
- $C_{IC}$ — Incoherent Response
- $C_{HA}$ — Hallucination (redefined; see Section 12)
- $C_{correct}$ — Correct
- $C_{PA}$ — Partial Answer
- $C_{MA}$ — Mixed Attribute

$\mathbb{C}$ contains exactly seven **core categories**. Under Design A, final classification output is not a bare core category but a core category paired with independent structural/content flags:

$$
\text{Final classification} = \left(C_r,\ \mathrm{Ax}_5,\ \mathrm{Ax}_6,\ \mathrm{Ax}_7\right), \quad C_r \in \mathbb{C}
$$

Truncation (Axis 5 = truncated), Repetition (Axis 6 = present), and Unsupported Content (Axis 7 = present) are **not** standalone final categories; they are cross-cutting flags attached to whichever core category is assigned. This is a deliberate structural departure from the prior flat taxonomy, in which Truncation and Repetition were themselves top-priority categories (see `04_taxonomy_algorithm.md` for the retired priority order and its replacement).

---

## 11. Taxonomy Function

$$
\tau(T)=\left(C_r,\mathrm{Ax}_5(T),\mathrm{Ax}_6(T),\mathrm{Ax}_7(T)\right)
$$

where

$$
C_r \in \mathbb{C},
$$

$$
\mathrm{Ax}_5(T)\in\{\text{intact},\text{truncated}\},
$$

$$
\mathrm{Ax}_6(T)\in\{\text{absent},\text{present}\},
$$

and

$$
\mathrm{Ax}_7(T)\in\{\text{none},\text{present}\}.
$$

$$
\tau(T) = (C_r,\ \mathrm{Ax}_5(T),\ \mathrm{Ax}_6(T),\ \mathrm{Ax}_7(T))
$$

The core-category component $C_r$ is determined by Axes 1–4 (and the 2a/2b sub-signals); the flag components are determined independently by Axes 5–7. The full decision procedure, including precondition branches, is defined in `02_decision_functions.md` and `04_taxonomy_algorithm.md`.

---

## 12. Retirement of the Prior Hallucination Indicator

The prior formulation defined a hallucination indicator $H(G,P) \in \{0,1\}$ denoting the presence of unsupported factual content, independent of topical alignment. **This indicator is retired.** Its function is fully subsumed by Axis 7 (Unsupported Content), which uses a strict $G$-only support standard (Section 13) and applies across the whole response, not only to claims relevant to $K$.

In the revised taxonomy, the term **Hallucination** ($C_{HA}$) denotes a distinct, narrower concept: a response that is topically aligned ($\mathrm{Ax}_1 = \text{aligned}$) but whose $P_K$ claims are semantically incorrect relative to $G$ ($\mathrm{Ax}_3 = \text{incorrect}$) — i.e., wrong facts on the right topic. This is a deliberate terminological redefinition relative to the prior document set and must not be conflated with the retired $H$ indicator or with the Axis 7 flag.

---

## 13. Unsupported Content — G-Only Support Standard

Axis 7 evaluates every claim in $P$ (both $P_K$ and $P_{\bar K}$) against $G$ under a strict support rule:

- A claim $c \in \mathbb{S}$ is **supported** iff $c$ is explicitly stated in $G$, or directly logically entailed by $G$.
- A claim $c$ is **unsupported** iff neither condition holds.

Axis 7 measures traceability to $G$, not general-world truth: a claim may be factually true and still classified as unsupported if it is not stated or directly entailed by $G$. "Reasonable extension" or external world knowledge is explicitly excluded as a basis for support. This standard is unchanged from the originally locked Axis 7 definition and applies uniformly regardless of core category.

---

## 14. Core Mathematical Objects

| Object | Symbol | Space | Role |
|---|---|---|---|
| Dataset | $\mathbb{D}$ | — | Complete VQA dataset |
| Image | $I$ | $\mathbb{I}$ | Visual input |
| Question | $Q$ | $\mathbb{Q}$ | Question instance |
| Ground truth | $G$ | $\mathbb{G}$ | Reference answer |
| Prediction | $P$ | $\mathbb{P}$ | Model-generated answer |
| Claim space | $\mathbb{S}$ | — | Space of minimal checkable claims |
| On-topic claims | $P_K$ | $\subseteq \mathbb{S}$ | Claims relevant to $A_{set}(K)$ |
| Off-topic claims | $P_{\bar K}$ | $\subseteq \mathbb{S}$ | Claims outside $A_{set}(K)$ |
| Prediction tuple | $T$ | $\mathbb{T}$ | $(I,Q,G,P)$ |
| Concept | $K$ | $\mathbb{K}$ | Semantic concept targeted by $Q$ |
| Required attribute set | $A_{set}(K)$ | $\subseteq \mathbb{A}$ | Fixed set of required attributes for the template |
| Attribute | $A$ | $\mathbb{A}$ | Semantic attribute |
| Attribute value | $V$ | $\mathcal{V}$ | Value associated with an attribute |
| Ground-truth attribute mapping | $G_A$ | $A_{set}(K)\rightarrow\mathcal{V}$ | Maps each required attribute to its ground-truth value |
| Predicted attribute mapping | $P_A$ | $A_{set}(K)\rightarrow\mathcal{V}$ | Maps each required attribute to its predicted value |
| Category | $C_r$ | $\mathbb{C}$ | One of seven core categories |
| VQA model | $f_\theta$ | $\mathbb{I}\times\mathbb{Q}\rightarrow\mathbb{P}$ | Fine-tuned Qwen2.5-VL-3B-Instruct |
| Concept function | $h$ | $\mathbb{Q}\rightarrow\mathbb{K}$ | Maps a question to its semantic concept |
| Taxonomy function | $\tau$ | $\mathbb{T}\rightarrow\mathbb{C}\times\{\text{intact},\text{truncated}\}\times\{\text{absent},\text{present}\}\times\{\text{none},\text{present}\}$ | Maps a prediction tuple to a core category and three independent flags |

---

## 15. Central Formulation

$$
T = (I,Q,G,f_\theta(I,Q))
$$
$$
K = h(Q), \qquad A_{set}(K) \text{ fixed per template}
$$
$$
P\rightarrow(P_K,P_{\bar K}),
\qquad
P_K,P_{\bar K}\subseteq\mathbb{S},
$$
$$
G_A = g_G(G, A_{set}(K)), \qquad P_A = g_P(P_K, A_{set}(K))
$$
$$
\tau(T) = (C_r,\ \mathrm{Ax}_5,\ \mathrm{Ax}_6,\ \mathrm{Ax}_7)
$$

These definitions establish the mathematical vocabulary carried forward into `02_decision_functions.md`, which defines the precise measurement procedure and precondition structure for each signal.
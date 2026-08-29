# 02 — Decision Functions

## 1. Purpose

This document defines the operational measurement rules for the eight signals introduced in `01_mathematical_foundation.md`: preconditions, dependency order, and precondition branches over $\mathbb{T}$. It does not assign final categories — category derivation from signal combinations is `03_quantitative_formulation.md`; the ordered decision procedure is `04_taxonomy_algorithm.md`.

All notation follows `01` exactly: $T=(I,Q,G,P)$, $K=h(Q)$, $A_{\text{set}}(K)$, $P \rightarrow (P_K, P_{\bar K})$, $G_A, P_A : A_{\text{set}}(K) \rightarrow \mathcal{V}^\bot$.

---

## 2. Measurement Order

$$
\mathrm{Ax}_{2a} \;\rightarrow\; \mathrm{Ax}_1 \;\rightarrow\; \mathrm{Ax}_3 \;\rightarrow\; \mathrm{Ax}_4
$$

$\mathrm{Ax}_{2b}, \mathrm{Ax}_5, \mathrm{Ax}_6, \mathrm{Ax}_7$ are measured independently of this chain, subject to the forced values in Section 9 and Section 10.

---

## 3. Axis 2a — Substantive Content Present

$$
\mathrm{Ax}_{2a}(T) = \text{yes} \iff \mathrm{Claims}(P) \neq \emptyset
$$

else $\mathrm{Ax}_{2a}(T) = \text{no}$. Measured first; gates $\mathrm{Ax}_1$.

---

## 4. Axis 2b — Uncertainty/Refusal Marker

$$
\mathrm{Ax}_{2b}(T) = \text{yes} \iff P \text{ contains explicit hedging, refusal, or uncertainty language}
$$

Independent of $\mathrm{Ax}_{2a}$; both combinations are valid (e.g. "I'm not sure, but it's bamboo" vs. "I don't know").

---

## 5. Axis 1 — Question/Semantic Alignment

**Precondition:** $\mathrm{Ax}_{2a}(T) = \text{no} \implies \mathrm{Ax}_1(T) = \text{indeterminate}$.

**When $\mathrm{Ax}_{2a}(T) = \text{yes}$:**

$$
\mathrm{Ax}_1(T) =
\begin{cases}
\text{aligned}, & P_K \neq \emptyset \\
\text{misaligned}, & P_K = \emptyset \text{ and an identifiable concept } K' \neq K \text{ is targeted} \\
\text{indeterminate}, & P_K = \emptyset \text{ and no identifiable concept can be assigned}
\end{cases}
$$

Alignment requires only $P_K \neq \emptyset$; a response may be `aligned` while also containing $P_{\bar K}$ claims.

---

## 6. Axis 3 — Semantic Correctness

**Preconditions:**

$$
\mathrm{Ax}_1(T) = \text{misaligned} \implies \mathrm{Ax}_3(T) = \text{not applicable}
$$

$$
\mathrm{Ax}_1(T) = \text{indeterminate} \text{ and } \mathrm{Ax}_{2a}(T) = \text{yes} \implies \mathrm{Ax}_3(T) = \text{indeterminate}
$$

$$
\mathrm{Ax}_{2a}(T) = \text{no} \implies \mathrm{Ax}_3(T) = \text{not applicable}
$$

**When applicable**, evaluated over $P_A$ against $G_A$ per attribute $A_j \in A_{\text{set}}(K)$. An attribute is **comparable** iff $G_A(A_j) \neq \bot$ and $P_A(A_j) \neq \bot$; only comparable attributes contribute evidence. The no-comparable-evidence case is checked first to avoid a vacuous-truth condition.

$$
\mathrm{Ax}_3(T) =
\begin{cases}
\text{indeterminate}, & \text{no } A_j \text{ is comparable} \\
\text{mixed}, & \text{at least one comparable attribute is consistent and at least one is contradictory} \\
\text{incorrect}, & \text{all comparable attributes are contradictory} \\
\text{correct}, & \text{all comparable attributes are consistent} \\
\text{indeterminate}, & \text{at least one comparable claim cannot be reliably verified}
\end{cases}
$$

$P_A(A_j) = \bot$ (missing prediction value) makes that attribute non-comparable — it is not treated as correct or incorrect, and is evaluated only by Axis 4, as missing coverage. For q1–q8, $|A_{\text{set}}(K)|=1$ and this collapses to a single comparison. Correctness measures consistency with $G$; it is independent of Axis 7.

---

## 7. Axis 4 — Completeness

**Precondition:**

$$
\mathrm{Ax}_3(T) \in \{\text{not applicable}, \text{incorrect}\} \implies \mathrm{Ax}_4(T) = \text{not applicable}
$$

**When $\mathrm{Ax}_3(T) \in \{\text{correct}, \text{mixed}\}$:** let $n = |A_{\text{set}}(K)|$, and let $m$ be the number of $A_j \in A_{\text{set}}(K)$ with $G_A(A_j) \neq \bot$ for which $P_A(A_j)$ correctly covers $G_A(A_j)$. The indeterminate case is checked first, since $m$ is only well-defined once coverage is reliably determined for every required attribute.

$$
\mathrm{Ax}_4(T) =
\begin{cases}
\text{indeterminate}, & \text{coverage of at least one required attribute cannot be reliably determined} \\
\text{complete}, & m = n \\
\text{partial}, & m < n \text{ (including } m=0\text{)}
\end{cases}
$$

An attribute $A_j$ with $G_A(A_j) = \bot$ can never contribute to $m$, since no reference realization exists for $P_A(A_j)$ to be checked against — but it is **not removed from $n$**. For q1–q8, $n=1$ and a single present fact yields `complete`; absence yields `partial`.

**Documented dataset limitation (Toka, q9):** the required attribute set remains

$$
A_{\text{set}}(q9) = \{\text{cultural significance},\ \text{role in Assamese music}\}
$$

for every instrument, including Toka. For Toka:

`G_A(cultural_significance) = ⊥`

because the authored reference answer does not independently realize that attribute. The required-attribute count nevertheless remains $n=2$; $\bot$ does not remove an attribute from the required set or create an instrument-specific completeness rule. Since this attribute can never be covered, Toka's q9 completeness is structurally capped at `partial`. Any Toka q9 completeness result must explicitly disclose this reference-data limitation.

---

## 8. Axis 5 — Termination Integrity

$$
\mathrm{Ax}_5(T) =
\begin{cases}
\text{intact}, & \text{stop reason} = \text{EOS} \\
\text{truncated}, & \text{stop reason} = \text{max length / forced cutoff}
\end{cases}
$$

Secondary evidence permitted only if stop reason is unavailable; grammatical incompleteness alone is not a primary detector. Measured independently of all other axes.

---

## 9. Axis 6 — Repetition

$$
\mathrm{Ax}_6(T) = \text{present} \iff P \text{ contains redundant/degenerate repetition without new information}
$$

**Forced value:** $\mathrm{Ax}_{2a}(T) = \text{no} \implies \mathrm{Ax}_6(T) = \text{absent}$. Otherwise independent of Axes 1, 3, 4 — repetition can co-occur with any $\mathrm{Ax}_3$ value, including `indeterminate`.

---

## 10. Axis 7 — Unsupported Content

$$
\mathrm{Ax}_7(T) = \text{present} \iff \exists\, c \in (P_K \cup P_{\bar K}) : c \text{ is unsupported by } G
$$

using the strict $G$-only support standard (`01` Section 13): $c$ is supported iff explicitly stated in or directly entailed by $G$; unsupported otherwise. World knowledge and "reasonable extension" are excluded.

Evaluated over the **entire response**, per the routing in `01` Section 7 ($P_K \rightarrow$ Axis 3 and 7; $P_{\bar K} \rightarrow$ Axis 7 only). Evaluated even when $\mathrm{Ax}_1 = \text{misaligned}$ or $\mathrm{Ax}_3 = \text{indeterminate}$.

**Forced value:** $\mathrm{Ax}_{2a}(T) = \text{no} \implies \mathrm{Ax}_7(T) = \text{none}$.

---

## 11. Precondition Branches

| Branch | Condition | Axis 3 | Axis 4 | Axis 6 | Axis 7 | Other free signals |
|---|---|---|---|---|---|---|
| A | Axis 2a = no | not applicable | not applicable | absent (forced) | none (forced) | Axis 2b, Axis 5 |
| B | Axis 1 = misaligned | not applicable | not applicable | free | evaluated | Axis 2b, Axis 5, Axis 6 |
| G | Axis 1 = indeterminate, Axis 2a = yes | indeterminate | indeterminate | evaluated | evaluated | — |
| D | Axis 1 = aligned, Axis 3 = incorrect | (incorrect) | not applicable | free | evaluated | — |
| E | Axis 1 = aligned, Axis 3 = correct | (correct) | complete or partial | free | evaluated | — |
| F | Axis 1 = aligned, Axis 3 = mixed | (mixed) | complete or partial | free | evaluated | — |

Additional notes:

- Branch A: Axis 1 is forced to `indeterminate` (per Axis 1's own precondition in Section 5).
- Axis 5 (Termination Integrity) is free in every branch — it is never forced or excluded, per Section 8.
- Axis 6 and Axis 7, except where explicitly forced in Branch A, cut across all branches and are not branch-defining.

---

## 12. Decision-Function Requirements

1. **Explicitness** — every condition defined in terms of $T$, $K$, $A_{\text{set}}(K)$, $G_A$, $P_A$, or $(P_K,P_{\bar K})$.
2. **Consistency** — identical conditions applied to every prediction, including $\bot$-handling.
3. **Semantic grounding** — Axis 3/4 depend on $G_A$-relative comparison, not surface matching.
4. **Independence where specified** — Axes 5, 6, 7 measured without reference to Axes 1, 3, 4, except the Branch A forced values.
5. **Reproducibility** — identical $T$ and evidence yield identical signal values.

---

## 13. Transition to Quantitative Formulation

`03_quantitative_formulation.md` maps the branches and signal combinations defined here to the seven core categories in $\mathbb{C}$, and specifies how the Axis 5/6/7 flags attach to each core-category assignment under Design A.
# 04 — Taxonomy Algorithm

## 1. Purpose

This document defines the ordered, operational procedure for classifying a single prediction tuple $T = (I,Q,G,P)$ using the signals defined in `02_decision_functions.md` and the category derivation defined in `03_quantitative_formulation.md`. It also defines the manual-review resolution procedure for the internal $C_{review}$ outcome, and the scope and reporting requirements for the resulting dataset-level statistics.

This document assumes `01`, `02`, and `03` as settled. No new axes, categories, or notation are introduced here.

---

## 2. Inputs

For a given prediction tuple $T = (I,Q,G,P)$:

- $K = h(Q)$ and $A_{\text{set}}(K)$ are looked up from the fixed template table (`01` Section 6).
- $P$ is decomposed into $(P_K, P_{\bar K})$ (`01` Section 7).
- $G_A = g_G(G, A_{\text{set}}(K))$ and $P_A = g_P(P_K, A_{\text{set}}(K))$ are computed (`01` Section 8).
- stop_reason is retrieved for Axis 5.

These four inputs are prerequisites for every subsequent step and must be computed before any signal is measured.

---

## 3. Step 1 — Measure Axis 2a and Axis 2b

Compute $`\mathrm{Ax}_{2a}(T)`$ and $`\mathrm{Ax}_{2b}(T)`$ per 02 Sections 3–4.

$`\mathrm{Ax}_{2a}`$ gates Step 2; $`\mathrm{Ax}_{2b}`$ is recorded but does not affect category derivation.

---

## 4. Step 2 — Measure Axis 1

Compute $\mathrm{Ax}_1(T)$ per `02` Section 5, using the Step 1 result as precondition input.

If $\mathrm{Ax}_{2a}(T) = \text{no}$, $\mathrm{Ax}_1(T)$ is forced to `indeterminate` and Steps 3–4 are skipped entirely (Branch A applies; proceed to Step 6).

---

## 5. Step 3 — Measure Axis 3

Compute $\mathrm{Ax}_3(T)$ per `02` Section 6, using $G_A$, $P_A$, and the Step 2 result.

If $\mathrm{Ax}_1(T) = \text{misaligned}$, $\mathrm{Ax}_3(T)$ is forced to `not applicable` and Step 4 is skipped (Branch B applies; proceed to Step 6).

If $`\mathrm{Ax}_1(T) = \text{indeterminate}`$ while $`\mathrm{Ax}_{2a}(T) = \text{yes}`$, $`\mathrm{Ax}_3(T)`$ is forced to `indeterminate` (Branch G applies; proceed to Step 6, then Step 7 routes to $`C_{review}`$).

---

## 6. Step 4 — Measure Axis 4

Compute $\mathrm{Ax}_4(T)$ per `02` Section 7, using the Step 3 result.

This step only executes when $\mathrm{Ax}_3(T) \in \{\text{correct}, \text{mixed}\}$ (Branch E or F). When $\mathrm{Ax}_3(T) = \text{incorrect}$ (Branch D), $\mathrm{Ax}_4(T)$ is forced to `not applicable`.

---

## 7. Step 5 — Identify Branch or Review State

Using the results of Steps 1–4, identify the applicable semantic branch defined in `02_decision_functions.md`.

If the state is unresolved because $`\mathrm{Ax}_3(T) = \text{indeterminate}`$ or $`\mathrm{Ax}_4(T) = \text{indeterminate}`$, mark the instance for $`C_{review}`$ rather than forcing it into Branch D, E, or F.

Otherwise, identify exactly one of Branches A, B, G, D, E, or F.

---

## 8. Step 6 — Measure Axis 5, Axis 6, Axis 7

Compute $\mathrm{Ax}_5(T)$, $\mathrm{Ax}_6(T)$, and $\mathrm{Ax}_7(T)$ per `02` Sections 8–10.

These are computed regardless of which branch was identified in Step 5, subject to the forced values in Branch A ($\mathrm{Ax}_6 = \text{absent}$, $\mathrm{Ax}_7 = \text{none}$).

---

## 9. Step 7 — Derive Core Category

Apply $\delta(T)$ as defined in `03` Section 5, using the branch identified in Step 5 and, where applicable, the Axis 4 value from Step 4.

$$
\delta(T) \in \mathbb{C} \cup \{C_{review}\}
$$

If $\delta(T) = C_{review}$, proceed to Step 8 (Review Resolution) before finalizing. Otherwise, proceed directly to Step 9.

---

## 10. Step 8 — Review Resolution

$C_{review}$ arises only from the two cases in `03` Section 4: Axis 3 = indeterminate, or Axis 4 = indeterminate. Both indicate that the automated signal measurement could not reliably resolve a required value.

**Resolution procedure:**

1. A human reviewer re-examines $T$ directly, applying the same Axis 3 / Axis 4 definitions from `02` Sections 6–7 manually.
2. The reviewer assigns the attribute-level comparability and correctness judgments that the automated procedure could not determine.
3. Once the unresolved comparison is resolved, recompute the affected signal(s) and any downstream dependent signal(s) according to `02` Sections 6–7.
4. Step 7 is re-applied with the updated value, yielding a final $\delta(T) \in \mathbb{C}$.

$C_{review}$ is never included in final category statistics (`03` Section 9); every instance must be resolved to one of the seven core categories before reporting.

---

## 11. Step 9 — Final Classification

$$
\tau(T) = \left(\delta(T),\ \mathrm{Ax}_5(T),\ \mathrm{Ax}_6(T),\ \mathrm{Ax}_7(T)\right)
$$

with $\delta(T) \in \mathbb{C}$ (post-resolution). This matches `01` Section 11 and `03` Section 6.

---

## 12. Complete Procedure Summary

| Step | Action | Depends on |
|---|---|---|
| 1 | Measure Axis 2a, 2b | $T$ |
| 2 | Measure Axis 1 | Step 1 |
| 3 | Measure Axis 3 | Step 2, $G_A$, $P_A$ |
| 4 | Measure Axis 4 | Step 3 |
| 5 | Identify branch | Steps 1–4 |
| 6 | Measure Axis 5, 6, 7 | $T$, Step 1 (forced values only) |
| 7 | Derive $\delta(T)$ | Step 5, Step 4 |
| 8 | Resolve review (if needed) | Step 7 |
| 9 | Assemble $\tau(T)$ | Steps 7/8, Step 6 |

Steps 1–7 are fully deterministic given the same $T$ and the same claim-extraction/entailment judgments; Step 8 is the only step requiring human input, and only for the subset of predictions where automated measurement is genuinely insufficient.

---

## 13. Mutual Exclusivity and Determinism

By `03` Section 7, the branch identified in Step 5 is unique for every $T$. The automated portion of the procedure is deterministic given the same $T$, claim extraction, entailment judgments, and generation metadata. Cases requiring $C_{review}$ are resolved using the documented review procedure (Step 8); final reproducibility additionally depends on consistent reviewer application of the specified criteria.

---

## 14. Coverage

By `03` Section 8, every $T \in \mathbb{T}$ is assigned a branch in Step 5 and, after Step 8 where needed, a core category in Step 7. No prediction tuple exits the procedure without a final $\tau(T)$.

---

## 15. Scope

This algorithm defines the per-instance classification procedure. It does not define:

- claim-extraction implementation (how $\mathrm{Claims}(P)$, $P_K$, $P_{\bar K}$ are computed from raw text),
- the entailment-checking procedure used for Axis 3 and Axis 7 comparisons,
- inter-annotator agreement or reviewer-calibration procedures for Step 8.

These are implementation and annotation-protocol concerns, addressed separately from the mathematical taxonomy.

**Documented dataset limitation carried forward:** for Toka, q9: `G_A(cultural_significance) = ⊥` (per `01` Section 6, `02` Section 7). This algorithm applies Steps 1–9 identically to Toka q9 instances as to all others; no branch or step is instrument-specific. The resulting structural cap on Axis 4 (`03` Section 10) is a property of the input data, not of this algorithm, and must be disclosed wherever Toka q9 results are reported.

A category may have zero observed instances ($N_r = 0$) in any given dataset without violating coverage or exclusivity, consistent with `01` Section 10.
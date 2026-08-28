# 03 — Quantitative Formulation

## 1. Purpose

This document derives the seven core categories in $\mathbb{C}$ from the branch structure and signal values defined in `02_decision_functions.md`. It also specifies how the three cross-cutting flags (Axis 5, Axis 6, Axis 7) attach to a core-category assignment under Design A.

This document does not define the ordered execution procedure used to process a dataset — that is `04_taxonomy_algorithm.md`. Here, category derivation is expressed as direct mappings from precondition branches (Section 11 of `02`) and Axis 3/4 values to a unique $C_r \in \mathbb{C}$.

---

## 2. Category Space (recap)

$$
\mathbb{C} = \{C_{NA}, C_{QM}, C_{IC}, C_{HA}, C_{correct}, C_{PA}, C_{MA}\}
$$

Definitions are unchanged from `01` Section 10 and Section 12. $C_{HA}$ (Hallucination) denotes topically-aligned, semantically-incorrect content (Branch D), not the retired unsupported-content indicator.

---

## 3. Category Derivation from Branches

Each of the six branches defined in `02` Section 11 maps to exactly one core category. The mapping is direct — no branch maps to more than one category, and no branch is left unmapped.

| Branch | Defining condition | Core category | Rationale |
|---|---|---|---|
| A | Axis 2a = no | $C_{NA}$ — Non-Answer / Abstention | No substantive content; response is a refusal or empty answer regardless of hedging language |
| B | Axis 1 = misaligned | $C_{QM}$ — Question Misunderstanding | Response addresses an identifiable concept other than the one asked |
| G | Axis 1 = indeterminate, Axis 2a = yes | $C_{IC}$ — Incoherent Response | Substantive content present, but no identifiable concept can be assigned to it |
| D | Axis 1 = aligned, Axis 3 = incorrect | $C_{HA}$ — Hallucination | Right topic, comparable claims all contradict $G$ |
| E, Axis 4 = complete | Axis 1 = aligned, Axis 3 = correct, Axis 4 = complete | $C_{correct}$ — Correct | All comparable claims consistent with $G$; full required coverage |
| E, Axis 4 = partial | Axis 1 = aligned, Axis 3 = correct, Axis 4 = partial | $C_{PA}$ — Partial Answer | All comparable claims consistent with $G$; incomplete coverage |
| F | Axis 1 = aligned, Axis 3 = mixed | $C_{MA}$ — Mixed Attribute | Right topic, some comparable claims consistent and some contradict $G$ |

Branch E splits into two categories depending on Axis 4; this is the only branch that does not map 1:1 to a single category, since completeness is the distinguishing factor between Correct and Partial Answer.

---

# 03 — Quantitative Formulation

## 1. Purpose

This document derives the seven core categories in $\mathbb{C}$ from the branch structure and signal values defined in `02_decision_functions.md`. It also specifies how the three cross-cutting flags (Axis 5, Axis 6, Axis 7) attach to a core-category assignment under Design A.

This document does not define the ordered execution procedure used to process a dataset — that is `04_taxonomy_algorithm.md`. Here, category derivation is expressed as direct mappings from precondition branches (Section 11 of `02`) and Axis 3/4 values to a unique $C_r \in \mathbb{C}$.

---

## 2. Category Space (recap)

$$
\mathbb{C} = \{C_{NA}, C_{QM}, C_{IC}, C_{HA}, C_{correct}, C_{PA}, C_{MA}\}
$$

Definitions are unchanged from `01` Section 10 and Section 12. $C_{HA}$ (Hallucination) denotes topically-aligned, semantically-incorrect content (Branch D), not the retired unsupported-content indicator.

---

## 3. Category Derivation from Branches

Each of the six branches defined in `02` Section 11 maps to exactly one core category. The mapping is direct — no branch maps to more than one category, and no branch is left unmapped.

| Branch | Defining condition | Core category | Rationale |
|---|---|---|---|
| A | Axis 2a = no | $C_{NA}$ — Non-Answer / Abstention | No substantive content; response is a refusal or empty answer regardless of hedging language |
| B | Axis 1 = misaligned | $C_{QM}$ — Question Misunderstanding | Response addresses an identifiable concept other than the one asked |
| G | Axis 1 = indeterminate, Axis 2a = yes | $C_{IC}$ — Incoherent Response | Substantive content present, but no identifiable concept can be assigned to it |
| D | Axis 1 = aligned, Axis 3 = incorrect | $C_{HA}$ — Hallucination | Right topic, comparable claims all contradict $G$ |
| E, Axis 4 = complete | Axis 1 = aligned, Axis 3 = correct, Axis 4 = complete | $C_{correct}$ — Correct | All comparable claims consistent with $G$; full required coverage |
| E, Axis 4 = partial | Axis 1 = aligned, Axis 3 = correct, Axis 4 = partial | $C_{PA}$ — Partial Answer | All comparable claims consistent with $G$; incomplete coverage |
| F | Axis 1 = aligned, Axis 3 = mixed | $C_{MA}$ — Mixed Attribute | Right topic, some comparable claims consistent and some contradict $G$ |

Branch E splits into two categories depending on Axis 4; this is the only branch that does not map 1:1 to a single category, since completeness is the distinguishing factor between Correct and Partial Answer.

---

## 4. Cases Excluded from Direct Mapping

Two signal states are not covered by Section 3 and require explicit disposition:

**Axis 3 = indeterminate:** when Axis 1 = `aligned` and Axis 3 cannot be resolved to `correct`, `incorrect`, or `mixed`, no core category in $\mathbb{C}$ is assigned. This case is routed to the internal review outcome $C_{review}$ defined in `04_taxonomy_algorithm.md`.

**Axis 4 = indeterminate** (within Branch E/F, i.e. Axis 3 is correct or mixed but per-attribute coverage cannot be reliably determined): not covered by Section 3's Branch E/F mapping, since that mapping requires Axis 4 $\in \{\text{complete}, \text{partial}\}$. This case is also routed to the internal review outcome.

Both cases preserve determinism: rather than forcing an uncertain signal state into one of the seven categories, the taxonomy defers to manual review, consistent with the "review is resolved before final statistics" principle carried over from the retired prior taxonomy.

---

## 5. Formal Category-Derivation Function

Let $\delta : \mathbb{T} \rightarrow \mathbb{C} \cup \{C_{review}\}$ denote the core-category derivation function.

$$
\delta(T) =
\begin{cases}
C_{NA}, & \text{Branch A} \\
C_{QM}, & \text{Branch B} \\
C_{IC}, & \text{Branch G} \\
C_{HA}, & \text{Branch D} \\
C_{correct}, & \text{Branch E and } \mathrm{Ax}_4(T) = \text{complete} \\
C_{PA}, & \text{Branch E and } \mathrm{Ax}_4(T) = \text{partial} \\
C_{MA}, & \text{Branch F} \\
C_{review}, & \text{Axis 3 or Axis 4 indeterminate within Branch D, E, or F's applicable region}
\end{cases}
$$

$C_{review}$ is not a member of $\mathbb{C}$; it is an internal outcome requiring manual resolution before category statistics are computed, consistent with `04`.

---

## 6. Flag Attachment (Design A)

Under Design A, final classification pairs $\delta(T)$ with the three cross-cutting flags:

$$
\tau(T) = \left(\delta(T),\ \mathrm{Ax}_5(T),\ \mathrm{Ax}_6(T),\ \mathrm{Ax}_7(T)\right)
$$

This matches the taxonomy function stated in `01` Section 11. Flags are computed independently of $\delta(T)$ (per `02` Sections 8–10). Flags are never used to select or override the core category. During internal review, the flags may be retained alongside the provisional $C_{review}$ outcome. After manual resolution, the final classification contains one of the seven core categories together with the three independent flags.
**Example final labels** (for illustration only, not new categories):

| Core category | Axis 5 | Axis 6 | Axis 7 | Reported as |
|---|---|---|---|---|
| $C_{correct}$ | truncated | absent | none | Correct + truncated |
| $C_{PA}$ | truncated | absent | present | Partial Answer + truncated + unsupported |
| $C_{correct}$ | intact | present | none | Correct + repetitive |
| $C_{NA}$ | intact | absent | none | Non-Answer / Abstention |

---

## 7. Mutual Exclusivity

For every $T \in \mathbb{T}$, the branch conditions in `02` Section 11 (Branches A, B, G, D, E, F) are pairwise disjoint and jointly determined by the ordered preconditions on Axis 2a, Axis 1, and Axis 3 — no prediction tuple satisfies more than one branch's defining condition. Within Branch E, the further split by Axis 4 (complete vs. partial) is also disjoint, since Axis 4 takes exactly one value per $T$.

Consequently:

$$
\forall T \in \mathbb{T}: \left|\{C_r \in \mathbb{C} \cup \{C_{review}\} : \delta(T) = C_r\}\right| = 1
$$

Each prediction receives exactly one core-category outcome. This holds independently of the flag values (Axis 5, 6, 7), which are computed separately and do not affect $\delta(T)$.

---

## 8. Coverage

Every prediction satisfies exactly one of the following semantic outcomes: Branch A, Branch B, Branch G, Branch D, Branch E, Branch F, or an explicit $C_{review}$ outcome for an unresolved indeterminate state. Branch A covers all cases with no substantive content. Branches B and G cover the two non-aligned substantive-content cases. Branches D, E, and F cover aligned responses with `incorrect`, `correct`, and `mixed` semantic correctness, respectively. Any remaining indeterminate semantic or completeness state is routed to $C_{review}$ rather than silently assigned to a core category.

Together, Sections 3–5 assign a defined outcome — one of seven categories, or review — to every $T \in \mathbb{T}$; no prediction tuple is left unclassified.

---

## 9. Quantitative Category Statistics

For a dataset of $N$ evaluated prediction tuples, after all $C_{review}$ instances have been manually resolved into one of the seven categories:

$$
\mathbf{1}_r(T_i) =
\begin{cases}
1, & \delta(T_i) = C_r \\
0, & \text{otherwise}
\end{cases}
$$

$$
N_r = \sum_{i=1}^{N} \mathbf{1}_r(T_i), \qquad \rho_r = \frac{N_r}{N}
$$

$$
\sum_r N_r = N, \qquad \sum_r \rho_r = 1
$$

Flag prevalence is reported separately and independently of category counts, since flags are not mutually exclusive with each other or with category membership:

$$
N_5^{truncated} = \sum_{i=1}^N \mathbf{1}[\mathrm{Ax}_5(T_i) = \text{truncated}], \quad \text{similarly for } N_6^{present}, N_7^{present}
$$

These may be reported overall or cross-tabulated against $C_r$ (e.g. proportion of $C_{correct}$ instances that are also truncated).

---

## 10. Toka Q9 Reporting Requirement

Per `02` Section 7, any $T$ with instrument = Toka and $K = $ description (q9) has $G_A(\text{cultural significance}) = \bot$, structurally capping $\mathrm{Ax}_4$ at `partial`. For Toka q9, the documented reference-data limitation means that $C_{correct}$ cannot be reached through the `complete` branch under the current reference annotation. This is a consequence of the reference data, not an instrument-specific taxonomy rule. Aggregate or per-instrument completeness statistics (Section 9) must disclose this limitation when Toka q9 instances are included, per `01` Section 6.

---

## 11. Scope

This document defines category derivation and flag attachment. It does not define:

- the ordered execution sequence for processing raw predictions (`04_taxonomy_algorithm.md`),
- the manual-review resolution procedure for $C_{review}$ cases (`04_taxonomy_algorithm.md`),
- claim-extraction or entailment-checking implementation details (out of scope for this document set; an implementation/annotation guide).

A category may have $N_r = 0$ in a given evaluated dataset without invalidating the taxonomy's completeness or exclusivity properties, consistent with `01` Section 10.S
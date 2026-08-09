# Quantitative Formulation

## 1. Purpose

This document formalizes the evaluation variables and decision conditions used to assign each VQA prediction to one of the seven defined categories in the proposed failure taxonomy.

The formulation operates on the complete prediction tuple:

$$
T = (I, Q, G, P)
$$

where:

- $I$ is the input image.
- $Q$ is the question.
- $G$ is the ground-truth answer.
- $P$ is the model prediction.

The taxonomy categories are:

$$
C = \{C_{\text{correct}}, C_{\text{QM}}, C_{\text{HA}}, C_{\text{PA}}, C_{\text{TR}}, C_{\text{REP}}, C_{\text{MA}}\}
$$

where:

- $C_{\text{correct}}$: Correct
- $C_{\text{QM}}$: Question Misunderstanding
- $C_{\text{HA}}$: Hallucination
- $C_{\text{PA}}$: Partial Answer / Incomplete Answer
- $C_{\text{TR}}$: Truncation
- $C_{\text{REP}}$: Repetition
- $C_{\text{MA}}$: Mixed Attribute

---

## 2. Semantic Evaluation Variables

Exact string equality is insufficient for the proposed taxonomy because semantically equivalent answers may use different wording.

Therefore, define the following binary indicators:

$$
m_G(G, P) \in \{0,1\}
$$

where $m_G = 1$ indicates that the prediction is semantically consistent with the ground-truth answer.

Define:

$$
m_Q(Q, P) \in \{0,1\}
$$

where $m_Q = 1$ indicates that the prediction addresses the question that was actually asked.

The semantic concept associated with the question is represented by:

$$
K = \alpha(Q)
$$

where $\alpha$ maps a question to its evaluated semantic concept.

These variables provide the basis for distinguishing different types of incorrect predictions.

---

## 3. Completeness

Let

$$
\kappa(G, P) \in [0,1]
$$

represent the degree to which the required information in the ground truth is covered by the prediction.

A prediction with sufficiently complete semantic coverage satisfies:

$$
\kappa(G, P) \geq \tau_{\text{complete}}
$$

while an incomplete prediction satisfies:

$$
0 < \kappa(G, P) < \tau_{\text{complete}}
$$

This variable supports the distinction between a fully correct answer and a partial or incomplete answer.

The threshold $\tau_{\text{complete}}$ must be fixed before applying the taxonomy and documented in the experimental implementation.

---

## 4. Hallucination

Define a hallucination indicator:

$$
H(G, P) \in \{0,1\}
$$

where:

- $H(G, P) = 1$ if $P$ introduces unsupported factual content.
- $H(G, P) = 0$ otherwise.

The hallucination category therefore concerns factual content that is not supported by the available ground-truth information.

A prediction may contain both supported and unsupported information. Such cases require comparison with the Mixed Attribute condition defined below.

---

## 5. Truncation

Define the truncation indicator:

$$
R_{\text{tr}}(P) \in \{0,1\}
$$

where:

- $R_{\text{tr}}(P) = 1$ if the response is visibly cut off before completion.
- $R_{\text{tr}}(P) = 0$ otherwise.

Truncation is therefore determined from the structural completeness of the generated response rather than from semantic disagreement with the ground truth.

---

## 6. Repetition

Define the repetition indicator:

$$
R_{\text{rep}}(P) \in \{0,1\}
$$

where:

- $R_{\text{rep}}(P) = 1$ if information is unnecessarily repeated.
- $R_{\text{rep}}(P) = 0$ otherwise.

This captures cases where the model repeatedly generates the same information without contributing additional relevant content.

---

## 7. Mixed Attribute

Define the Mixed Attribute indicator:

$$
MA(G, P) \in \{0,1\}
$$

where:

- $MA(G, P) = 1$ if $P$ contains both correct and incorrect attribute-level information.
- $MA(G, P) = 0$ otherwise.

This category is used when the prediction combines correct and incorrect information and none of the other defined categories alone sufficiently describes the error.

---

## 8. Correctness

A prediction is assigned to the Correct category when it answers the question appropriately, is semantically consistent with the ground truth, and provides sufficiently complete information.

Define:

$$
C_{\text{correct}}(T) = 1
$$

when:

$$
m_Q(Q, P) = 1
$$

and

$$
m_G(G, P) = 1
$$

and

$$
\kappa(G, P) \geq \tau_{\text{complete}}
$$

and

$$
R_{\text{tr}}(P) = 0
$$

and

$$
R_{\text{rep}}(P) = 0
$$

Minor wording differences are therefore permitted when the semantic content of the prediction remains correct.

---

## 9. Hallucination and Mixed Attribute Distinction

Hallucination and Mixed Attribute must be explicitly distinguished during taxonomy assignment.

Hallucination applies when the prediction introduces unsupported factual content without sufficient correct content to constitute a mixed response.

Mixed Attribute applies when the prediction contains both correct and incorrect attribute-level information and this mixed content is the most appropriate explanation of the failure.

Therefore, when both conditions are potentially satisfied, Mixed Attribute takes priority over the general Hallucination category.

The operational priority is:

$$
C_{\text{MA}} \rightarrow C_{\text{HA}}
$$

This priority is implemented in the taxonomy decision algorithm rather than by redefining the categories themselves.

---

## 10. Taxonomy Decision Function

The final taxonomy is represented by the decision function:

$$
\tau : \mathbb{T} \rightarrow C
$$

where:

- $\mathbb{T}$ is the space of complete prediction tuples.
- $C$ is the category space.

Thus:

$$
\tau(T) = C_r
$$

assigns prediction instance $T$ to one category $C_r$.

The decision process can be represented conceptually as:

$$
T \rightarrow \text{semantic evaluation}
\rightarrow \text{failure conditions}
\rightarrow \tau(T)
\rightarrow C_r
$$

The exact priority and mutually exclusive decision rules are defined in the subsequent taxonomy algorithm.

---

## 11. Category Interpretation

| Category | Mathematical evidence |
|---|---|
| Correct | $m_Q = 1$, $m_G = 1$, $\kappa \geq \tau_{\text{complete}}$, $R_{\text{tr}} = 0$, $R_{\text{rep}} = 0$ |
| Question Misunderstanding | $m_Q = 0$ |
| Hallucination | $H = 1$ when Mixed Attribute does not apply |
| Partial Answer / Incomplete Answer | Correct semantic content with $0 < \kappa < \tau_{\text{complete}}$ |
| Truncation | $R_{\text{tr}} = 1$ |
| Repetition | $R_{\text{rep}} = 1$ |
| Mixed Attribute | $MA = 1$ |

These conditions describe the evidence used by the taxonomy rather than treating the category names themselves as numerical labels.

The final category assignment is determined by the operational decision order defined in the taxonomy algorithm.

---

## 12. Quantitative Category Statistics

For a dataset containing $N$ evaluated prediction instances, define the category indicator:

$$
\mathbf{1}_r(T_i) =
\begin{cases}
1, & \text{if } \tau(T_i) = C_r \\
0, & \text{otherwise}
\end{cases}
$$

The number of predictions assigned to category $C_r$ is:

$$
N_r = \sum_{i=1}^{N} \mathbf{1}_r(T_i)
$$

The category proportion is:

$$
\rho_r = \frac{N_r}{N}
$$

and the percentage representation is:

$$
100\rho_r
$$

Because each prediction receives one final taxonomy category, the category counts satisfy:

$$
\sum_r N_r = N
$$

and:

$$
\sum_r \rho_r = 1
$$

These quantities provide the basis for reporting the distribution of failure types in the test set.

---

## 13. Scope

This formulation defines the measurable variables and category-level conditions required by the proposed taxonomy.

It does not assume that every category must occur in the current test set. A category may be formally defined but have:

$$
N_r = 0
$$

in the evaluated data.

The subsequent taxonomy algorithm specifies the operational decision order used to assign each prediction to exactly one category.
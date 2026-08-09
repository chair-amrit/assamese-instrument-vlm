# Decision Functions

## 1. Purpose

This document defines the decision functions used to transform the mathematical representation of a VQA prediction into an objective failure-taxonomy decision.

The formulation builds directly on the objects defined in `01_mathematical_foundation.md`:

- $T = (I, Q, G, P)$
- $P = f_{\theta}(I, Q)$
- $K = h(Q)$
- $A = \alpha(K)$
- $E(G, P)$
- $C = \tau(T)$

The purpose of this stage is to define **how the evidence in $T$ is evaluated**, rather than simply assigning a failure label by manual inspection.

---

## 2. Prediction Evaluation

The first decision is whether the model prediction agrees with the ground-truth answer.

Define the correctness function:

$$
\delta : \mathbb{T} \rightarrow \{0,1\}
$$

such that

$$
\delta(T) =
\begin{cases}
1, & \text{if } P \text{ is correct with respect to } G \\
0, & \text{otherwise}
\end{cases}
$$

Equivalently:

$$
\delta(T) = E(G,P)
$$

when $E$ is expressed as a binary evaluation.

Thus:

$$
\delta(T) = 1
$$

represents a correct prediction, while

$$
\delta(T) = 0
$$

represents an incorrect prediction.

This provides the first branch of the taxonomy.

---

## 3. Semantic Comparison

Exact string equality is not sufficient for all VQA predictions.

Two answers may differ lexically while expressing the same semantic attribute, while two answers may appear related lexically but represent different attributes.

Therefore, the taxonomy requires semantic comparison between the ground-truth answer and the model prediction.

Let

$$
s_G(T) \in \mathbb{A}
$$

represent the semantic attribute expressed by the ground-truth answer.

Let

$$
s_P(T) \in \mathbb{A}
$$

represent the semantic attribute expressed by the model prediction.

The semantic agreement function is then:

$$
\sigma : \mathbb{T} \rightarrow \{0,1\}
$$

where

$$
\sigma(T) =
\begin{cases}
1, & \text{if } s_G(T) = s_P(T) \\
0, & \text{otherwise}
\end{cases}
$$

This separates semantic correctness from exact textual matching.



## 4. Concept Condition

The question determines which semantic concept is being evaluated:

$$
K = h(Q)
$$

The corresponding relevant attribute is:

$$
A = \alpha(K)
$$

Therefore, the taxonomy evaluates the prediction relative to the attribute required by the question.

Define the attribute-consistency function:

$$
\kappa : \mathbb{T} \rightarrow \{0,1\}
$$

where

$$
\kappa(T) =
\begin{cases}
1, & \text{if the prediction addresses the attribute } A \text{ required by } K \\
0, & \text{otherwise}
\end{cases}
$$

This allows the taxonomy to distinguish an answer that is wrong because of its value from an answer that addresses the wrong semantic attribute.

---

## 5. Ground-Truth Attribute

The ground-truth answer can be mapped to the attribute expected for the evaluated concept.

Define:

$$
A_G = g(G,K)
$$

where $g$ is a semantic interpretation function.

The expected attribute therefore satisfies:

$$
A_G \in \mathbb{A}
$$

Similarly, the predicted answer can be interpreted as:

$$
A_P = g(P,K)
$$

where

$$
A_P \in \mathbb{A}
$$

The semantic comparison can then be expressed as:

$$
\sigma(T) =
\begin{cases}
1, & \text{if } A_G = A_P \\
0, & \text{otherwise}
\end{cases}
$$

This provides a semantic basis for comparing the reference answer and the model prediction.

---

## 6. Prediction-Type Decision

The combination of correctness and semantic agreement provides the basic decision structure.

### Correct prediction

If

$$
\delta(T) = 1
$$

then the prediction is considered correct with respect to the evaluation criterion.

### Incorrect prediction

If

$$
\delta(T) = 0
$$

then further analysis is required.

For an incorrect prediction, the framework examines:

- the semantic concept $K$,
- the required attribute $A$,
- the ground-truth attribute $A_G$,
- the predicted attribute $A_P$.

This prevents all incorrect predictions from being treated as one homogeneous error type.



## 7. Taxonomy Decision Function

The taxonomy function is defined as:

$$
\tau : \mathbb{T} \rightarrow \mathbb{C}
$$

and assigns a category to each complete prediction tuple:

$$
C = \tau(T)
$$

The decision can be represented conceptually as:

$$
T
\rightarrow
\delta(T)
\rightarrow
\text{Correct?}
\rightarrow
\begin{cases}
\text{Correct}, & \delta(T)=1 \\[4pt]
\text{Failure analysis}, & \delta(T)=0
\end{cases}
$$

For an incorrect prediction, the framework evaluates:

$$
K = h(Q)
$$

$$
A = \alpha(K)
$$

and compares the ground-truth and predicted attributes:

$$
A_G = g(G,K)
$$

$$
A_P = g(P,K)
$$

The resulting evidence is then used to determine the appropriate failure category:

$$
C = \tau(T)
$$

The exact category-specific conditions are defined by the final taxonomy.




## 8. Category Decision Rules

Each failure category must be represented by an explicit decision predicate.

For category \(C_r\), define:

$$
d_r : \mathbb{T} \rightarrow \{0,1\}
$$

where

$$
d_r(T)=1
$$

means that prediction tuple \(T\) satisfies the decision conditions for category \(C_r\).

The taxonomy function assigns the corresponding category when:

$$
\tau(T)=C_r
$$

provided that:

$$
d_r(T)=1
$$

The category predicates must be defined so that overlapping conditions are resolved by the operational decision order specified in the taxonomy algorithm.





## 9. Correctness and Failure Separation

The taxonomy first distinguishes correct predictions from predictions requiring failure analysis.

Define the correct-prediction category as:

$$
C_{\mathrm{correct}} \in \mathbb{C}
$$

Then:

$$
\tau(T)=C_{\mathrm{correct}}
$$

when:

$$
\delta(T)=1
$$

For predictions satisfying:

$$
\delta(T)=0
$$

the taxonomy proceeds to failure analysis rather than assigning the Correct category.

Thus:

$$
\delta(T)=0
\quad\Rightarrow\quad
\tau(T)\neq C_{\mathrm{correct}}
$$

The remaining decision functions operate on the subset of predictions requiring failure analysis.

This creates a hierarchical decision structure rather than treating every category independently.




## 10. Semantic Decision Structure

For an incorrect prediction, the relevant decision variables are:

$$
K = h(Q)
$$

$$
A = \alpha(K)
$$

$$
A_G = g(G, K)
$$

$$
A_P = g(P, K)
$$

The semantic relationship is then evaluated through:

$$
\sigma(T) = \mathbb{1}[A_G = A_P]
$$

where $\mathbb{1}[\cdot]$ denotes the indicator function.

Thus:

$$
\sigma(T) =
\begin{cases}
1, & A_G = A_P \\
0, & A_G \neq A_P
\end{cases}
$$

indicates semantic agreement or disagreement, respectively.

The taxonomy can therefore use semantic evidence rather than relying only on surface-form comparison.

---

## 11. Decision-Function Dependency

The functions introduced in this framework form the following dependency structure.

### Question-to-attribute pathway

$$
Q
\rightarrow h(Q)
\rightarrow K
\rightarrow \alpha(K)
\rightarrow A
$$

### Prediction pathway

$$
(I,Q)
\rightarrow f_\theta
\rightarrow P
$$

### Semantic comparison pathway

$$
G \rightarrow A_G
$$

$$
P \rightarrow A_P
$$

followed by:

$$
(A_G,A_P)
\rightarrow \sigma(T)
$$

All of these decisions are associated with the complete prediction tuple:

$$
T=(I,Q,G,P)
$$

and ultimately produce the taxonomy category:

$$
C=\tau(T)
$$

---

## 12. Formal Decision Pipeline

The complete decision process is:

$$
P=f_\theta(I,Q)
$$

$$
T=(I,Q,G,P)
$$

$$
K=h(Q)
$$

$$
A=\alpha(K)
$$

$$
A_G=g(G,K)
$$

$$
A_P=g(P,K)
$$

$$
\delta(T)=E(G,P)
$$

$$
\sigma(T)=\mathbb{1}[A_G=A_P]
$$

$$
C=\tau(T)
$$

Therefore, the complete pipeline can be summarized as:

$$
(I,Q)
\rightarrow f_\theta
\rightarrow P
\rightarrow T
\rightarrow
\{K,A,A_G,A_P,\delta,\sigma\}
\rightarrow \tau
\rightarrow C
$$

This establishes the mathematical interface between the prediction representation and the final failure taxonomy.

---

## 13. Important Constraint

The decision functions must be defined before assigning category labels.

A category should not be defined as:

> "This prediction looks like hallucination."

Instead, each category must be defined through measurable conditions involving the established variables.

For example, a final category rule should have the general form:

$$
C_r = \tau(T)
$$

when a specified combination of decision functions satisfies the conditions for category $C_r$.

The category-specific conditions are therefore defined independently of individual prediction examples.

---

## 14. Core Functions

| Function | Domain → Codomain | Purpose |
|---|---|---|
| $f_\theta$ | $\mathbb{I} \times \mathbb{Q} \rightarrow \mathbb{P}$ | Produces the model prediction |
| $h$ | $\mathbb{Q} \rightarrow \mathbb{K}$ | Identifies the semantic concept represented by the question |
| $\alpha$ | $\mathbb{K} \rightarrow \mathbb{A}$ | Identifies the relevant semantic attribute |
| $E$ | $\mathbb{G} \times \mathbb{P} \rightarrow \mathbb{R}$ | Produces an evaluation score for the prediction against the ground truth |
| $\delta$ | $\mathbb{T} \rightarrow \{0,1\}$ | Determines binary semantic correctness |
| $g_G$ | $\mathbb{G} \times \mathbb{K} \rightarrow \mathbb{A}$ | Extracts the ground-truth attribute associated with the evaluated concept |
| $g_P$ | $\mathbb{P} \times \mathbb{K} \rightarrow \mathbb{A}$ | Extracts the predicted attribute associated with the evaluated concept |
| $\sigma$ | $\mathbb{T} \rightarrow \{0,1\}$ | Determines semantic agreement between ground-truth and predicted attributes |
| $\gamma$ | $\mathbb{T} \rightarrow \{0,1\}$ | Determines attribute-level consistency |
| $d_r$ | $\mathbb{T} \rightarrow \{0,1\}$ | Tests whether prediction tuple $T$ satisfies the conditions associated with category $C_r$ |
| $\tau$ | $\mathbb{T} \rightarrow \mathbb{C}$ | Assigns the final taxonomy category |

Here, $\gamma$ is used specifically for attribute-level consistency so that $\kappa$ can be reserved for the completeness measure defined in the quantitative formulation.

---

## 15. Central Decision Formulation

The core decision structure is:

$$
P = f_\theta(I,Q)
$$

$$
K = h(Q)
$$

$$
A = \alpha(K)
$$

$$
T = (I,Q,G,P)
$$

The evaluation score is:

$$
E(G,P)
$$

The binary correctness decision is obtained from the evaluation criterion:

$$
\delta(T)=
\begin{cases}
1, & \text{if }E(G,P)\text{ satisfies the predefined correctness criterion},\\
0, & \text{otherwise}.
\end{cases}
$$

The ground-truth and predicted attributes associated with the evaluated concept are:

$$
A_G = g_G(G,K)
$$

$$
A_P = g_P(P,K)
$$

Semantic agreement is then defined as:

$$
\sigma(T)=
\begin{cases}
1, & \text{if }A_G=A_P,\\
0, & \text{otherwise}.
\end{cases}
$$

The final taxonomy category is assigned by:

$$
C=\tau(T)
$$

The subsequent quantitative formulation defines the measurable variables and category-specific conditions used by these decision functions.



## 16. Decision-Function Requirements

The decision functions must satisfy the following requirements:

1. **Explicitness** — each decision condition must be formally defined.
2. **Consistency** — the same conditions must be applied to every prediction.
3. **Semantic grounding** — decisions must depend on semantic relationships rather than only surface-form differences.
4. **Category distinction** — different failure mechanisms must be distinguishable through their evaluation conditions.
5. **Reproducibility** — the same prediction and evaluation evidence should produce the same decision.

These requirements ensure that the taxonomy represents an operational evaluation framework rather than a subjective collection of labels.

---

## 17. Transition to Quantitative Formulation

The preceding sections establish the mathematical objects and decision-function interfaces required by the failure taxonomy.

The next stage formalizes the measurable quantities and category-specific conditions used by these decision functions.

In particular, the subsequent formulation defines the variables required to distinguish:

- Question Misunderstanding
- Hallucination
- Partial Answer / Incomplete Answer
- Truncation
- Repetition
- Mixed Attribute
- Correct

The resulting quantitative formulation provides the mathematical basis for the deterministic taxonomy algorithm.
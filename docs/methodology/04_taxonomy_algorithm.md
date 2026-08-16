# Taxonomy Algorithm

## 1. Purpose

This document defines the ordered decision procedure used to assign each VQA prediction to one of the seven final categories defined in the quantitative formulation.

The algorithm operates on the complete prediction tuple:

$$
T = (I, Q, G, P)
$$

and produces an operational outcome:

$$
\tau(T) \in \mathbb{O}
$$

where the extended outcome space is:

$$
\mathbb{O} = \mathbb{C} \cup \{C_{\mathrm{review}}\}
$$

The seven final taxonomy categories are:

$$
\mathbb{C} =
\{C_{\mathrm{correct}}, C_{\mathrm{QM}}, C_{\mathrm{HA}},
C_{\mathrm{PA}}, C_{\mathrm{TR}}, C_{\mathrm{REP}}, C_{\mathrm{MA}}\}
$$

where:

- $C_{\mathrm{correct}}$: Correct
- $C_{\mathrm{QM}}$: Question Misunderstanding
- $C_{\mathrm{HA}}$: Hallucination
- $C_{\mathrm{PA}}$: Partial Answer / Incomplete Answer
- $C_{\mathrm{TR}}$: Truncation
- $C_{\mathrm{REP}}$: Repetition
- $C_{\mathrm{MA}}$: Mixed Attribute

The additional outcome $C_{\mathrm{review}}$ is an internal fallback used when the available evaluation evidence is insufficient for automatic assignment. It is not a final taxonomy category and is resolved before final category statistics are computed.


## 2. Decision Variables

The algorithm uses the evaluation variables defined in the quantitative formulation:

- $m_G(G,P) := \sigma(T)$ — semantic agreement between the ground-truth and predicted attributes (doc 02's $\sigma$).
- $m_Q(Q,P) := \gamma(T)$ — attribute-consistency of the prediction with the question (doc 02's $\gamma$).
- $\kappa(G,P)$ — completeness of the prediction.
- $H(G,P)$ — hallucination indicator.
- $R_{\mathrm{tr}}(P)$ — truncation indicator.
- $R_{\mathrm{rep}}(P)$ — repetition indicator.
- $MA(G,P)$ — Mixed Attribute indicator.
- $\theta_{\mathrm{complete}}$ — predefined completeness threshold.

These variables provide the evidence used by the ordered decision procedure.

The semantic concept associated with the question is represented by:

$$
K = h(Q)
$$

where $h$ maps a question to its evaluated semantic concept.

The decision procedure evaluates these variables in a fixed order so that overlapping failure conditions are resolved consistently.


## 3. Deterministic Decision Order

To ensure that overlapping conditions are resolved consistently, the decision procedure evaluates the categories in the following fixed priority order:

1. Truncation
2. Repetition
3. Question Misunderstanding
4. Mixed Attribute
5. Hallucination
6. Partial Answer / Incomplete Answer
7. Correct
8. Review

The priority order determines which category is assigned when more than one evaluation condition appears to apply.

The first satisfied condition determines the operational outcome $\tau(T)$.

The Review outcome is used only when none of the defined conditions provides sufficient evidence for automatic classification. It is an internal fallback and is not included among the seven final taxonomy categories.


## 4. Decision Rules

For a prediction tuple

$$
T = (I,Q,G,P)
$$

the taxonomy function $\tau(T)$ is evaluated according to the following ordered rules.

### Rule 1 — Truncation

If

$$
R_{\mathrm{tr}}(P)=1
$$

then

$$
\tau(T)=C_{\mathrm{TR}}
$$

The prediction is classified as Truncation because the response is structurally incomplete.

---

### Rule 2 — Repetition

If Rule 1 does not apply and

$$
R_{\mathrm{rep}}(P)=1
$$

then

$$
\tau(T)=C_{\mathrm{REP}}
$$

The prediction is classified as Repetition when unnecessary repeated information is the primary failure.

---

### Rule 3 — Question Misunderstanding

If Rules 1–2 do not apply and

$$
m_Q(Q,P)=0
$$

then

$$
\tau(T)=C_{\mathrm{QM}}
$$

The prediction is classified as Question Misunderstanding when it does not answer the question that was asked.

---

### Rule 4 — Mixed Attribute

If Rules 1–3 do not apply and

$$
MA(G,P)=1
$$

then

$$
\tau(T)=C_{\mathrm{MA}}
$$

The prediction is classified as Mixed Attribute when it contains both correct and incorrect attribute-level information and this category provides the most appropriate explanation of the error.

---

### Rule 5 — Hallucination

If Rules 1–4 do not apply and

$$
H(G,P)=1
$$

then

$$
\tau(T)=C_{\mathrm{HA}}
$$

The prediction is classified as Hallucination when it introduces unsupported factual content without constituting a Mixed Attribute case.

Mixed Attribute therefore takes priority over Hallucination when both conditions are potentially satisfied.

---

### Rule 6 — Partial Answer / Incomplete Answer

If Rules 1–5 do not apply and

$$
m_Q(Q,P)=1
$$

and

$$
m_G(G,P)=1
$$

and

$$
0<\kappa(G,P)<\tau_{\mathrm{complete}}
$$

then

$$
\tau(T)=C_{\mathrm{PA}}
$$

The prediction contains semantically correct information but does not provide sufficient coverage of the required answer.

Here, $m_G(G,P)=1$ indicates that the information present in the prediction is semantically correct, while $\kappa(G,P)$ independently measures how much of the required information is covered.

---

### Rule 7 — Correct

If none of the previous failure conditions applies and

$$
m_Q(Q,P)=1
$$

and

$$
m_G(G,P)=1
$$

and

$$
\kappa(G,P)\geq\theta_{\mathrm{complete}}
$$

and

$$
H(G,P)=0 \quad \text{and} \quad MA(G,P)=0
$$

then

$$
\tau(T)=C_{\mathrm{correct}}
$$

Minor wording differences are permitted when the semantic content remains correct.

---

### Rule 8 — Review

If none of Rules 1–7 applies, then

$$
\tau(T)=C_{\mathrm{review}}
$$

This fallback is used only when the available evaluation variables do not provide sufficient evidence for automatic assignment to one of the seven final taxonomy categories.

Predictions assigned to $C_{\mathrm{review}}$ must be reviewed and resolved before final seven-category statistics are computed.

$C_{\mathrm{review}}$ is an internal operational outcome and is not part of the final taxonomy category space $\mathbb{C}$.


## 5. Compact Decision Function

The complete decision procedure can be represented as the following ordered piecewise function.

Define the ordered decision conditions:

- $D_{\mathrm{TR}}$: $R_{\mathrm{tr}}(P)=1$
- $D_{\mathrm{REP}}$: $R_{\mathrm{tr}}(P)=0$ and $R_{\mathrm{rep}}(P)=1$
- $D_{\mathrm{QM}}$: $R_{\mathrm{tr}}(P)=0$, $R_{\mathrm{rep}}(P)=0$, and $m_Q(Q,P)=0$
- $D_{\mathrm{MA}}$: $R_{\mathrm{tr}}(P)=0$, $R_{\mathrm{rep}}(P)=0$, $m_Q(Q,P)=1$, and $MA(G,P)=1$
- $D_{\mathrm{HA}}$: $R_{\mathrm{tr}}(P)=0$, $R_{\mathrm{rep}}(P)=0$, $m_Q(Q,P)=1$, $MA(G,P)=0$, and $H(G,P)=1$
- $D_{\mathrm{PA}}$: $R_{\mathrm{tr}}(P)=0$, $R_{\mathrm{rep}}(P)=0$, $m_Q(Q,P)=1$, $m_G(G,P)=1$, and $0<\kappa(G,P)<\theta_{\mathrm{complete}}$
- $D_{\mathrm{correct}}$: $R_{\mathrm{tr}}(P)=0$, $R_{\mathrm{rep}}(P)=0$, $m_Q(Q,P)=1$, $m_G(G,P)=1$, $\kappa(G,P)\geq\theta_{\mathrm{complete}}$, $H(G,P)=0$, and $MA(G,P)=0$

The taxonomy function is then:

$$
\tau(T)=
\begin{cases}
C_{\mathrm{TR}} & \text{if } D_{\mathrm{TR}} \\
C_{\mathrm{REP}} & \text{if } D_{\mathrm{REP}} \\
C_{\mathrm{QM}} & \text{if } D_{\mathrm{QM}} \\
C_{\mathrm{MA}} & \text{if } D_{\mathrm{MA}} \\
C_{\mathrm{HA}} & \text{if } D_{\mathrm{HA}} \\
C_{\mathrm{PA}} & \text{if } D_{\mathrm{PA}} \\
C_{\mathrm{correct}} & \text{if } D_{\mathrm{correct}} \\
C_{\mathrm{review}} & \text{otherwise}
\end{cases}
$$

The conditions are evaluated from top to bottom. Therefore, the first satisfied condition determines the operational outcome.

This ordered structure gives Mixed Attribute priority ($C_{\mathrm{MA}} \succ C_{\mathrm{HA}}$) and assigns unresolved cases to the internal Review outcome.

$C_{\mathrm{review}}$ is not included in the seven final taxonomy categories and must be resolved before final category statistics are reported.



## 6. Mutual Exclusivity

The ordered decision procedure assigns each prediction exactly one operational outcome from the extended outcome space:

$$
\mathbb{O} = \mathbb{C} \cup \{C_{\mathrm{review}}\}
$$

For every evaluated prediction $T_i$:

$$
\sum_{r=1}^{|\mathbb{C}|}
\mathbf{1}\!\left[\tau(T_i)=C_r\right]
+
\mathbf{1}\!\left[\tau(T_i)=C_{\mathrm{review}}\right]
=1
$$

where $\mathbf{1}[\cdot]$ is the indicator function.

This means that each prediction receives exactly one operational outcome during the decision procedure.

After all predictions assigned to $C_{\mathrm{review}}$ have been manually resolved into one of the seven final taxonomy categories, the final category assignment satisfies:

$$
\sum_{r=1}^{|\mathbb{C}|}
\mathbf{1}\!\left[\tau(T_i)=C_r\right]
=1
$$

for every evaluated prediction $T_i$.

Therefore, for a dataset containing $N$ evaluated predictions, the final category counts satisfy:

$$
\sum_{r=1}^{|\mathbb{C}|} N_r = N
$$

where $N_r$ denotes the number of predictions assigned to category $C_r$.

Thus, the seven final taxonomy categories form a complete and mutually exclusive partition of the evaluated test set after all Review cases have been resolved.



## 7. Operational Interpretation

The decision procedure can be viewed as a two-stage evaluation process:

$$
T
\;\longrightarrow\;
\text{Structural Evaluation}
\;\longrightarrow\;
\text{Semantic Evaluation}
\;\longrightarrow\;
\tau(T)
$$

### Structural Evaluation

The structural stage evaluates properties of the generated response that can be identified independently of the semantic correctness of its content:

- **Truncation**, represented by $R_{\mathrm{tr}}(P)$.
- **Repetition**, represented by $R_{\mathrm{rep}}(P)$.

These conditions are evaluated first because they describe observable structural properties of the generated response and have higher priority in the taxonomy decision order.

### Semantic Evaluation

If no structural failure is detected, the prediction is evaluated semantically using:

- **Question Misunderstanding**, represented by $m_Q(Q,P)$.
- **Mixed Attribute**, represented by $MA(G,P)$.
- **Hallucination**, represented by $H(G,P)$.
- **Partial Answer / Incomplete Answer**, determined using $m_G(G,P)$ and $\kappa(G,P)$.
- **Correct**, determined using $m_Q(Q,P)$, $m_G(G,P)$, and $\kappa(G,P)$.

The ordered decision procedure then applies the first satisfied condition to determine the operational outcome $\tau(T)$.

If none of the defined conditions provides sufficient evidence for automatic classification, the prediction is assigned to the internal Review outcome $C_{\mathrm{review}}$ and must be resolved before final seven-category statistics are computed.

This structure ensures that the same evaluation procedure is applied consistently to every prediction while preserving the predefined priority among overlapping failure conditions.



## 8. Scope

This algorithm defines the operational decision procedure for assigning VQA predictions to the seven-category failure taxonomy.

The procedure:

- operates on the complete prediction tuple $T=(I,Q,G,P)$;
- evaluates the predefined structural and semantic indicators;
- applies the fixed priority order to resolve overlapping failure conditions;
- permits an internal $C_{\mathrm{review}}$ outcome when automatic evidence is insufficient; and
- requires all Review cases to be resolved before final seven-category statistics are reported.

The taxonomy does not require every category to occur in the evaluated dataset. For any category $C_r \in \mathbb{C}$, its observed count may therefore be zero:

$$
N_r = 0.
$$

After all Review cases have been resolved, every evaluated prediction is assigned to exactly one of the seven final taxonomy categories. Consequently, the final category counts form a complete partition of the evaluated test set:

$$
\sum_{r=1}^{|C|} N_r = N.
$$

The algorithm is intended to provide a reproducible mapping from each prediction tuple to a final taxonomy category when the same evaluation variables, decision order, thresholds, and Review procedure are applied consistently.
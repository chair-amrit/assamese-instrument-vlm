# Mathematical Foundation

## 1. Purpose

This document establishes the mathematical foundation of the VQA failure-taxonomy framework used in the Assamese Musical Instrument VLM project.

The framework represents each VQA prediction as a structured mathematical object and defines the spaces and functions required to evaluate and classify model behavior.

The formulation is designed to support the subsequent decision functions, quantitative formulation, taxonomy algorithm, and final paper formulation.

---

## 2. VQA Dataset

Let the complete VQA dataset be

$$
\mathbb{D} = \{x_i\}_{i=1}^{N}
$$

where each ground-truth sample is

$$
x_i = (I_i, Q_i, G_i)
$$

and:

- $I_i$ is the input image.
- $Q_i$ is the associated question.
- $G_i$ is the ground-truth answer.
- $N$ is the total number of VQA samples.

For the current dataset:

$$
N = 2016
$$

The dataset contains seven Assamese musical instruments with 32 images per instrument:

$$
N_I = 7 \times 32 = 224
$$

Each image is associated with nine VQA concepts:

$$
224 \times 9 = 2016
$$

Thus, the dataset contains 224 unique images and 2016 VQA samples.

---

## 3. Image Space

Let $\mathbb{I}$ denote the image space.

An individual image is represented by

$$
I \in \mathbb{I}
$$

For the current dataset:

$$
\mathbb{I} = \{I_1, I_2, \ldots, I_{224}\}
$$

The image space represents the visual inputs provided to the VQA model.

---


## 4. Question Space

Let $\mathbb{Q}$ denote the question space.

An individual question is represented by

$$
Q \in \mathbb{Q}
$$

If $N_Q$ denotes the number of question instances, then

$$
\mathbb{Q} = \{Q_1, Q_2, \ldots, Q_{N_Q}\}
$$

Questions may have different linguistic formulations while representing the same underlying semantic concept.

Therefore, the exact wording of a question is separated from the semantic concept being evaluated.

---

## 5. Ground-Truth Answer Space

Let $\mathbb{G}$ denote the ground-truth answer space.

An individual ground-truth answer is represented by

$$
G \in \mathbb{G}
$$

For a VQA sample,

$$
G_i
$$

is the reference answer associated with

$$
(I_i, Q_i)
$$

The ground truth represents the expected answer against which the model prediction is evaluated.

---

## 6. Prediction Space and Model

Let $\mathbb{P}$ denote the model prediction space.

An individual prediction is represented by

$$
P \in \mathbb{P}
$$

The fine-tuned Qwen2.5-VL-3B model produces a prediction from an image-question pair:

$$
P = f_{\theta}(I,Q)
$$

where:

- $f_{\theta}$ is the fine-tuned VQA model.
- $\theta$ represents the learned model parameters.
- $I$ is the input image.
- $Q$ is the input question.
- $P$ is the generated answer.

The model is therefore represented as

$$
f_{\theta} : \mathbb{I} \times \mathbb{Q} \rightarrow \mathbb{P}
$$

---


## 7. Prediction-Tuple Space

Let $\mathbb{T}$ denote the space of complete prediction tuples.

A complete prediction instance is represented by

$$
T = (I, Q, G, P)
$$

Since

$$
P = f_{\theta}(I,Q)
$$

the tuple can equivalently be written as

$$
T = \left(I, Q, G, f_{\theta}(I,Q)\right)
$$

The tuple $T$ contains the image, question, ground truth, and model prediction required for subsequent failure analysis.

---

## 8. Semantic Concept Space

Let $\mathbb{K}$ denote the semantic concept space.

An individual concept is represented by

$$
K \in \mathbb{K}
$$

For the current VQA task, concepts correspond to the semantic properties being evaluated, including:

- instrument identity
- material
- origin
- festival
- sound
- traditional player
- playing method
- instrument type
- detailed description

A concept function maps a question to its underlying semantic concept:

$$
h : \mathbb{Q} \rightarrow \mathbb{K}
$$

Therefore,

$$
K = h(Q)
$$

Different linguistic formulations can map to the same concept:

$$
Q_1,\ Q_2,\ Q_3 \rightarrow K_{\mathrm{material}}
$$

This allows the taxonomy to analyze the semantic task rather than treating different question phrasings as different tasks.

---

## 9. Attribute Space

Let $\mathbb{A}$ denote the semantic attribute space.

An individual attribute is represented by

$$
A \in \mathbb{A}
$$

An attribute represents the semantic value relevant to the concept being evaluated.

Examples include:

$$
A_{\mathrm{material}} = \mathrm{bamboo}
$$

$$
A_{\mathrm{instrument}} = \mathrm{pepa}
$$

The attribute function is defined as

$$
\alpha : \mathbb{K} \rightarrow \mathbb{A}
$$

Therefore,

$$
A = \alpha(K)
$$

Combining the concept and attribute mappings gives:

$$
Q \rightarrow h \rightarrow K \rightarrow \alpha \rightarrow A
$$

This provides a structured representation of what semantic attribute the question evaluates.

---


## 10. Ground-Truth and Prediction Evaluation

Given

$$
T = (I, Q, G, P)
$$

the prediction must be evaluated with respect to its ground truth.

Define the evaluation function

$$
E : \mathbb{G} \times \mathbb{P} \rightarrow \mathbb{R}
$$

At the simplest level:

$$
E(G,P) =
\begin{cases}
1, & \text{if } P \text{ is correct with respect to } G \\
0, & \text{otherwise}
\end{cases}
$$

However, binary correctness is not sufficient for the proposed failure taxonomy.

For example, two incorrect predictions may differ semantically even though both receive

$$
E(G,P) = 0
$$

Therefore, additional semantic and decision functions are required to determine the nature of an incorrect prediction.

---

## 11. Failure-Category Space

Let $\mathbb{C}$ denote the failure-category space.

An individual category is represented by

$$
C \in \mathbb{C}
$$

The taxonomy contains a finite set of categories:

$$
\mathbb{C} = \{C_1, C_2, \ldots, C_R\}
$$

where $R$ is the number of categories defined by the final failure taxonomy.

The categories represent distinct model-error mechanisms identified by the proposed framework.

---

## 12. Taxonomy Function

The failure taxonomy is represented as a mapping from the complete prediction-tuple space to the category space:

$$
\tau : \mathbb{T} \rightarrow \mathbb{C}
$$

For a prediction tuple $T$:

$$
C = \tau(T)
$$

or equivalently:

$$
\tau(T) = C
$$

where:

- $T$ contains the evidence associated with the prediction.
- $\tau$ is the taxonomy decision function.
- $C$ is the assigned failure category.

The important distinction is:

$$
T = (I, Q, G, P)
$$

represents the prediction instance,

while

$$
\tau(T) = C
$$

represents the classification of that instance.

The internal decision rules used by $\tau$ are defined in the subsequent decision-function formulation.

---


## 13. Mathematical Structure of the Framework

The complete mathematical structure can be expressed as:

### Dataset representation

$$
x = (I, Q, G)
$$

### Model inference

$$
P = f_{\theta}(I,Q)
$$

### Complete prediction instance

$$
T = (I, Q, G, P)
$$

### Semantic interpretation

$$
K = h(Q)
$$

### Attribute extraction

$$
A = \alpha(K)
$$

### Prediction evaluation

$$
E(G,P)
$$

### Failure classification

$$
C = \tau(T)
$$

Thus, the framework can be viewed as:

$$
(I,Q,G)
\;\downarrow\;
P = f_{\theta}(I,Q)
\;\downarrow\;
T = (I,Q,G,P)
\;\downarrow\;
K = h(Q)
\;\downarrow\;
A = \alpha(K)
\;\downarrow\;
E(G,P)
\;\downarrow\;
C = \tau(T)
$$

The individual functions provide the mathematical interfaces required for the subsequent failure-decision rules.

---

## 14. Why the Formulation Requires Multiple Functions

A single correctness function cannot adequately describe the behavior of a VQA model.

The framework separates the problem into distinct operations:

1. **Model inference**

   $$
   f_{\theta}(I,Q) \rightarrow P
   $$

2. **Concept identification**

   $$
   h(Q) \rightarrow K
   $$

3. **Attribute extraction**

   $$
   \alpha(K) \rightarrow A
   $$

4. **Prediction evaluation**

   $$
   E(G,P) \rightarrow \text{evaluation outcome}
   $$

5. **Failure classification**

   $$
   \tau(T) \rightarrow C
   $$

This separation ensures that semantic interpretation, prediction evaluation, and failure classification are not conflated.

It also allows each component to be independently defined and evaluated.

---

## 15. Core Mathematical Objects

| Object | Symbol | Space | Role |
|---|---|---|---|
| Dataset | $\mathbb{D}$ | — | Complete VQA dataset |
| Image | $I$ | $\mathbb{I}$ | Individual visual input |
| Question | $Q$ | $\mathbb{Q}$ | Individual question |
| Ground truth | $G$ | $\mathbb{G}$ | Reference answer |
| Prediction | $P$ | $\mathbb{P}$ | Model-generated answer |
| Prediction tuple | $T$ | $\mathbb{T}$ | Complete prediction instance |
| Concept | $K$ | $\mathbb{K}$ | Semantic concept evaluated by $Q$ |
| Attribute | $A$ | $\mathbb{A}$ | Relevant semantic attribute |
| Category | $C$ | $\mathbb{C}$ | Assigned failure category |
| VQA model | $f_{\theta}$ | $\mathbb{I} \times \mathbb{Q} \rightarrow \mathbb{P}$ | Fine-tuned Qwen2.5-VL-3B |
| Concept function | $h$ | $\mathbb{Q} \rightarrow \mathbb{K}$ | Question → concept |
| Attribute function | $\alpha$ | $\mathbb{K} \rightarrow \mathbb{A}$ | Concept → attribute |
| Evaluation function | $E$ | $\mathbb{G} \times \mathbb{P} \rightarrow \mathbb{R}$ | Ground-truth/prediction evaluation |
| Taxonomy function | $\tau$ | $\mathbb{T} \rightarrow \mathbb{C}$ | Prediction → category |

---


## 16. Central Formulation

The mathematical foundation of the proposed framework is summarized by:

$$
T = \left(I, Q, G, f_{\theta}(I,Q)\right)
$$

$$
K = h(Q)
$$

$$
A = \alpha(K)
$$

$$
E(G,P)
$$

$$
C = \tau(T)
$$

These definitions establish a consistent mathematical vocabulary for the complete failure-taxonomy framework.

The next stage is to formally define the internal decision functions that determine how the relationship between $G$, $P$, $K$, and $A$ produces each failure category.
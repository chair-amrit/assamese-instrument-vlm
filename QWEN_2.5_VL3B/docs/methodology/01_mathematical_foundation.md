# Mathematical Foundation

## 1. Purpose

This document establishes the mathematical representation of the VQA task used in the Assamese Musical Instrument VLM project. The definitions introduced here provide the formal foundation for the subsequent failure taxonomy, decision functions, quantitative formulation, and taxonomy algorithm.

The objective is to represent each VQA prediction as a mathematical object that can later be mapped to a well-defined failure category.

---

## 2. VQA Dataset

Let the complete VQA dataset be represented by

$$
\mathcal{D} = \{x_i\}_{i=1}^{N}
$$

where each sample is

$$
x_i = (I_i, Q_i, G_i)
$$

and:

- \(I_i\) is the input image.
- \(Q_i\) is the question associated with the image.
- \(G_i\) is the ground-truth answer.
- \(N\) is the total number of VQA samples.

For the current dataset:

$$
N = 2016
$$

The dataset contains seven Assamese musical instruments, with 32 images per instrument:

$$
N_I = 7 \times 32 = 224
$$

Each image is associated with nine VQA concepts, producing:

$$
224 \times 9 = 2016
$$

VQA samples.

---

## 3. Image Space

Let \(\mathcal{I}\) denote the image space.

An individual image is represented as

$$
I \in \mathcal{I}
$$

For the 224 unique images used in the project:

$$
\mathcal{I} =
\{I_1, I_2, \ldots, I_{224}\}
$$

The image space represents the visual inputs provided to the VQA model.

---

## 4. Question Space

Let \(\mathcal{Q}\) denote the space of questions used in the VQA dataset.

An individual question is represented as

$$
Q \in \mathcal{Q}
$$

If the dataset contains \(N_Q\) question instances, then

$$
\mathcal{Q} =
\{Q_1, Q_2, \ldots, Q_{N_Q}\}
$$

Questions are associated with semantic concepts such as instrument identity, material, origin, festival, sound, playing method, and other instrument-related attributes.

Different linguistic formulations may represent the same underlying semantic concept.

---

## 5. Ground-Truth Answer Space

Let \(\mathcal{G}\) denote the ground-truth answer space.

An individual ground-truth answer is represented as

$$
G \in \mathcal{G}
$$

For a VQA sample, \(G\) is the reference answer associated with the image-question pair.

Therefore,

$$
G_i = \text{ground-truth answer for } (I_i,Q_i)
$$

The ground truth is treated as the reference against which the model prediction is evaluated.

---

## 6. Prediction Space

Let \(\mathcal{P}\) denote the model prediction space.

An individual prediction is represented as

$$
P \in \mathcal{P}
$$

The fine-tuned Qwen2.5-VL-3B model produces a prediction from an image-question pair:

$$
P = f_{\theta}(I,Q)
$$

where:

- \(f_{\theta}\) is the trained VQA model.
- \(\theta\) represents the learned model parameters.
- \(I\) is the input image.
- \(Q\) is the input question.
- \(P\) is the generated answer.

Thus, the model can be formally represented as the mapping

$$
f_{\theta} :
\mathcal{I} \times \mathcal{Q}
\rightarrow
\mathcal{P}
$$

---

## 7. VQA Sample

A single ground-truth VQA sample is represented as

$$
x = (I,Q,G)
$$

This contains the complete input and reference information required before model inference.

After inference, the corresponding prediction is

$$
P = f_{\theta}(I,Q)
$$

The complete prediction instance is therefore represented by

$$
T = (I,Q,G,P)
$$

or equivalently,

$$
T = (I,Q,G,f_{\theta}(I,Q))
$$

The tuple \(T\) contains all information required for subsequent error and failure analysis.

---

## 8. Semantic Concepts

Each question in the dataset is associated with a semantic concept.

Let

$$\mathcal{K} = \{K_1, K_2, \ldots, K_M\}$$

denote the set of semantic concepts considered in the VQA task.

For a particular sample,

$$K = h(Q)$$

where \(h\) maps a question to its underlying semantic concept.

For example, different question formulations may correspond to the same concept:

$$Q_1, Q_2, Q_3 \rightarrow K_{\text{material}}$$

This separation is important because failure analysis should distinguish the semantic concept being evaluated from the exact wording of the question.

---

## 9. Attributes

A semantic concept may require one or more attributes to determine whether a prediction is correct.

Let the attribute space be

$$
\mathcal{A}
$$

and let

$$
A \in \mathcal{A}
$$

represent an attribute relevant to the evaluated concept.

For example, a material-related question may involve the attribute

$$
A_{\text{material}} = \text{bamboo}
$$

while an instrument-related question may involve

$$
A_{\text{instrument}} = \text{pepa}
$$

The attribute representation allows the failure taxonomy to compare the semantic content of \(G\) and \(P\), rather than relying only on exact string matching.

---

## 10. Prediction Evaluation

Given a prediction tuple

$$
T=(I,Q,G,P)
$$

the relationship between the ground truth \(G\) and prediction \(P\) determines the outcome of the VQA prediction.

Define an evaluation function

$$
E :
\mathcal{G} \times \mathcal{P}
\rightarrow
\mathcal{R}
$$

where \(\mathcal{R}\) represents the space of evaluation outcomes.

At the simplest level,

$$
E(G,P)=
\begin{cases}
1, & \text{if } P \text{ is correct with respect to } G\\
0, & \text{otherwise}
\end{cases}
$$

However, exact equality alone is insufficient for the proposed failure taxonomy because an incorrect answer may represent different types of errors.

Therefore, the taxonomy requires additional semantic and decision functions.

---

## 11. Failure Taxonomy as a Mapping

Let

$$\mathcal{C} = \{C_1, C_2, \ldots, C_R\}$$

denote the set of failure categories defined by the proposed taxonomy.

The taxonomy is represented as a mapping

$$\tau : \mathcal{T} \rightarrow \mathcal{C}$$

where \(\mathcal{T}\) is the space of complete prediction tuples.

Thus,

$$\tau(T) = C_r$$

means that the prediction instance \(T\) is assigned to failure category \(C_r\).

The important distinction is:

$$T = (I,Q,G,P)$$

contains the evidence, while

$$\tau(T)$$

represents the resulting taxonomy decision.

---

## 12. Formal Pipeline

The complete mathematical process can therefore be expressed as

$$
(I,Q,G)
\xrightarrow{f_{\theta}}
(I,Q,G,P)
\xrightarrow{\tau}
C
$$

or equivalently,

$$
T=(I,Q,G,f_{\theta}(I,Q))
$$

followed by

$$
\tau(T)=C
$$

This establishes the foundation for the subsequent modules.

The later formulation will define the internal decision functions used by \(\tau\), determine how ground truth and predictions are compared, and formally distinguish the different failure categories.

---

## 13. Summary of Core Definitions

| Symbol | Definition |
|---|---|
| 𝔻 | Complete VQA dataset |
| 𝕀 | Image space |
| I | Individual image |
| ℚ | Question space |
| Q | Individual question |
| 𝔾 | Ground-truth answer space |
| G | Ground-truth answer |
| ℙ | Prediction space |
| P | Model prediction |
| f_θ | Fine-tuned VQA model |
| x = (I, Q, G) | Ground-truth VQA sample |
| T = (I, Q, G, P) | Complete prediction tuple |
| 𝕂 | Semantic concept space |
| K | Semantic concept |
| 𝔸 | Attribute space |
| A | Relevant semantic attribute |
| ℂ | Failure-category space |
| τ | Failure-taxonomy mapping |
| C | Assigned failure category |

The central formulation established in this module is:

T = (I, Q, G, f_θ(I, Q))

and

τ(T) = C

These two expressions form the mathematical starting point for formalizing the proposed VQA failure taxonomy.

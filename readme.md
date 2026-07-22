# Assamese Instrument Visual Question Answering (VQA)

A research project on **fine-tuning Vision-Language Models (VLMs)** for **Visual Question Answering (VQA)** on traditional Assamese musical instruments using **Parameter-Efficient Fine-Tuning (PEFT)**.

The objective is to develop a reproducible benchmarking framework for evaluating multiple open-source VLMs on a culturally significant, low-resource dataset.

---

# Project Overview

This repository contains:

- Custom Assamese Instrument VQA dataset
- Fine-tuning pipelines
- Model evaluation
- Quantitative analysis
- Qualitative error analysis
- Research artifacts

Rather than focusing on a single model, this repository is designed to compare different Vision-Language Models under the same dataset and evaluation protocol.

---

# Models

Current implementations:

- ✅ Qwen2.5-VL-3B
- ✅ PaliGemma

Planned:

- InternVL
- LLaVA
- SmolVLM
- Other open-source VLMs

---

# Dataset

The dataset contains Visual Question Answering annotations for seven traditional Assamese musical instruments.

Instrument classes:

- Bahi
- Bihu Dhol
- Gogona
- Khutitaal
- Pepa
- Toka
- Xutuli

Each image is annotated with questions covering:

- Festival
- Origin
- Material
- Parts
- Playing Method
- Sound
- Traditional Player
- Instrument Type
- Description

---

# Repository Structure

```text
assamese-instrument-vlm/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── Paligemma/
│   └── ...
│
└── QWEN_2.5_VL3B/
    └── ...
```

Each model directory contains:

- Fine-tuning notebooks
- Dataset metadata
- Model checkpoints
- Inference outputs
- Evaluation results
- Research analyses

---

# Evaluation

Each model is evaluated using:

- Exact Match Accuracy
- Cosine Similarity
- LaVe Score
- Instrument-wise Performance
- Concept-wise Performance
- Best/Worst Prediction Analysis
- Failure Category Analysis
- Attribute Confusion Analysis

---

# Repository Goal

Provide a reproducible benchmark for:

- Fine-tuning Vision-Language Models
- Evaluating cultural-domain VQA datasets
- Comparing multiple PEFT-based VLMs
- Studying model failure modes through qualitative analysis

---

# License

This repository is intended for research and educational purposes.

Images are not included due to copyright and repository size limitations.

Only annotations, metadata, notebooks, trained adapters (where applicable), evaluation outputs, and research artifacts are provided.

---

# Citation

If you use this repository or dataset in your research, please cite the corresponding publication once available.
# Assamese Musical Instrument Visual Question Answering (VQA)

## Overview

This project develops a **Visual Question Answering (VQA)** pipeline for traditional Assamese musical instruments. Rather than focusing on a single vision-language model, the objective is to build a **reproducible fine-tuning and evaluation framework** that can be applied to different open-source Vision Language Models (VLMs).

The primary goal is to improve question answering performance on a culturally specific domain where publicly available datasets are extremely limited. Throughout this project, multiple VLMs—including **Google PaliGemma-3B** and **Qwen2.5-VL-3B**—were fine-tuned and compared using the same dataset, training strategy, and evaluation pipeline.

The system answers questions about traditional Assamese instruments such as:

- Bahi
- Bihu Dhol
- Gogona
- Khutitaal
- Pepa
- Toka
- Xutuli

Questions cover cultural significance, construction, playing technique, sound characteristics, origin, festival usage, and instrument descriptions.

---

# Project Objectives

- Build a high-quality VQA dataset for Assamese musical instruments.
- Design a reusable fine-tuning pipeline compatible with multiple Vision Language Models.
- Compare different VLM architectures under identical experimental conditions.
- Evaluate model performance using both semantic similarity and LLM-based factual evaluation.
- Identify strengths and limitations of modern VLMs on culturally grounded visual reasoning tasks.

---

# Dataset

### Instruments

- Bahi: 51 usable images
- Bihu Dhol: 60 usable images
- Gogona: 51 usable images
- Khutitaal: 48 usable images
- Pepa: 50 usable images
- Toka: 35 usable images
- Xutuli: 32 usable images

Each image contains nine question-answer pairs covering:

- Festival
- Origin
- Material
- Parts
- Sound
- Traditional Player
- Playing Method
- Instrument Type
- Cultural Description

---

# Models Evaluated

This repository is designed to support multiple Vision Language Models.

Current experiments include:

| Model | Fine-tuning |
|--------|------------|
| Google PaliGemma-3B | QLoRA |
| Qwen2.5-VL-3B-Instruct | QLoRA |

Additional VLMs can be incorporated into the same pipeline with minimal modification.

---

# Fine-tuning Pipeline

The training pipeline includes:

- Dataset verification
- JSONL conversation generation
- HuggingFace Processor integration
- QLoRA fine-tuning (4-bit)
- LoRA adapter training
- Validation-based checkpoint selection
- Inference on unseen test images
- Automatic evaluation
- Result visualization

---

# Evaluation Metrics

Two complementary metrics are used.

### LAVE (LLM-Assisted Visual Evaluation)

Predictions are evaluated using **Gemini** as an LLM judge.

LAVE measures:

- factual correctness
- semantic accuracy
- completeness

---

### Cosine Similarity

Sentence embeddings are generated using Sentence-Transformers and compared using cosine similarity.

This metric measures semantic similarity between predicted and reference answers.

---

# Experimental Results

## PaliGemma-3B

| Metric | Score |
|---------|-------:|
| Cosine Similarity | 0.66 |
| LAVE | 0.13 |

---

## Qwen2.5-VL-3B

| Metric | Score |
|---------|-------:|
| Cosine Similarity | **0.8333** |
| LAVE | **0.7675** |

Qwen2.5-VL significantly outperformed the earlier PaliGemma baseline under the same evaluation protocol.

---

# Key Findings

- QLoRA enables efficient fine-tuning of large Vision Language Models on consumer GPUs.
- Larger instruction-tuned VLMs substantially outperform smaller baselines on culturally grounded VQA.
- Questions requiring direct visual recognition (material, sound, playing method) achieve the highest accuracy.
- Questions involving historical or cultural knowledge (origin, festival) remain more challenging because they require knowledge beyond purely visual cues.
- Combining embedding-based metrics (Cosine Similarity) with LLM-based evaluation (LAVE) provides a more reliable assessment than using either metric alone.

---

# Technology Stack

- PyTorch
- HuggingFace Transformers
- PEFT (LoRA / QLoRA)
- BitsAndBytes
- Sentence Transformers
- Google Gemini API
- Pandas
- Matplotlib

---

# Future Work

- Retrain models using techniques, learnt from previous training runs which held back generalisation.
- Create dataset accordingly to support generalisation of VQA for this dataset.
- Compare other Vision Language Models.
- Investigate multilingual VQA for Assamese and English.
- Explore Retrieval-Augmented VQA for questions requiring historical or cultural knowledge.

---

# Acknowledgement

This project aims to contribute toward AI systems capable of understanding and preserving the cultural heritage of Assamese traditional musical instruments through Vision Language Models and Visual Question Answering.
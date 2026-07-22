# Qwen2.5-VL-3B Fine-Tuning for Assamese Instrument VQA

This directory contains the complete implementation, experiments, and evaluation for fine-tuning **Qwen2.5-VL-3B** using **QLoRA (PEFT)** on a custom Visual Question Answering (VQA) dataset of traditional Assamese musical instruments.

---

## Project Objective

Fine-tune Qwen2.5-VL-3B to answer natural language questions about Assamese musical instruments from images.

The model is trained on a manually curated VQA dataset containing seven traditional instruments and evaluated through both quantitative and qualitative analyses.

---

## Dataset

**Number of instrument classes:** 7

- Bahi
- Bihu Dhol
- Gogona
- Khutitaal
- Pepa
- Toka
- Xutuli

**Images per class:** 32

Dataset split:

| Split | Images/Class |
|--------|-------------:|
| Train | 22 |
| Validation | 5 |
| Test | 5 |

Total Images:

- 224

Question categories:

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

## Directory Structure

```text
QWEN_2.5_VL3B/
│
├── notebooks/
├── dataset/
├── inference/
├── models/
├── research/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Folder Description

### notebooks/

Contains all Jupyter notebooks used throughout the project.

- Dataset preparation
- Fine-tuning
- Model inference
- Error analysis

---

### dataset/

Contains dataset metadata.

Includes:

- final_vqa.csv
- questions.csv
- answers.csv

Images are intentionally excluded from this repository.

---

### models/

Contains trained QLoRA adapter weights and optional checkpoints.

---

### inference/

Contains raw model outputs.

Examples:

- test_predictions.json
- categorized_preds.csv
- categorized_preds_with_attribute.csv

---

### results/

Contains processed evaluation outputs.

Examples:

- evaluation tables
- cosine similarity results
- LaVe evaluation
- best/worst predictions
- loss curve
- experiment summary

---

### research/

Contains research artifacts generated during analysis.

Examples:

- confusion matrices
- normalized confusion matrices
- qualitative analysis outputs
- research figures

---

## Training

Base Model

- Qwen2.5-VL-3B

Fine-tuning Method

- QLoRA
- PEFT

Training Framework

- Hugging Face Transformers
- TRL
- Accelerate

---

## Evaluation

The model is evaluated using:

- Exact Match Accuracy
- Cosine Similarity
- LaVe Score
- Instrument-wise Accuracy
- Concept-wise Accuracy
- Best/Worst Prediction Analysis
- Failure Category Analysis
- Attribute Confusion Analysis

---

## Qualitative Analysis

Errors are categorized into:

- Question Misunderstanding
- Mixed Attribute
- Partial Answer
- Truncation
- Hallucination
- Repetition

Attribute-level confusion analysis is performed to understand semantic error patterns.

---

## Reproducibility

Install dependencies

```bash
pip install -r requirements.txt
```

Run notebooks in order:

1. Dataset Preparation
2. Fine-Tuning
3. Inference
4. Error Analysis

---

## Notes

- Images are not included due to copyright and repository size.
- Only metadata, annotations, model outputs, and research artifacts are provided.
- The repository is organized to support reproducible experiments and future comparison with additional Vision-Language Models.
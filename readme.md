# Assamese Instrument VLM — Cultural Visual Question Answering

Fine-tuning and explainability analysis of a Vision-Language Model on traditional Assamese musical instruments, developed as part of an Ethical Cultural AI research internship at Gauhati University.

---

## Overview

Standard Vision-Language Models (VLMs) fail to recognize and reason about culturally specific objects due to the absence of regional data in their pretraining corpora. This project fine-tunes **Qwen2.5-VL-3B-Instruct** on a custom-curated VQA dataset of 7 traditional Assamese musical instruments, followed by a systematic failure taxonomy and cross-modal explainability analysis.

**Supervisor:** Dr. Rupam Bhattacharya, Gauhati University
**Research theme:** Ethical Cultural AI

---

## Instruments Covered

| Instrument | Type |
|---|---|
| Bahi | Wind (transverse flute) |
| Bihu Dhol | Percussion |
| Gogona | Idiophone (jaw harp) |
| Khutitaal | Percussion (cymbals) |
| Pepa | Wind (buffalo horn) |
| Toka | Percussion (bamboo clapper) |
| Xutuli | Wind (end-blown) |

---

## Dataset

- **7 instrument classes**, 32 images per instrument → 224 unique images
- **9 questions per image** covering festival usage, origin, material, parts, sound, gender, interaction region, instrument type, and detailed description
- **224 × 9 = 2016 total VQA samples**
- **Split:** 70/15/15 (image-level, stratified by instrument) — 22 train / 5 val / 5 test images per instrument
- No image appears across multiple splits (verified, zero leakage)

### Dataset files

```
dataset/
├── final_vqa.csv
├── questions.csv
├── answers.csv
├── train.csv
├── val.csv
├── test.csv
└── dataset_32images/
```


---

## Model & Training

| Component | Choice |
|---|---|
| Base model | Qwen2.5-VL-3B-Instruct |
| Fine-tuning method | QLoRA (4-bit NF4 quantization) |
| LoRA rank | r = 8, alpha = 16 |
| Target modules | q_proj, v_proj |
| Precision | fp16 (bf16 avoided — unstable on Tesla T4) |
| Environment | Kaggle, Tesla T4 (16GB) |
| Gradient checkpointing | Enabled |

Training used a teacher-forced instruction format (`<image> answer en {question}` with the answer as target), image-question-answer triples built from the 63 hand-verified ground-truth answers (8 factual + 1 descriptive question × 7 instruments).

---

## Evaluation

Two complementary metrics assess model output quality:

- **LAVE (LLM-Assisted VQA Evaluation)** — an LLM judge scores factual correctness of predicted answers against ground truth, tolerant of paraphrasing
- **Cosine Similarity** — embedding-based semantic closeness (`all-MiniLM-L6-v2`) between prediction and reference answer

---

## Failure Taxonomy

A systematic taxonomy was developed to categorize model errors, built on a formal framework:
```
T = (I, Q, G, P) — Image, Question, Ground truth, Prediction
Q → concept K → attribute A — semantic decomposition of each question
```

with mapping functions:
```
fθ : 𝕀 × ℚ → ℙ (model prediction function)
h : ℚ → 𝕂 (question → concept)
α : 𝕂 → 𝔸 (concept → attribute)
τ : 𝕋 → ℂ (sample → failure category)
```

### Seven failure categories (mutually exclusive, fixed priority order)

| Priority | Category | Meaning |
|---|---|---|
| 1 | **CTR** | Truncation — answer cut off |
| 2 | **CREP** | Repetition — looping/repeated content |
| 3 | **CQM** | Question Misunderstanding |
| 4 | **CMA** | Mixed Attribute — conflated facts |
| 5 | **CHA** | Hallucination — factually invented content |
| 6 | **CPA** | Partial Answer — incomplete but correct |
| 7 | **Ccorrect** | Correct |

An internal `Creview` fallback flags ambiguous cases for manual resolution before final statistics.

Methodology documented in `docs/methodology/01_mathematical_foundation.md` through `05_paper_formulation.md`.

---

## Explainability

Guide-mandated cross-modal explainability analysis on 35 hand-selected samples (5 per instrument, spanning all 7 failure categories), answering: **is the model's output actually grounded in the image, or hallucinated from language priors?**

### Methods

| Method | What it shows |
|---|---|
| **Cross-modal attention** | Which image patches the model attends to when generating each answer token (per-layer attention averaged across layers and answer tokens — not rollout, which fails under causal masking across the input→answer boundary) |
| **Cross-modal attribution** | Which image patches *causally* drive the output, via Captum Layer Integrated Gradients (n_steps=25, validated for stability at n_steps=50) |

### Architecture traced

```
Image
↓
Vision encoder (SigLIP-style, 2×2 patch merging)
↓
Image patch tokens (raw 52×28 → merged 26×14 = 364 tokens)
↓
Multimodal fusion (interleaved with text tokens, <|image_pad|> = 151655)
↓
Language decoder (Qwen2 LM backbone)
↓
Generated answer (teacher-forced for attention/attribution extraction)
```

### Outputs
Per-sample attention and attribution heatmaps overlaid on original images, aggregated per failure category to reveal category-specific grounding patterns (e.g. hallucinated answers show diffuse/background attention vs. correct answers showing instrument-focused attention).

---

## Repository Structure
```
assamese-instrument-vlm/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── configs/
│ ├── model_config.py
│ ├── lora_config.py
│ └── training_config.py
│
├── dataset/
│ ├── README.md
│ ├── final_vqa.csv
│ ├── questions.csv
│ ├── answers.csv
│ ├── train.csv / val.csv / test.csv
│ └── dataset_32images/
│
├── notebooks/
│ ├── 01_dataset_preparation.ipynb
│ ├── 02_model_training.ipynb
│ ├── 03_model_evaluation.ipynb
│ ├── 04_failure_analysis.ipynb
│ └── 05_explainability.ipynb
│
├── src/
│ ├── data/
│ │ ├── dataset_builder.py
│ │ ├── split_dataset.py
│ │ └── preprocessing.py
│ ├── training/
│ │ ├── train.py
│ │ ├── inference.py
│ │ ├── evaluate.py
│ │ └── load_model.py
│ ├── analysis/
│ │ ├── failure_taxonomy.py
│ │ ├── error_statistics.py
│ │ └── metrics.py
│ ├── explainability/
│ │ ├── attention.py
│ │ ├── attribution.py
│ │ ├── visualization.py
│ │ └── evaluation.py
│ └── utils/
│ ├── io.py
│ ├── plotting.py
│ └── seed.py
│
├── docs/
│ ├── methodology/
│ │ ├── 01_mathematical_foundation.md
│ │ ├── 02_decision_functions.md
│ │ ├── 03_quantitative_formulation.md
│ │ ├── 04_taxonomy_algorithm.md
│ │ └── 05_paper_formulation.md
│ └── explainability/
│ ├── protocol.md
│ ├── qualitative_analysis.md
│ └── quantitative_analysis.md
│
├── results/
│ ├── evaluation/
│ ├── failure_analysis/
│ ├── explainability/
│ └── tables/
│
├── assets/
│ ├── architecture.png
│ ├── pipeline.png
│ └── sample_predictions/
│
└── models/
└── README.md (checkpoint download / access info — weights not stored in repo)
```

---

## Results

| Instrument | LAVE | Cosine Similarity |
|---|---|---|
| Bahi | — | — |
| Bihu Dhol | — | — |
| Gogona | — | — |
| Khutitaal | — | — |
| Pepa | — | — |
| Toka | — | — |
| Xutuli | — | — |
| **Average** | — | — |

*(Updated upon full evaluation completion — see `results/tables/`)*

---

## Setup

```bash
git clone https://github.com/chair-amrit/assamese-instrument-vlm
cd assamese-instrument-vlm
pip install -r requirements.txt
```

Key dependencies: `transformers==4.53.3`, `peft==0.17.0`, `accelerate==1.8.1`, `trl==0.19.1`, `bitsandbytes==0.46.1`, `captum`, `sentence-transformers`, `qwen-vl-utils`.

---

## Acknowledgements

Research internship at Gauhati University under the guidance of Dr. Rupam Bhattacharya, as part of the Ethical Cultural AI research group.

## License

MIT — see [LICENSE](LICENSE).
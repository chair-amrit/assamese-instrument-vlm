# AssameseInstrumentVQA

Fine-tuning a Vision Language Model (VLM) on a custom dataset of 7 traditional Assamese musical instruments for visual question answering (VQA), achieving high performance on cultural object identification, part identification, and multi-dimensional factual reasoning.

---

## Overview

Standard VLMs fail to recognize and reason about culturally specific objects due to the lack of regional data in their pretraining corpus. This project addresses that gap by fine-tuning PaliGemma-3B using QLoRA on a curated dataset of Assamese musical instruments, evaluated using LAVE and Cosine Similarity metrics.

This work is part of an ongoing research internship at Gauhati University focused on Ethical Cultural AI.

---

## Instruments Covered

| Instrument | Type |
|---|---|
| Dhol | Percussion |
| Pepa | Wind |
| Gogona | Idiophone |
| Toka | Percussion |
| Khuti-Taal | Percussion |
| Bahi | Wind |
| Xutuli | Wind |

---

## Dataset

- 7 instrument classes
- ~7-11 images per instrument (full shots, part close-ups, performance context)
- 9 questions per instrument (8 factual + 1 descriptive)
- 63 unique ground truth QA pairs
- ~577 base image-question-answer samples → ~2500+ after augmentation

**Question dimensions covered:**
Festival usage, origin, material, parts, sound, gender, interaction region, instrument type, detailed description.

---

## Model

| Component | Choice |
|---|---|
| Base model | PaliGemma-3B |
| Fine-tuning method | QLoRA (4-bit quantization) |
| LoRA rank | r=8 |
| Target modules | q_proj, v_proj |
| Vision encoder | Frozen |

---

## Evaluation Metrics

- **LAVE (LLM-Assisted VQA Evaluation)** — LLM judge scores factual correctness of predicted answers against ground truth
- **Cosine Similarity** — embedding-based semantic closeness between prediction and reference answer

---

## Project Structure
AssameseInstrumentVQA/

├── dataset/
│   ├── Images/
│   ├── Questions.txt
│   └── dataset.json
├── src/
│   ├── build_dataset.py
│   ├── augment.py
│   ├── train.py
│   └── evaluate.py
├── results/
│   └── scores.json
└── README.md

---

## Results

| Instrument | LAVE | Cosine Similarity |
|---|---|---|
| Dhol | - | - |
| Pepa | - | - |
| Gogona | - | - |
| Toka | - | - |
| Khuti-Taal | - | - |
| Bahi | - | - |
| Xutuli | - | - |
| **Average** | - | - |

*Results will be updated after training completion.*

---

## Setup

```bash
git clone https://github.com/yourusername/AssameseInstrumentVQA
cd AssameseInstrumentVQA
pip install transformers peft accelerate bitsandbytes
pip install sentence-transformers datasets Pillow
```

---

## Acknowledgements

Research internship at Gauhati University under the guidance of Dr Rupam Bhattacharya, as part of the Ethical Cultural AI research group.
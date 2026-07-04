# Assamese Musical Instrument VQA using PaliGemma-3B (QLoRA Fine-tuning)

## Overview
This project fine-tunes Google's PaliGemma-3B vision-language model to perform Visual Question Answering (VQA) on 7 traditional Assamese musical instruments: **Toka, Khutitaal, Xutuli, Bihudhol, Pepa, Gogona, and Bahi**. The goal is to build a culturally grounded AI system capable of answering questions about instrument origin, material, sound, usage, and cultural significance.

## Dataset
- 64 base images across 7 instrument classes
- 9 question-answer pairs per image (576 base samples)
- Questions cover: material, festival/usage context, origin, parts, sound, gender association, playing mechanism, instrument type, and detailed description
- Split into train/validation/test sets, with data augmentation applied to the training set only
- Augmented train set: 1980 samples | Validation: 90 | Test: 90

## Model & Training
- **Base model:** `google/paligemma-3b-pt-224`
- **Fine-tuning method:** QLoRA (4-bit quantization + LoRA adapters)
- **LoRA config:** r=8, alpha=16, target modules: `q_proj`, `v_proj` (attention layers only)
- **Frozen components:** Vision encoder (SigLIP) and base language model weights — only LoRA adapters trained
- **Training setup:** 3 epochs, effective batch size 16 (via gradient accumulation), fp16 precision, gradient checkpointing
- **Hardware:** Kaggle T4 GPU (single GPU)
- **Prompt format:** `<image> answer en {question}` (PaliGemma's native VQA convention)

## Results
| Metric | Score |
|---|---|
| Cosine Similarity (overall) | 0.66 |
| LAVE Score (overall, LLM-judged via Gemini) | 0.13 |

### Per-Instrument Cosine Similarity
| Instrument | Score |
|---|---|
| Bahi | 0.80 |
| Toka | 0.74 |
| Bihudhol | 0.70 |
| Pepa | 0.67 |
| Xutuli | 0.61 |
| Khutitaal | 0.61 |
| Gogona | 0.60 |

## Key Finding
Training loss converged cleanly (4.0 → 0.02) with no train/val divergence, indicating stable QLoRA fine-tuning. However, evaluation revealed a significant gap between **Cosine Similarity (0.66)** and **LAVE (0.13)** scores. Manual inspection showed the model correctly learns high-level, memorizable facts (material, festival name, gender association) but **confabulates fine-grained mechanical details** (parts, playing mechanism, instrument type) — often using plausible-sounding but factually incorrect descriptions.

This divergence demonstrates that **embedding-based similarity metrics alone can overstate model quality**, since incorrect answers using similar domain vocabulary still score well on cosine similarity. LLM-judged metrics like LAVE are essential for catching factual hallucination that surface-level similarity misses.

The likely root cause is **data scarcity**: with only ~9 base images per instrument class, fine-grained/mechanical questions have too few unique training examples for the model to learn instrument-specific details, causing it to generalize from generic "Assamese folk instrument" patterns rather than true instance-level understanding.

## Repository Structure
```text
├── notebook.ipynb              # Full training and evaluation pipeline
├── results/
│   ├── predictions.json        # Model predictions on test set
│   ├── cosine_scores.json      # Cosine similarity evaluation results
│   ├── lave_scores.json        # LAVE evaluation results (Gemini-judged)
│   ├── final_results.csv       # Aggregated metrics
│   └── loss_curve.png          # Training vs validation loss curve
└── README.md
```
## Model Weights
Fine-tuned LoRA adapter weights are hosted on HuggingFace:
**[IsHereAmrit/paligemma-assamese-instruments-qlora](https://huggingface.co/IsHereAmrit/paligemma-assamese-instruments-qlora)**

## Tech Stack
`PyTorch` · `HuggingFace Transformers` · `PEFT (LoRA/QLoRA)` · `BitsAndBytes` · `Sentence-Transformers` · `Google Gemini API` (LAVE evaluation)

## Future Work
- Increase real (non-augmented) image diversity per instrument class
- Investigate semantic grounding via counterfactual image-swap testing and embedding-space clustering, beyond attention-based methods (e.g., GradCAM)
- Rebalance training data toward underrepresented fine-grained question types (parts, mechanism, instrument type)
# Dataset

## Overview

This directory contains the dataset resources used for training and evaluating the Qwen2.5-VL-3B-Instruct model on Assamese musical instrument Visual Question Answering (VQA).

The dataset was designed to evaluate the model's ability to answer questions about traditional Assamese musical instruments from visual inputs.

The dataset contains:

- Structured question and answer definitions
- The final VQA dataset
- Conversation-format JSONL files for Qwen2.5-VL fine-tuning
- Train, validation, and test splits

The original image collection is not included in this repository.

---

## Dataset Organization

    dataset/
    ├── README.md
    ├── Assamese_Musical_Instrument_answers.csv
    ├── Assamese_Musical_Instrument_questions.csv
    ├── final_vqa.csv
    └── jsonl/
        ├── train.jsonl
        ├── validation.jsonl
        └── test.jsonl

---

## Instruments

The dataset covers seven traditional Assamese musical instruments:

1. Bahi
2. Bihu Dhol
3. Gogona
4. Khutitaal
5. Pepa
6. Toka
7. Xutuli

The instrument labels used in the dataset are:

    bahi
    bihu_dhol
    gogona
    khutitaal
    pepa
    toka
    xutuli

---

## Dataset Construction

The dataset was constructed by associating each instrument image with a fixed set of VQA concepts.

Each image is paired with nine question-answer concepts covering different properties and characteristics of the instrument.

The question concepts are:

| Question ID | Concept |
|-------------|---------|
| q1 | Festival |
| q2 | Origin |
| q3 | Material |
| q4 | Parts |
| q5 | Sound |
| q6 | Traditional Player |
| q7 | Playing Method |
| q8 | Instrument Type |
| q9 | Description |

This design allows the model to be evaluated across different semantic properties rather than only instrument identification.

---

## Question Formulation

The question definitions are maintained in:

    Assamese_Musical_Instrument_questions.csv

The corresponding ground-truth answers are maintained in:

    Assamese_Musical_Instrument_answers.csv

The question formulation uses multiple linguistic variants for selected questions to reduce dependence on a single fixed wording.

The final VQA records combine the image, instrument, split, concept, question formulation, and corresponding answer.

---

## Final VQA Dataset

The consolidated dataset is stored in:

    final_vqa.csv

The final dataset contains the following fields:

| Field | Description |
|-------|-------------|
| `image_name` | Name of the image associated with the VQA sample |
| `instrument` | Assamese musical instrument category |
| `split` | Dataset partition: train, validation, or test |
| `concept` | Semantic concept addressed by the question |
| `phrase_used` | Question phrasing variant used for the sample |
| `question` | VQA question presented to the model |
| `answer` | Ground-truth answer |

---

## Dataset Splits

The dataset is divided into three independent partitions:

- Train
- Validation
- Test

The final JSONL dataset contains:

| Split | Samples |
|-------|---------:|
| Train | 1,386 |
| Validation | 315 |
| Test | 315 |
| **Total** | **2,016** |

Each image is associated with nine question-answer pairs.

Therefore:

    1,386 / 9 = 154 training images
    315 / 9  = 35 validation images
    315 / 9  = 35 test images

The dataset verification confirmed that:

- Every JSONL line is valid JSON.
- There are no duplicate `(image, question)` pairs within a split.
- Every image has exactly nine question-answer pairs.
- No image appears in more than one split.

---

## Split Integrity

Image-level separation is maintained between the training, validation, and test partitions.

An image appearing in one partition must not appear in another partition.

This prevents the model from being evaluated on an image that was already observed during training.

The final dataset verification reported:

| Split | Samples | Unique Images | QA Pairs / Image |
|-------|--------:|--------------:|-----------------:|
| Train | 1,386 | 154 | 9 |
| Validation | 315 | 35 | 9 |
| Test | 315 | 35 | 9 |
| **Total** | **2,016** | **224** | **9** |

Split leakage was not detected during dataset verification.

---

## JSONL Dataset

The `jsonl/` directory contains the conversation-format datasets used for Qwen2.5-VL fine-tuning:

    jsonl/
    ├── train.jsonl
    ├── validation.jsonl
    └── test.jsonl

The JSONL representation follows a multimodal conversation structure containing the image input, user question, and assistant answer.

Conceptually, each sample represents:

    Image
      +
    Question
      ↓
    Ground-truth Answer

This format is used when loading the dataset for multimodal fine-tuning.

---

## Dataset Verification

Before training, the JSONL files were programmatically validated.

The verification checks include:

1. File existence
2. JSON validity
3. Expected sample counts
4. Duplicate image-question pairs
5. Unique image counts
6. Number of QA pairs per image
7. Image-level split leakage

The final verification completed successfully with:

    PASSED: 19
    FAILED: 0

The dataset was therefore considered ready for Qwen2.5-VL fine-tuning.

---

## Image Data

The original image collection used to construct the dataset is intentionally not included in this GitHub repository.

Reasons include:

- The image collection is relatively large.
- Some collected images contain visible Shutterstock watermarks.
- Redistribution of externally sourced images may be subject to copyright and licensing restrictions.

Therefore, this repository contains the structured dataset metadata, annotations, questions, answers, and JSONL representations rather than redistributing the original image collection.

Users attempting to reproduce the experiments must obtain or provide the corresponding image data separately and maintain the expected image paths referenced by the JSONL files.

---

## Reproducibility

To reproduce the dataset-dependent experiments:

1. Obtain the corresponding image collection separately.
2. Recreate the expected image directory structure.
3. Place the JSONL files under `dataset/jsonl/`.
4. Ensure that the image paths referenced by the JSONL files resolve correctly.
5. Use the provided train, validation, and test partitions.
6. Follow the training and evaluation procedures documented elsewhere in this repository.

The JSONL files represent the finalized dataset partitions used in the model training and evaluation pipeline.

---

## Relationship to the Research Pipeline

The dataset is the first stage of the overall research pipeline:

    Image Collection
           ↓
    Question & Answer Definition
           ↓
    VQA Dataset Construction
           ↓
    Train / Validation / Test Split
           ↓
    JSONL Conversion
           ↓
    Qwen2.5-VL Fine-tuning
           ↓
    Model Evaluation
           ↓
    Failure Taxonomy
           ↓
    Explainability Analysis

The dataset therefore provides the foundation for both model training and the subsequent failure analysis and explainability experiments.

---

## Intended Use

The dataset is intended for research on:

- Visual Question Answering
- Vision-Language Models
- Assamese cultural heritage
- Assamese musical instruments
- Multimodal model evaluation
- VQA failure analysis
- Explainability of Vision-Language Models

It is particularly designed to support analysis beyond aggregate accuracy by providing questions that target different semantic attributes of the same visual object.

---

## Dataset Limitations

The dataset is relatively small and focuses on seven Assamese musical instrument categories.

The image collection also contains externally sourced images, some of which include visible watermarks. The original images are therefore not redistributed through this repository.

The dataset should consequently be considered a focused research dataset for controlled VQA experimentation rather than a large-scale benchmark.

---

## Files Summary

| File / Directory | Purpose |
|------------------|---------|
| `Assamese_Musical_Instrument_answers.csv` | Ground-truth answer definitions |
| `Assamese_Musical_Instrument_questions.csv` | Question concepts and phrasing variants |
| `final_vqa.csv` | Consolidated VQA metadata |
| `jsonl/train.jsonl` | Training split for multimodal fine-tuning |
| `jsonl/validation.jsonl` | Validation split |
| `jsonl/test.jsonl` | Held-out test split |
| `README.md` | Dataset documentation |

---

## Citation

If this dataset or the resulting model is used in academic work, please cite the corresponding research paper associated with this repository.
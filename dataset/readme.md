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

## Dataset Contents

```text
dataset/
├── README.md
├── final_vqa.csv
├── Assamese_Musical_Instrument_questions.csv
├── Assamese_Musical_Instrument_answers.csv
├── jsonl/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
└── dataset_32images/
```

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

The corresponding dataset labels are:

`bahi`, `bihu_dhol`, `gogona`, `khutitaal`, `pepa`, `toka`, and `xutuli`.

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

The dataset construction and verification utilities are maintained under
[`../src/data/`](../src/data/).

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

The consolidated `final_vqa.csv` is the metadata-level representation of the
VQA dataset and is used for dataset inspection, analysis, and downstream
processing.

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

### Split Integrity

The split is performed at the image level. An image assigned to one partition
must not appear in another partition.

The final dataset therefore contains:

| Split | Samples | Unique Images | QA Pairs / Image |
|---|---:|---:|---:|
| Train | 1,386 | 154 | 9 |
| Validation | 315 | 35 | 9 |
| Test | 315 | 35 | 9 |
| **Total** | **2,016** | **224** | **9** |

Dataset verification checks for duplicate image-question pairs, expected sample
counts, image coverage, and cross-split image leakage.


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

These files are the finalized dataset partitions consumed by the Qwen2.5-VL
training and evaluation workflow.

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

The verification workflow was used to validate the finalized dataset before
model training.

The checks cover structural validity, sample counts, duplicate detection,
image coverage, question-answer pairing, and split integrity.

---

## Image Data and Provenance

The original image collection used to construct the dataset is not included in
this public repository.

The collection has mixed provenance, and some images contain visible watermarks.
Because the applicable rights and redistribution conditions vary by source, the
original image files are not redistributed through GitHub.

The repository therefore provides:

- dataset annotations
- question definitions
- ground-truth answers
- consolidated VQA metadata
- finalized train/validation/test JSONL files
- dataset construction and verification utilities

Users who obtain the corresponding images separately are responsible for
verifying the applicable copyright, licensing, attribution, and redistribution
requirements for each source.


---

## Reproducibility

To reproduce the image-dependent dataset and model experiments:

1. Obtain the corresponding image collection separately.
2. Recreate the expected image directory structure under
   `dataset/dataset_32images/`.
3. Ensure that the image paths referenced by the JSONL files resolve correctly.
4. Use the finalized `train.jsonl`, `validation.jsonl`, and `test.jsonl`
   partitions.
5. Review the project paths in [`../configs/paths.py`](../configs/paths.py).
6. Use the dataset preparation and verification utilities under
   [`../src/data/`](../src/data/).
7. Follow the training and evaluation workflow documented in the root
   [`README.md`](../README.md).

The JSONL files represent the finalized dataset partitions used by the model
training and evaluation pipeline.

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

The dataset is relatively small and focuses on seven Assamese musical
instrument categories.

The image collection has mixed provenance and is not redistributed through this
repository. Some source images also contain visible watermarks.

The dataset should therefore be considered a focused research dataset for
controlled VQA experimentation rather than a large-scale general-purpose VQA
benchmark.

Its findings should be interpreted within the scope of the selected
instruments, concepts, images, and evaluation methodology.


---


## Citation

This dataset is part of the ongoing Assamese Musical Instrument VLM research
project.

A formal academic citation will be added when the associated research work is
submitted or published.

For the current project, please reference the repository:

**Assamese Musical Instrument VLM**  
https://github.com/chair-amrit/assamese-instrument-vlm
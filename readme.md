# Assamese Musical Instrument VLM

[![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--3B-blue)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
[![Task](https://img.shields.io/badge/Task-VQA-purple)](https://github.com/chair-amrit/assamese-instrument-vlm)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Ongoing%20Research-orange)](https://github.com/chair-amrit/assamese-instrument-vlm)

**Fine-tuning and analysis of Qwen2.5-VL-3B-Instruct for Visual Question Answering on Assamese traditional musical instruments.**

**Author:** Amrit Rajkumar  
**Institution:** Assam Don Bosco University  
**Program:** B.Tech in Artificial Intelligence and Machine Learning

[GitHub](https://github.com/chair-amrit/assamese-instrument-vlm) ·
[Email](mailto:amritrajkumar6421@gmail.com)

## Project Overview

This repository contains the research and engineering workflow for a Visual
Question Answering (VQA) system focused on Assamese traditional musical
instruments.

The project uses **Qwen2.5-VL-3B-Instruct** as the multimodal foundation model
and adapts it to the target domain using **QLoRA-based fine-tuning**.

The repository covers the complete experimental workflow:

- VQA dataset preparation and verification
- Image-level train/validation/test splitting
- Model configuration and QLoRA fine-tuning
- Inference and semantic evaluation
- Automated failure categorization
- Mathematical formulation of the failure taxonomy
- Attention- and attribution-based explainability
- Quantitative and qualitative analysis
- Reproducible result export and visualization

The current research focuses on understanding not only whether the model
produces a correct answer, but also **how and why incorrect predictions occur**.

## Research Scope

The current study focuses on seven Assamese musical instruments:

| Instrument |
|---|
| Bahi |
| Bihu Dhol |
| Gogona |
| Khutitaal |
| Pepa |
| Toka |
| Xutuli |

The final dataset contains **224 unique images** and **2,016 VQA samples**,
covering nine semantic concepts per image.

The project uses an image-level train/validation/test organization to prevent
the same source image from appearing across multiple splits.

The evaluation stage extends beyond conventional answer matching by applying
a structured failure taxonomy and subsequent explainability analysis.

### Research Status

This repository represents **ongoing research**.

The experimental pipeline, mathematical methodology, failure taxonomy,
evaluation workflow, and explainability components are being developed and
refined as part of the research process. Results presented in this repository
should therefore be treated as experimental research outputs rather than as
final benchmark claims.

Conference submission and publication are planned for a later stage.

## 4. Research Objective

The primary objective of this project is to develop and systematically analyze a
Visual Question Answering (VQA) system for Assamese traditional musical
instruments using **Qwen2.5-VL-3B-Instruct**.

The project goes beyond measuring whether a generated answer is correct. It
also investigates **how model predictions fail and how those failures can be
systematically characterized and interpreted**.

The research therefore has two main objectives:

### Domain-Specific VQA

- Adapt **Qwen2.5-VL-3B-Instruct** to Assamese musical-instrument VQA.
- Use **QLoRA-based parameter-efficient fine-tuning** for domain adaptation.
- Evaluate the fine-tuned model on a held-out test set.

### Failure Analysis and Explainability

- Categorize model predictions using the project's seven-category failure
  taxonomy.
- Distinguish different failure mechanisms rather than treating all incorrect
  predictions as a single error type.
- Analyze model behavior using attention- and attribution-based explainability
  methods.
- Connect qualitative visual evidence with quantitative failure analysis.

The overall objective is therefore to study both **VQA performance and model
failure behavior** within the target Assamese musical-instrument domain.

---

## 5. Research Contributions

The repository brings together the following research and engineering
components.

### 5.1 Domain-Specific Assamese Musical Instrument VQA

A dedicated VQA dataset is prepared for seven Assamese traditional musical
instruments, with image-level separation between training, validation, and
test data.

The current dataset contains:

- **7 Assamese musical instruments**
- **224 unique images**
- **2,016 VQA samples**
- **9 semantic concepts per image**
- **70% / 15% / 15% train-validation-test organization**

The repository contains the dataset annotations and generated JSONL data.
The original image collection is not included in the public repository.

### 5.2 Parameter-Efficient VLM Adaptation

The project fine-tunes **Qwen2.5-VL-3B-Instruct** using QLoRA.

The project-specific configuration includes:

- 4-bit NF4 quantization
- LoRA adaptation
- `q_proj` and `v_proj` target modules
- LoRA rank `r = 8`
- LoRA alpha `α = 16`
- LoRA dropout `0.1`
- Frozen vision encoder during fine-tuning

These settings describe the project's fine-tuning configuration rather than the
base Qwen2.5-VL architecture itself.

### 5.3 Structured Failure Taxonomy

The project defines seven final prediction categories:

- **Correct**
- **Question Misunderstanding**
- **Hallucination**
- **Partial Answer / Incomplete Answer**
- **Truncation**
- **Repetition**
- **Mixed Attribute**

The taxonomy is formally specified through mathematical definitions,
decision functions, quantitative conditions, and a deterministic
priority-ordered classification procedure.

The corresponding formulation is documented in
[`docs/methodology/`](./docs/methodology/).

### 5.4 Explainability-Oriented Analysis

The project extends failure analysis with model-behavior analysis based on:

- Answer-token to visual-token attention
- Layer-wise attention analysis
- Visual grounding through image-patch representations
- Gradient-based attribution
- Integrated Gradients / Captum-based analysis where applicable
- Qualitative comparison of attention and attribution patterns across failure
  categories

The purpose is to examine visual evidence associated with model predictions
and relate that evidence to the defined failure categories.

### 5.5 Reproducible Research Workflow

The repository separates the major stages of the project into dedicated
modules for:

- configuration
- dataset preparation
- model training
- inference
- evaluation
- failure analysis
- explainability
- methodology
- results
- visualization assets

This organization is intended to make the research workflow easier to inspect,
reproduce, and extend.

---

## 6. System Overview

The project follows an end-to-end workflow from dataset preparation through
model analysis:

```text
Dataset
   ↓
Preprocessing / Train–Validation–Test Split
   ↓
QLoRA Fine-Tuning
   ↓
Inference
   ↓
Generated Answers
   ↓
Evaluation
   ├── LAVE
   └── Cosine Similarity
   ↓
Failure Taxonomy
   ↓
Explainability
   ├── Attention
   └── Attribution
   ↓
Quantitative + Qualitative Analysis
   ↓
Research Results
```

### Project-Level Pipeline

The project-level pipeline describes the complete research workflow:

1. Prepare and verify the Assamese musical-instrument VQA data.
2. Organize the data into training, validation, and test splits.
3. Fine-tune Qwen2.5-VL-3B-Instruct using the project QLoRA configuration.
4. Generate answers for held-out test samples.
5. Evaluate the generated predictions using the implemented evaluation
   procedures.
6. Assign predictions to the seven-category failure taxonomy.
7. Select representative predictions for qualitative analysis.
8. Perform attention- and attribution-based explainability analysis.
9. Aggregate quantitative and qualitative evidence for interpretation.

The project-level workflow is illustrated in
[`assets/pipeline.png`](./assets/pipeline.png).

### Model Architecture

The underlying VLM is **Qwen2.5-VL-3B-Instruct**. Its relevant visual,
multimodal, and language-processing components are documented separately in
the architecture figure:

[`assets/architecture.png`](./assets/architecture.png)

The architecture figure focuses on **how the model processes multimodal
information**, while the pipeline figure focuses on **what was done in the
overall research workflow**.

### Research Outputs

The repository organizes the resulting outputs into dedicated locations:

- [`results/evaluation/`](./results/evaluation/) — evaluation outputs
- [`results/failure_analysis/`](./results/failure_analysis/) — failure-analysis
  results
- [`results/explainability/`](./results/explainability/) — explainability
  outputs
- [`results/tables/`](./results/tables/) — generated result tables
- [`assets/sample_predictions/`](./assets/sample_predictions/) — curated
  qualitative prediction cards

These components provide a traceable path from the input dataset and trained
VLM to evaluation, failure classification, explainability, and final research
analysis.

## 7. Dataset

The project uses a domain-specific Visual Question Answering dataset focused on
seven Assamese traditional musical instruments:

- **Bahi**
- **Bihu Dhol**
- **Gogona**
- **Khutitaal**
- **Pepa**
- **Toka**
- **Xutuli**

The current dataset contains:

- **224 unique images**
- **2,016 VQA samples**
- **9 semantic concepts per image**
- **70% / 15% / 15% train-validation-test organization**

The nine evaluated concepts are:

1. Festival
2. Origin
3. Material
4. Parts
5. Sound
6. Traditional player
7. Playing method
8. Instrument type
9. Detailed description

The repository contains the structured annotations and generated JSONL files used
by the experimental workflow. The original image collection is not included in
the public repository.

### Dataset Files

The main dataset resources are located under
[`dataset/`](./dataset/):

```text
dataset/
├── README.md
├── final_vqa.csv
├── Assamese_Musical_Instrument_questions.csv
├── Assamese_Musical_Instrument_answers.csv
├── jsonl/
│   ├── train
│   ├── val
│   └── test
└── dataset_32images/
```

The repository also contains dataset preparation and verification utilities under
[`src/data/`](./src/data/), including JSONL generation and output-validation
scripts.

### Data Provenance

The image collection has **mixed provenance**. The original image files are
therefore not distributed in this public repository, including images containing
watermarks.

The repository currently provides the associated annotations, structured data,
and processing code required to understand and reproduce the dataset-building
workflow.

Users should verify the applicable rights and source conditions before
redistributing any original image material.

---

## 8. Model and QLoRA Configuration

The project uses **Qwen2.5-VL-3B-Instruct** as the multimodal foundation model.

The repository separates the pretrained model architecture from the
project-specific adaptation configuration.

### Base Model

```text
Model
└── Qwen2.5-VL-3B-Instruct
```

The model architecture is documented separately in
[`assets/architecture.png`](./assets/architecture.png).

### Project Fine-Tuning Configuration

The model is adapted using **QLoRA** with the following project configuration:

| Configuration | Value |
|---|---|
| Base model | Qwen2.5-VL-3B-Instruct |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA target modules | `q_proj`, `v_proj` |
| LoRA rank | `r = 8` |
| LoRA alpha | `α = 16` |
| LoRA dropout | `0.1` |
| Vision encoder | Frozen during project fine-tuning |

Configuration files are maintained under
[`configs/`](./configs/):

```text
configs/
├── model_config.py
├── lora_config.py
├── training_config.py
└── paths.py
```

This separation makes it possible to distinguish the underlying multimodal model
from the parameters and training decisions specific to this research project.

---

## 9. Training and Inference

The training workflow is implemented under
[`src/training/`](./src/training/).

```text
Training Data
     ↓
Qwen2.5-VL-3B-Instruct
     ↓
QLoRA Adaptation
     ↓
Fine-Tuned Model
```

The main training components include:

- [`train.py`](./src/training/train.py) — model fine-tuning
- [`load_model.py`](./src/training/load_model.py) — model loading utilities
- [`inference.py`](./src/training/inference.py) — answer generation
- [`evaluate.py`](./src/training/evaluate.py) — prediction evaluation

The repository also contains the original experimental notebooks under
[`notebooks/`](./notebooks/), which document stages of dataset preparation,
model training, evaluation, result export, and explainability experimentation.

### Inference Workflow

For evaluation, the fine-tuned model receives an image-question pair and
generates a VQA answer:

```text
Image + Question
       ↓
Qwen2.5-VL-3B-Instruct
       ↓
Generated Answer
       ↓
Ground-Truth / Semantic Evaluation
       ↓
Failure Analysis
```

The generated predictions are subsequently passed to the evaluation and failure
analysis stages described in the later sections of this README.

### Reproducibility

The repository keeps configuration, data-processing utilities, training code,
inference code, and result-generation scripts separate so that individual
stages can be inspected and rerun independently.

Additional experimental artifacts and model outputs are organized under
[`models/`](./models/) and [`results/`](./results/), while the public repository
does not assume that large model checkpoints or original image files are
distributed through Git.

## 10. Evaluation

The evaluation stage assesses the generated VQA predictions before they are
passed to the failure-taxonomy and explainability stages.

The repository separates evaluation from failure classification so that prediction
quality and prediction failure type remain distinct analytical components.

### Evaluation Workflow

```text
Ground Truth
      +
Model Prediction
      ↓
Semantic / Answer Evaluation
      ↓
Evaluation Outputs
      ↓
Failure Taxonomy
```

The current evaluation workflow includes:

- **LAVE** — used as an LLM-based answer verification component.
- **Cosine Similarity** — used to measure semantic similarity between the
  reference answer and generated prediction.

These evaluation procedures are implemented and exported through the project's
evaluation workflow. The corresponding outputs are stored under
[`results/evaluation/`](./results/evaluation/).

The repository intentionally does **not** report a single headline performance
number in this README while the research is ongoing. Detailed experimental
outputs should be consulted in the corresponding result files.

---

## 11. Failure Taxonomy

Incorrect VQA predictions are further analyzed using a structured seven-category
failure taxonomy.

The final taxonomy is:

| Category | Description |
|---|---|
| **Correct** | The generated prediction satisfies the evaluation criterion for the question. |
| **Question Misunderstanding** | The prediction addresses a different semantic concept from the one requested by the question. |
| **Hallucination** | The prediction introduces unsupported or incorrect information. |
| **Partial Answer / Incomplete Answer** | The prediction contains only part of the information required by the reference answer. |
| **Truncation** | The generated response is terminated before the required answer is completed. |
| **Repetition** | The prediction unnecessarily repeats information or answer content. |
| **Mixed Attribute** | The prediction combines the requested concept with an incorrect or conflicting attribute. |

### Taxonomy Decision Process

The classification methodology is designed as an ordered decision procedure:

```text
Prediction Tuple
      ↓
Structural Evaluation
      ↓
Semantic Evaluation
      ↓
Category Conditions
      ↓
Priority-Ordered Decision
      ↓
Final Category
```

The methodology explicitly defines the mathematical objects, decision functions,
quantitative criteria, and classification algorithm used in this process.

The methodology is documented in:

- [`01_mathematical_foundation.md`](./docs/methodology/01_mathematical_foundation.md)
- [`02_decision_functions.md`](./docs/methodology/02_decision_functions.md)
- [`03_quantitative_formulation.md`](./docs/methodology/03_quantitative_formulation.md)
- [`04_taxonomy_algorithm.md`](./docs/methodology/04_taxonomy_algorithm.md)

The resulting per-prediction classifications are used for the subsequent
failure-distribution and qualitative analysis stages.

---

## 12. Explainability

The project investigates model behavior beyond the generated answer by
analyzing how visual information contributes to prediction generation.

The explainability workflow focuses on two complementary analysis paths:

```text
                    Model Prediction
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
         Attention                  Attribution
              ↓                         ↓
     Visual-token analysis       Gradient-based analysis
              │                         │
              └────────────┬────────────┘
                           ↓
                  Visual Grounding
                           ↓
               Failure-category analysis
```

### Attention Analysis

The attention analysis examines the relationship between generated answer
tokens and visual representations.

The project's analysis focuses on:

- Answer-token to visual-token attention
- Layer-wise attention behavior
- Visual-token / image-patch relevance
- Heatmap-based visualization
- Comparison of attention patterns across failure categories

The attention implementation is organized under
[`src/explainability/attention.py`](./src/explainability/attention.py),
[`src/explainability/rollout.py`](./src/explainability/rollout.py), and
[`src/explainability/token_analysis.py`](./src/explainability/token_analysis.py).

### Attribution Analysis

The attribution pathway investigates which visual representations contribute
to model outputs using gradient-based methods.

The current explainability work includes:

- Gradient-based attribution
- Integrated Gradients / Captum analysis where applicable
- Patch-level attribution scores
- Spatial visualization of attribution
- Comparison with attention-based visual evidence

The corresponding implementation and visualization components are maintained
under [`src/explainability/`](./src/explainability/).

### Qualitative and Quantitative Explainability

Explainability is analyzed at two levels:

**Qualitative analysis**

- Representative attention and attribution heatmaps
- Prediction-specific visual evidence
- Failure-case inspection
- Comparison of representative examples across taxonomy categories

**Quantitative analysis**

- Aggregated visual relevance measures
- Category-level comparison
- Image-region / patch-level analysis
- Stability and consistency analysis where implemented

Detailed protocols and analysis documentation are available under
[`docs/explainability/`](./docs/explainability/).

The resulting explainability outputs are organized under
[`results/explainability/`](./results/explainability/).

## 13. Repository Structure

The repository is organized to separate data preparation, model development,
analysis, explainability, methodology, and experimental outputs.

```text
assamese-instrument-vlm/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── configs/
│   ├── model_config.py
│   ├── lora_config.py
│   ├── training_config.py
│   └── paths.py
│
├── dataset/
│   ├── README.md
│   ├── final_vqa.csv
│   ├── Assamese_Musical_Instrument_questions.csv
│   ├── Assamese_Musical_Instrument_answers.csv
│   ├── jsonl/
│   │   ├── train
│   │   ├── val
│   │   └── test
│   └── dataset_32images/
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_export_results.ipynb
│   ├── assamese-instrument-vlm-qwen2-5-vl-3b.ipynb
│   └── qwen-explainability-instruments.ipynb
│
├── src/
│   ├── data/
│   │   ├── edit_images.py
│   │   ├── make_jsonlpy
│   │   ├── verify_jsonl.py
│   │   ├── verify_output_csv.py
│   │   └── vqa_dataset_builder.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── inference.py
│   │   ├── evaluate.py
│   │   └── load_model.py
│   │
│   ├── analysis/
│   │   ├── categorize_preds.py
│   │   ├── confusion_matrix.py
│   │   ├── create_sample_prediction_cards.py
│   │   └── failure_distribution.py
│   │
│   ├── explainability/
│   │   ├── attention.py
│   │   ├── rollout.py
│   │   ├── token_analysis.py
│   │   ├── visualization.py
│   │   └── evaluation.py
│   │
│   └── utils/
│       ├── io.py
│       ├── plotting.py
│       └── seed.py
│
├── docs/
│   ├── methodology/
│   │   ├── 01_mathematical_foundation.md
│   │   ├── 02_decision_functions.md
│   │   ├── 03_quantitative_formulation.md
│   │   ├── 04_taxonomy_algorithm.md
│   │   └── readme.md
│   │
│   └── explainability/
│       ├── protocol.md
│       ├── qualitative_analysis.md
│       └── quantitative_analysis.md
│
├── results/
│   ├── evaluation/
│   ├── failure_analysis/
│   ├── explainability/
│   └── tables/
│
├── assets/
│   ├── architecture.png
│   ├── pipeline.png
│   └── sample_predictions/
│
└── models/
    ├── explainability_assets/
    ├── qwen2.5vl_lora/
    ├── training_output/
    └── README.md
```

### Directory Roles

| Directory | Purpose |
|---|---|
| `configs/` | Model, LoRA, training, and path configuration |
| `dataset/` | Dataset annotations and structured training/evaluation data |
| `notebooks/` | Experimental and exploratory workflows |
| `src/data/` | Dataset preparation and verification utilities |
| `src/training/` | Training, inference, model loading, and evaluation |
| `src/analysis/` | Failure categorization, statistics, and qualitative-output generation |
| `src/explainability/` | Attention, attribution, visualization, and explainability analysis |
| `src/utils/` | Shared I/O, plotting, and reproducibility utilities |
| `docs/` | Formal methodology and explainability documentation |
| `results/` | Experimental outputs, tables, and analysis results |
| `assets/` | Research figures and curated qualitative examples |
| `models/` | Model-related artifacts and model-output documentation |

---

## 14. Installation and Environment

The project uses Python-based tooling for dataset preparation, QLoRA
fine-tuning, inference, evaluation, and explainability.

### Clone the Repository

```bash
git clone https://github.com/chair-amrit/assamese-instrument-vlm.git
cd assamese-instrument-vlm
```

### Create a Virtual Environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The dependency specification is maintained in
[`requirements.txt`](./requirements.txt).

### Hardware and Runtime

The project was developed around GPU-based execution for multimodal model
fine-tuning and inference. The training workflow was designed for
memory-constrained environments using QLoRA and quantized model loading.

The repository should therefore be treated as a GPU-oriented research
workflow rather than a CPU-only application.

For the exact training configuration, model loading settings, and runtime
parameters, refer to:

- [`configs/model_config.py`](./configs/model_config.py)
- [`configs/lora_config.py`](./configs/lora_config.py)
- [`configs/training_config.py`](./configs/training_config.py)
- [`src/training/train.py`](./src/training/train.py)

### Dataset Preparation

The public repository contains the structured dataset resources and processing
code, but the original image collection is not included.

Before running image-dependent workflows, ensure that the required image data
is available locally and that the paths defined in
[`configs/paths.py`](./configs/paths.py) are correctly configured.

Dataset construction and validation utilities are available under
[`src/data/`](./src/data/).

---

## 15. Reproduction Workflow

The recommended reproduction path follows the same logical sequence used by the
research project.

### Step 1 — Prepare and Verify the Dataset

Use the dataset preparation utilities to construct and validate the structured
VQA data.

```text
Raw / source data
      ↓
Dataset preparation
      ↓
VQA annotations
      ↓
JSONL generation
      ↓
Verification
```

Relevant scripts are located in:

```text
src/data/
├── vqa_dataset_builder.py
├── make_jsonlpy
├── verify_jsonl.py
└── verify_output_csv.py
```

### Step 2 — Configure the Experiment

Review the configuration files before training:

```text
configs/
├── model_config.py
├── lora_config.py
├── training_config.py
└── paths.py
```

These files provide the project-level configuration used by the training and
inference workflow.

### Step 3 — Fine-Tune the Model

Run the training workflow implemented in:

```text
src/training/train.py
```

The training stage uses the project QLoRA configuration for
Qwen2.5-VL-3B-Instruct.

### Step 4 — Generate Predictions

After training, use:

```text
src/training/inference.py
```

to generate predictions for the required evaluation samples.

The generated predictions can then be passed to the evaluation and failure
analysis stages.

### Step 5 — Evaluate Predictions

Use:

```text
src/training/evaluate.py
```

to perform the project's answer-evaluation workflow.

Evaluation outputs are organized under:

```text
results/evaluation/
```

### Step 6 — Categorize Prediction Failures

Run the failure-analysis workflow under:

```text
src/analysis/
```

The categorization stage assigns predictions to the seven final taxonomy
categories defined by the project's methodology.

Additional utilities in this directory generate failure distributions,
confusion-matrix outputs, and curated qualitative prediction cards.

### Step 7 — Run Explainability Analysis

The explainability workflow is implemented under:

```text
src/explainability/
```

The relevant components cover attention extraction, rollout, token analysis,
visualization, and explainability evaluation.

Detailed protocols are documented under:

[`docs/explainability/`](./docs/explainability/).

### Step 8 — Inspect and Organize Results

Generated outputs should be organized into:

```text
results/
├── evaluation/
├── failure_analysis/
├── explainability/
└── tables/
```

Figures and curated examples are maintained under:

```text
assets/
├── architecture.png
├── pipeline.png
└── sample_predictions/
```

This separation keeps the source code, experimental outputs, and presentation
assets distinct.

### Notebook-Based Reproduction

The repository also preserves the experimental notebook workflow:

```text
notebooks/
├── 01_dataset_preparation.ipynb
├── 02_model_training.ipynb
├── 03_model_evaluation.ipynb
├── 04_export_results.ipynb
├── assamese-instrument-vlm-qwen2-5-vl-3b.ipynb
└── qwen-explainability-instruments.ipynb
```

The notebooks provide an additional record of the experimental development
process, while the `src/` modules provide the organized script-based
implementation.

For a clean reproduction, the recommended order is:

```text
Dataset Preparation
        ↓
Configuration
        ↓
Training
        ↓
Inference
        ↓
Evaluation
        ↓
Failure Analysis
        ↓
Explainability
        ↓
Results
```

## 16. Documentation

The repository maintains dedicated documentation for the mathematical methodology,
failure taxonomy, and explainability procedures used throughout the project.

### Methodology

The formal failure-taxonomy framework is documented under
[`docs/methodology/`](./docs/methodology/).

```text
docs/methodology/
├── 01_mathematical_foundation.md
├── 02_decision_functions.md
├── 03_quantitative_formulation.md
├── 04_taxonomy_algorithm.md
└── readme.md
```

The documents are designed as a sequential methodological chain:

```text
01 Mathematical Foundation
            ↓
02 Decision Functions
            ↓
03 Quantitative Formulation
            ↓
04 Taxonomy Algorithm
            ↓
Final Failure Classification
```

The methodology defines the mathematical objects, evaluation functions,
decision conditions, quantitative criteria, and deterministic taxonomy
procedure used to classify VQA predictions.

The methodology overview is available at
[`docs/methodology/readme.md`](./docs/methodology/readme.md).

### Explainability Documentation

Explainability protocols and analysis procedures are maintained separately:

```text
docs/explainability/
├── protocol.md
├── qualitative_analysis.md
└── quantitative_analysis.md
```

These documents describe the project's explainability workflow, including
attention-based analysis, attribution-based analysis, qualitative inspection,
and quantitative analysis.

---

## 17. Results and Experimental Artifacts

Experimental outputs are organized separately from source code and
documentation.

```text
results/
├── evaluation/
├── failure_analysis/
├── explainability/
└── tables/
```

### Evaluation Results

[`results/evaluation/`](./results/evaluation/) contains outputs produced during
model evaluation, including generated evaluation artifacts and related
analysis files.

### Failure Analysis

[`results/failure_analysis/`](./results/failure_analysis/) contains outputs
associated with the seven-category failure taxonomy, including distributions
and other failure-analysis artifacts.

### Explainability Results

[`results/explainability/`](./results/explainability/) contains generated
attention, attribution, visualization, and related explainability outputs.

### Tables

[`results/tables/`](./results/tables/) is reserved for structured result tables
used during analysis and reporting.

### Curated Qualitative Examples

Representative prediction cards are maintained under:

[`assets/sample_predictions/`](./assets/sample_predictions/)

These examples provide visual summaries of representative model predictions
across the defined failure categories.

The repository currently remains **result-neutral at the top level** because
the research is ongoing. Detailed experimental findings should therefore be
read from the corresponding result artifacts rather than inferred from this
README.

---

## 18. Limitations and Current Scope

This project is an ongoing research effort, and several limitations should be
considered when interpreting the current implementation.

### Dataset Scale

The current dataset contains **224 unique images** and **2,016 VQA samples**
covering seven Assamese musical instruments. The relatively limited visual
dataset size constrains the extent to which conclusions can be generalized to
larger or more diverse VQA datasets.

### Domain Scope

The current task is restricted to a selected set of Assamese traditional
musical instruments and a fixed set of semantic VQA concepts. The resulting
analysis should therefore be interpreted within this domain rather than as a
general evaluation of multimodal VQA systems.

### Image Distribution

The original image collection has mixed provenance and is not distributed in
the public repository. Consequently, reproduction of image-dependent
experiments requires access to the corresponding source images under
appropriate usage rights.

### Ongoing Methodology Development

The failure taxonomy and explainability procedures are part of an ongoing
research process. Definitions, decision criteria, and analysis procedures may
be refined as the research progresses.

### Generalization of Explainability Findings

Attention and attribution visualizations are treated as analytical evidence
for studying model behavior. They should not automatically be interpreted as
causal explanations of the model's internal reasoning.

Further validation is required before drawing broad conclusions about
general multimodal model behavior from the current explainability analyses.

---

## 19. Citation

This repository is currently part of ongoing research and does not yet have a
conference publication or DOI.

A formal citation entry will be added once the corresponding research work has
been submitted or published.

Until then, the project can be referenced through its public repository:

**Assamese Musical Instrument VLM**  
https://github.com/chair-amrit/assamese-instrument-vlm

A machine-readable citation file can be added to
[`CITATION.cff`](./CITATION.cff) when the publication metadata becomes
available.

---

## 20. License and Contact

### License

The source code in this repository is released under the
**MIT License**.

See [`LICENSE`](./LICENSE) for the complete license text.

The MIT License applies to the repository's source code unless otherwise
specified. It should not be interpreted as granting redistribution rights for
third-party dataset images or other externally sourced materials.

### Dataset and Third-Party Content

The repository does not distribute the original image collection because the
images have mixed provenance and may contain externally sourced or watermarked
content.

Users who obtain or redistribute any corresponding image data are responsible
for determining and complying with the applicable source-specific licensing,
copyright, attribution, and usage requirements.

### Contact

**Amrit Rajkumar**

Assam Don Bosco University  
B.Tech in Artificial Intelligence and Machine Learning

- GitHub: https://github.com/chair-amrit
- Repository: https://github.com/chair-amrit/assamese-instrument-vlm
- Email: amritrajkumar6421@gmail.com
- LinkedIn: www.linkedin.com/in/amrit-rajkumar-18257a37a

For research questions, reproducibility issues, or repository-related
discussions, please use the GitHub repository or the contact address above.

## 21. Current Research Direction

This is an ongoing research project. Future work will focus on further validating
the failure taxonomy, strengthening quantitative and explainability analyses,
evaluating robustness and generalization, and preparing the work for academic
publication.

---

## 22. Acknowledgements

This project was developed as part of academic and research work at
**Assam Don Bosco University**.

The project builds upon **Qwen2.5-VL** and the open-source machine-learning
ecosystem used throughout the implementation. Third-party software, models,
and datasets remain subject to their respective licenses and terms of use.

---

## 23. Disclaimer

This repository documents ongoing research and should not be interpreted as a
finalized benchmark, production system, or published scientific result.
Experimental methods and results may evolve as the research progresses.

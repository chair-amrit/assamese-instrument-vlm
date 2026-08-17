# Models

This directory contains model-related artifacts produced during the Assamese
Musical Instrument VLM research workflow.

The project uses **Qwen2.5-VL-3B-Instruct** as the base multimodal model and
adapts it to Assamese musical-instrument Visual Question Answering (VQA) using
**QLoRA**.

The directory separates fine-tuning artifacts, training outputs, and
explainability-related model assets from the source code and configuration.

---

## Directory Organization

```text
models/
├── explainability_assets/
├── qwen2.5vl_lora/
├── training_output/
└── README.md
```

### `explainability_assets/`

Contains model-related files or supporting assets used by the explainability
workflow.

These artifacts support attention, attribution, visual-grounding, and related
analysis and should be interpreted together with the implementation under
[`../src/explainability/`](../src/explainability/) and the documentation under
[`../docs/explainability/`](../docs/explainability/).

### `qwen2.5vl_lora/`

Contains the project-specific **LoRA adapter artifacts** produced during
QLoRA-based adaptation of Qwen2.5-VL-3B-Instruct.

The project configuration used for fine-tuning is maintained under:

```text
../configs/
├── model_config.py
├── lora_config.py
└── training_config.py
```

The adapter directory should be treated as a model artifact rather than as the
source of truth for the training configuration.

### `training_output/`

Contains training-related outputs and checkpoints generated during model
fine-tuning.

These files may include intermediate or final training artifacts and can be
substantially larger than the source-code components of the repository.

---

## Base Model

The project uses:

**Qwen2.5-VL-3B-Instruct**

Official model repository:

https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

The base model itself is not redistributed through this repository. Users
should obtain it directly from the official model source and comply with its
applicable license and terms.

---

## Project Fine-Tuning Configuration

The model is adapted using **QLoRA**.

The project configuration includes:

| Setting | Value |
|---|---|
| Base model | Qwen2.5-VL-3B-Instruct |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA target modules | `q_proj`, `v_proj` |
| LoRA rank | `r = 8` |
| LoRA alpha | `α = 16` |
| LoRA dropout | `0.1` |
| Vision encoder | Frozen during project fine-tuning |

These values describe the project-specific training configuration and are
maintained in the corresponding configuration files under
[`../configs/`](../configs/).

---

## Relationship to the Training Pipeline

Model artifacts are generated through the following workflow:

```text
Configuration
      ↓
Qwen2.5-VL-3B-Instruct
      ↓
4-bit QLoRA Preparation
      ↓
LoRA Fine-Tuning
      ↓
Training Checkpoints
      ↓
LoRA Adapter / Model Artifacts
      ↓
Inference
      ↓
Evaluation
      ↓
Failure Analysis
      ↓
Explainability
```

The main implementation is located under:

```text
../src/training/
```

with the primary components:

- [`train.py`](../src/training/train.py)
- [`load_model.py`](../src/training/load_model.py)
- [`inference.py`](../src/training/inference.py)
- [`evaluate.py`](../src/training/evaluate.py)

---

## Checkpoints and Large Artifacts

Training checkpoints and model weights can be large and are not necessarily
appropriate for normal Git version control.

The repository may therefore keep large or intermediate artifacts outside the
tracked source tree or exclude them using `.gitignore`.

Before committing model artifacts, check:

- file size
- whether the artifact is required for reproduction
- whether it is an intermediate checkpoint or final artifact
- whether an external model/artifact host would be more appropriate

The source code and configuration required to understand or reproduce the
training workflow should remain version-controlled even when large binary
artifacts are kept externally.

---

## Loading a LoRA Adapter

A LoRA adapter is not, by itself, a complete standalone Qwen2.5-VL model.

A compatible workflow generally requires:

```text
Base Qwen2.5-VL-3B-Instruct
          +
Project LoRA Adapter
          ↓
Adapted Model
```

The exact loading and merging procedure should follow the implementation in
[`../src/training/load_model.py`](../src/training/load_model.py) and the
configuration used for the corresponding experiment.

---

## Reproducibility

To reproduce model-related experiments:

1. Obtain the official Qwen2.5-VL-3B-Instruct base model.
2. Prepare the project dataset according to [`../dataset/README.md`](../dataset/README.md).
3. Review the project configuration under [`../configs/`](../configs/).
4. Run the training workflow under [`../src/training/`](../src/training/).
5. Preserve the resulting checkpoints or LoRA adapter artifacts.
6. Use the corresponding adapter/checkpoint for inference and downstream
   evaluation.
7. Record the configuration and artifact version used for each experiment.

The associated notebooks under [`../notebooks/`](../notebooks/) provide an
additional record of the experimental workflow.

---

## Artifact Provenance

Model artifacts should be treated as outputs of the project's experimental
pipeline.

Whenever an artifact is retained, its provenance should be traceable to:

- the base model version
- the project configuration
- the dataset version
- the training run
- the relevant checkpoint or adapter state

This is especially important when comparing multiple experiments or preparing
results for publication.

---

## Publication and Distribution Status

The project does not currently commit to distributing a public fine-tuned
Qwen2.5-VL checkpoint on Hugging Face or another model repository.

Any future public model release will be documented here with:

- the published model URL
- version information
- training configuration
- usage instructions
- applicable licensing information

Until then, users should not assume that the fine-tuned model weights are
publicly downloadable from this repository.

---

## Important Note

`models/` contains **artifacts**, not the primary implementation.

For the research source of truth, refer to:

- [`../configs/`](../configs/) — experiment configuration
- [`../src/training/`](../src/training/) — training and inference code
- [`../docs/methodology/`](../docs/methodology/) — failure-taxonomy methodology
- [`../docs/explainability/`](../docs/explainability/) — explainability protocol
- [`../results/`](../results/) — experimental outputs
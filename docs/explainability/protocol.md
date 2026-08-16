# Explainability Protocol

## Objective

To determine whether the fine-tuned Qwen2.5-VL-3B-Instruct model's predictions on
Assamese musical instrument VQA are grounded in actual visual evidence, or produced
independently of the image via language priors. This directly supports the project's
core goal — evaluating semantic understanding, not just output accuracy — and connects
explainability evidence to the failure taxonomy (T = (I, Q, G, P)) developed earlier
in this work.

## Scope

35 hand-selected samples, 5 per instrument across all 7 Assamese instruments (Bahi,
Bihu Dhol, Gogona, Khutitaal, Pepa, Toka, Xutuli), chosen to span all 7 failure
categories from the taxonomy: Correct, Question Misunderstanding, Hallucination,
Partial Answer / Incomplete Answer, Truncation, Repetition, Mixed Attribute.

## Methods

Two complementary cross-modal explainability methods were used, following guidance
to avoid GradCAM (which shows only where the model looks, not whether the model
understands) and to instead measure grounding directly.

### 1. Cross-Modal Attention

**What it measures:** which image patches the model attends to when generating each
answer token.

**Method:**
- A teacher-forced forward pass is run with the full sequence (image + question +
  generated answer) so that attention weights for answer tokens are available
- `output_attentions=True` is enabled via the eager attention backend (the default
  SDPA backend does not expose attention matrices)
- For each of the 36 transformer layers, attention is averaged across all 16 heads,
  then the submatrix corresponding to answer-token rows × image-patch-token columns
  is extracted
- These per-layer cross-modal submatrices are averaged across all layers, then
  averaged across all answer tokens, producing one attention score per image patch

**Note on method choice:** Attention rollout (the standard technique of multiplying
normalized attention matrices across layers) was attempted first but produced
all-zero cross-modal attention. This is because causal masking in the language model
prevents the rollout's matrix multiplication from correctly propagating attention
mass across the input→answer token boundary. Direct per-layer averaging was used
instead and produced correct, non-degenerate results.

- Image patch positions are located by finding all occurrences of the
  `<|image_pad|>` token (ID 151655) in the input sequence
- Qwen2.5-VL performs 2×2 spatial patch merging, so the raw vision-encoder patch
  grid (e.g. 52×28) is downsampled by a factor of 4 to the token-level grid used
  by the language model (e.g. 26×14) before reshaping attention scores back into
  a 2D spatial map
- The resulting patch-level attention map is upsampled (bilinear interpolation) to
  the original image resolution and overlaid as a heatmap

### 2. Cross-Modal Attribution

**What it measures:** which image regions *causally* drive the model's output,
as opposed to attention's correlational view.

**Method:**
- Captum's `IntegratedGradients` is applied directly on `pixel_values` (not on
  token embeddings — an embedding-layer hook was attempted first via
  `LayerIntegratedGradients` but produced zero gradients, likely due to the
  discrete `input_ids` forward path breaking gradient flow to the hook; using
  `pixel_values` directly as the differentiable input resolved this)
- The target is the logit of the first generated answer token
- Baseline is a zero-valued pixel tensor (representing absence of visual
  information)
- Attribution scores are computed per raw vision-encoder patch, summed across the
  embedding dimension, reshaped into the raw patch grid, and upsampled to the
  original image resolution

**Number of interpolation steps:** Initial runs used `n_steps=10` for computational
feasibility on a T4 GPU (16GB). A validation step (below) later showed this was
insufficient for stable results, and all final attribution results were recomputed
at `n_steps=50`.

## Validation — Attribution Stability

Before treating attribution results as final, a stability check was run: 5 samples
(one per distinct failure category, where available) were processed at both
`n_steps=10` and `n_steps=50`, and the resulting patch-level attribution maps were
compared via cosine similarity.

- **Threshold for stability:** cosine similarity > 0.9
- **Result:** average cosine similarity across the 5 validation samples was
  **0.48**, well below the stability threshold, with individual samples ranging
  from 0.18 to 0.74
- **Conclusion:** `n_steps=10` attribution is unreliable. All 35 samples were
  reprocessed at `n_steps=50` before final aggregation and reporting.

This validation step is treated as a methodological finding in its own right —
it demonstrates that attribution stability cannot be assumed at low step counts
for this model/task combination, and that explicit validation is necessary before
drawing conclusions from gradient-based attribution on multimodal transformers.

## Compute Environment

- Kaggle notebooks, Tesla T4 GPU (16GB)
- Model loaded in fp16 (bfloat16 avoided — produces NaN losses on T4)
- LoRA adapter merged into base Qwen2.5-VL-3B-Instruct before explainability
  analysis
- Attention extraction and attribution extraction were run as two separate
  Kaggle sessions per batch, since running both simultaneously (36-layer attention
  caching + Integrated Gradients' interpolation loop) exceeds available T4 memory
- Gradient checkpointing and parameter freezing (`requires_grad=False` on all
  model weights except the differentiable `pixel_values` input) were used to
  keep attribution within memory limits

## Outputs

For each of the 35 samples: an attention heatmap PNG, an attribution heatmap PNG,
and a JSON record containing per-patch scores, image coverage ratio, and peak
attention/attribution location. Aggregated outputs (per failure category and per
instrument) are documented in `quantitative_analysis.md`, with representative
visual patterns discussed in `qualitative_analysis.md`.
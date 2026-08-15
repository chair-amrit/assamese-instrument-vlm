"""
Cross-modal attention extraction for Qwen2.5-VL.

Extracts attention weights from generated answer tokens to image patch tokens,
using a teacher-forced forward pass and direct per-layer attention averaging
(not rollout — rollout fails under causal masking across the input->answer
boundary, see docs/explainability/protocol.md).
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

IMAGE_PAD_TOKEN_ID = 151655


def load_model_with_eager_attention(model_cls, peft_model_cls, model_id, adapter_dir,
                                     torch_dtype=torch.float16, device_map="auto"):
    """
    Load the fine-tuned Qwen2.5-VL model with the eager attention backend.

    The default SDPA backend does not expose attention weights. Attention-based
    explainability requires this eager backend to access output_attentions.
    """
    base_model = model_cls.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = peft_model_cls.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, base_model


def get_image_patch_positions(input_ids, image_pad_token_id=IMAGE_PAD_TOKEN_ID):
    """Locate all <|image_pad|> token positions in the input sequence."""
    input_ids_list = input_ids[0].cpu().tolist()
    positions = [i for i, tok in enumerate(input_ids_list) if tok == image_pad_token_id]
    return positions


def get_patch_grid_dims(image_grid_thw, n_image_patches):
    """
    Compute the merged (language-model-level) patch grid dimensions.

    Qwen2.5-VL performs 2x2 spatial patch merging, so the raw vision-encoder
    grid must be halved in each dimension to match the token-level grid.
    """
    grid_thw = image_grid_thw[0]
    grid_h, grid_w = grid_thw[1].item(), grid_thw[2].item()

    merge_factor = (grid_h * grid_w) // n_image_patches
    grid_h_tokens = grid_h // 2
    grid_w_tokens = grid_w // 2

    assert grid_h_tokens * grid_w_tokens == n_image_patches, \
        f"Mismatch: {grid_h_tokens * grid_w_tokens} vs {n_image_patches}"

    return grid_h_tokens, grid_w_tokens, merge_factor


def extract_cross_modal_attention(model, processor, inputs, prediction):
    """
    Run a teacher-forced forward pass and extract cross-modal attention.

    Returns:
        patch_attention: np.ndarray [n_image_patches] - averaged attention score per patch
        grid_h_tokens, grid_w_tokens: int - patch grid dimensions
    """
    image_patch_positions = get_image_patch_positions(inputs.input_ids)
    n_image_patches = len(image_patch_positions)
    grid_h_tokens, grid_w_tokens, _ = get_patch_grid_dims(inputs.image_grid_thw, n_image_patches)

    # Tokenize the generated answer
    answer_token_ids = processor.tokenizer(
        prediction, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)

    n_input_tokens = inputs.input_ids.shape[1]
    n_answer_tokens = answer_token_ids.shape[1]

    # Build teacher-forced full sequence (input + generated answer)
    tf_input_ids = torch.cat([inputs.input_ids, answer_token_ids], dim=1)
    tf_attention_mask = torch.ones_like(tf_input_ids)
    answer_positions = list(range(n_input_tokens, n_input_tokens + n_answer_tokens))

    with torch.no_grad():
        tf_outputs = model(
            input_ids=tf_input_ids,
            attention_mask=tf_attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            output_attentions=True,
            return_dict=True,
        )
    tf_attentions = tf_outputs.attentions

    # Direct per-layer attention averaging (NOT rollout - see docs/protocol.md)
    cross_modal_layers = []
    for layer_attn in tf_attentions:
        attn = layer_attn[0].mean(dim=0).cpu().float()  # avg heads -> [seq, seq]
        cm_layer = attn[answer_positions, :][:, image_patch_positions]
        cross_modal_layers.append(cm_layer)

    cross_modal = torch.stack(cross_modal_layers).mean(dim=0)  # [n_answer, n_patches]
    patch_attention = cross_modal.mean(dim=0).numpy()          # [n_patches]

    return patch_attention, grid_h_tokens, grid_w_tokens


def reshape_and_upsample(patch_scores, grid_h, grid_w, target_h, target_w):
    """Reshape flat patch scores to 2D grid and upsample to original image size."""
    patch_map_2d = patch_scores.reshape(grid_h, grid_w)
    patch_tensor = torch.tensor(patch_map_2d).unsqueeze(0).unsqueeze(0).float()
    upsampled = F.interpolate(
        patch_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
    ).squeeze().numpy()
    norm = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
    return patch_map_2d, upsampled, norm


def build_attention_result(sample, prediction, patch_attention, grid_h, grid_w, attn_norm):
    """Package attention scores into the standard result dict for aggregation."""
    peak_row, peak_col = np.unravel_index(patch_attention.argmax(), (grid_h, grid_w))
    return {
        "image": sample["image"],
        "instrument": sample["instrument"],
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "prediction": prediction,
        "failure_category": sample["failure_category"],
        "patch_attention": patch_attention.tolist(),
        "grid_h": grid_h,
        "grid_w": grid_w,
        "image_coverage_ratio": float((attn_norm > 0.5).mean()),
        "peak_patch_row": int(peak_row),
        "peak_patch_col": int(peak_col),
    }
"""
Attention rollout for Qwen2.5-VL (exploratory - not used in final pipeline).

Attention rollout combines attention across all transformer layers by
multiplying normalized, residual-augmented attention matrices layer by
layer. This is the standard technique for aggregating attention across
depth in a transformer.

NOTE: This method was tested and found to fail for cross-modal attention
extraction on this model. Rollout is computed on the input sequence only
(image + question), and causal masking in the language model prevents the
matrix multiplication from correctly propagating attention mass across the
input -> answer token boundary. When filtered to the cross-modal submatrix
(answer token rows x image patch columns), rollout produced all-zero
attention scores.

The final pipeline (see attention.py) uses direct per-layer attention
averaging instead, computed via a teacher-forced forward pass that includes
the generated answer tokens in the same forward call. This module is kept
for documentation of the negative result and for cases where rollout on the
input sequence alone (not cross-modal) is still a valid and useful signal.
"""

import torch


def compute_attention_rollout(attentions):
    """
    Compute attention rollout across all layers on a single forward pass.

    Args:
        attentions: tuple of torch.Tensor, one per layer, each
                    [batch, n_heads, seq_len, seq_len]

    Returns:
        torch.Tensor [seq_len, seq_len] - the rolled-out attention matrix,
        representing aggregated information flow from every token to every
        other token across all layers.

    Method:
        For each layer:
        1. Average attention across all heads
        2. Add identity matrix (residual connection)
        3. Normalize each row to sum to 1
        4. Multiply into the running rollout matrix
    """
    num_layers = len(attentions)
    seq_len = attentions[0].shape[-1]
    device = attentions[0].device

    rollout = torch.eye(seq_len, device=device)

    for layer in range(num_layers):
        attn = attentions[layer][0].mean(dim=0)                  # avg heads -> [seq, seq]
        attn = attn + torch.eye(seq_len, device=attn.device)     # residual connection
        attn = attn / attn.sum(dim=-1, keepdim=True)             # normalize rows
        rollout = attn @ rollout                                  # accumulate across layers

    return rollout.cpu().float()


def extract_cross_modal_from_rollout(rollout, answer_positions, image_patch_positions):
    """
    Attempt to extract the cross-modal submatrix from a rollout matrix.

    WARNING: This is known to produce degenerate (all-zero or near-zero)
    results for answer-token-generated sequences due to causal masking
    breaking rollout's propagation across the input->answer boundary.
    Kept here for documentation purposes only - use
    attention.extract_cross_modal_attention() for the working method.

    Args:
        rollout: torch.Tensor [seq_len, seq_len] from compute_attention_rollout()
        answer_positions: list[int] - row indices of answer tokens
        image_patch_positions: list[int] - column indices of image patch tokens

    Returns:
        torch.Tensor [n_answer_tokens, n_image_patches]
    """
    return rollout[answer_positions, :][:, image_patch_positions]
"""
Token sequence inspection for Qwen2.5-VL multimodal inputs.

Decodes input tokens and locates special tokens marking the boundaries
between image and text regions of the sequence.
"""

import pandas as pd

SPECIAL_TOKENS = {
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|image_pad|>": 151655,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
}


def decode_input_tokens(input_ids, processor):
    """
    Decode every token in the input sequence into a readable DataFrame.

    Args:
        input_ids: torch.Tensor [seq_len] - token IDs for one sample
        processor: the Qwen2.5-VL processor (for tokenizer access)

    Returns:
        pd.DataFrame with columns [Index, Token ID, Token]
    """
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    token_df = pd.DataFrame({
        "Index": range(len(tokens)),
        "Token ID": input_ids.cpu().tolist(),
        "Token": tokens,
    })
    return token_df


def locate_special_tokens(input_ids, special_tokens=SPECIAL_TOKENS, verbose=True):
    """
    Find the positions of all special tokens (vision boundaries, image
    patches, chat turn markers) in the input sequence.

    Args:
        input_ids: torch.Tensor [seq_len] - token IDs for one sample
        special_tokens: dict mapping token name -> token ID
        verbose: if True, print occurrence counts and first 10 positions

    Returns:
        dict mapping token name -> list of positions
    """
    input_ids_list = input_ids.cpu().tolist()
    results = {}

    for name, token_id in special_tokens.items():
        positions = [i for i, x in enumerate(input_ids_list) if x == token_id]
        results[name] = positions

        if verbose:
            print("=" * 60)
            print(name)
            print(f"Occurrences : {len(positions)}")
            if positions:
                print("First 10 Positions:", positions[:10])

    return results


def get_image_patch_positions(input_ids, image_pad_token_id=151655):
    """
    Convenience function: return just the <|image_pad|> token positions.
    (Duplicated in attention.py for module independence.)
    """
    input_ids_list = input_ids.cpu().tolist()
    return [i for i, tok in enumerate(input_ids_list) if tok == image_pad_token_id]
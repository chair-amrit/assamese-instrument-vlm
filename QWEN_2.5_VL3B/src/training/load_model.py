"""
Model and processor loading utilities for Qwen2.5-VL-3B-Instruct.

This module contains the model-loading logic used in the
Assamese Instrument VQA project.
"""

import torch

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from configs.model_config import (
    MODEL_ID,
    MIN_PIXELS,
    MAX_PIXELS,
    TRUST_REMOTE_CODE,
    TORCH_DTYPE,
    DEVICE_MAP,
    bnb_config,
)


def load_processor():
    """
    Load the Qwen2.5-VL processor.

    The processor combines the tokenizer and image processor
    used for multimodal input preparation.
    """

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=TRUST_REMOTE_CODE,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    return processor


def load_model():
    """
    Load the pretrained Qwen2.5-VL-3B-Instruct model
    using 4-bit NF4 quantization.
    """

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    return model


def load_model_and_processor():
    """
    Load both the Qwen2.5-VL model and processor.
    """

    processor = load_processor()

    model = load_model()

    return model, processor
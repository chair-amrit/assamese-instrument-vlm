"""
Model configuration for Qwen2.5-VL.
"""

import torch
from transformers import BitsAndBytesConfig

# Base Model
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Processor Configuration
MIN_PIXELS = 224 * 28 * 28
MAX_PIXELS = 384 * 28 * 28

TRUST_REMOTE_CODE = True

# Quantization
TORCH_DTYPE = torch.float16

DEVICE_MAP = {"": 0}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
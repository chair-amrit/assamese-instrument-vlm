"""
LoRA configuration.
"""

from peft import LoraConfig

# LoRA Hyperparameters
LORA_R = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.05

LORA_BIAS = "none"

TASK_TYPE = "CAUSAL_LM"

TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

# Vision Encoder
FREEZE_VISION_ENCODER = True

# PEFT Configuration
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    task_type=TASK_TYPE,
    target_modules=TARGET_MODULES,
)
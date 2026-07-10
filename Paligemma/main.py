# Step 1: Environment Setup
# Installs required libraries (Transformers, PEFT, BitsAndBytes, Datasets,
# Pillow, Sentence Transformers, Scikit-learn, Torch/TorchVision) for
# PaliGemma QLoRA fine-tuning, enables 4-bit quantization, and verifies
# CUDA/GPU availability (expected: Tesla T4 with ~15 GB VRAM).
# Install dependencies
!pip install -q transformers peft accelerate bitsandbytes
!pip install -q sentence-transformers datasets Pillow scikit-learn
!pip install -q torch torchvision
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Verify GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import torch, gc, subprocess

# Clear Python/PyTorch memory
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Check GPU state
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

# Check CUDA is initializable fresh
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0))
print("Free/Total memory (GB):", [round(x/1e9,2) for x in torch.cuda.mem_get_info()])

# Step 2: Load & Verify Dataset
# Loads train/validation/test JSON splits, verifies sample counts, checks for
# missing or corrupt images, converts images to RGB, and validates instrument
# distribution across splits before training.

import json
import os
from PIL import Image

# Paths
DATA_DIR = "/kaggle/input/datasets/bbeastboy/assamese-instrument-vlm-dataset/kaggle_dataset"
IMAGE_DIR = "/kaggle/input/datasets/bbeastboy/assamese-instrument-vlm-dataset/kaggle_dataset"

# Load splits
with open(os.path.join(DATA_DIR, "train_augmented.json")) as f:
    train_data = json.load(f)
with open(os.path.join(DATA_DIR, "val.json")) as f:
    val_data = json.load(f)
with open(os.path.join(DATA_DIR, "test.json")) as f:
    test_data = json.load(f)

print(f"Train samples: {len(train_data)}")
print(f"Val samples:   {len(val_data)}")
print(f"Test samples:  {len(test_data)}")

# Verify all images exist and are readable
def verify_split(split, name):
    missing = []
    corrupt = []
    for s in split:
        path = os.path.join(DATA_DIR, s["image"])
        if not os.path.exists(path):
            missing.append(s["image"])
        else:
            try:
                img = Image.open(path).convert("RGB")
            except:
                corrupt.append(s["image"])
    print(f"{name}: {len(missing)} missing, {len(corrupt)} corrupt")

verify_split(train_data, "Train")
verify_split(val_data, "Val")
verify_split(test_data, "Test")

# Verify instrument balance
from collections import Counter
print("\nTrain distribution:", Counter(s["instrument"] for s in train_data))
print("Val distribution:",   Counter(s["instrument"] for s in val_data))
print("Test distribution:",  Counter(s["instrument"] for s in test_data))

# Step 3: Load PaliGemma Model
# Authenticates with Hugging Face, loads the PaliGemma-3B model and processor,
# applies 4-bit (NF4) quantization for QLoRA, maps the model to the GPU, and
# verifies successful loading with available VRAM and parameter count.

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig
)

MODEL_ID = "google/paligemma-3b-pt-224"

# Login to HuggingFace (PaliGemma needs gated access)
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

hf_token = UserSecretsClient().get_secret("HF_Paligemma")
login(token=hf_token)

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    low_cpu_mem_usage=True
)

processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)

print(f"Free VRAM after load: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

print("Model loaded successfully")
print("Processor loaded successfully")
print(
    f"Trainable parameters before LoRA: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)

# Step 4: Attach LoRA Adapters
# Prepares the quantized model for QLoRA training, configures LoRA adapters
# (rank, alpha, dropout, target attention layers), attaches them to the model,
# and verifies that only ~0.1% (~8M of 3B) of parameters are trainable.

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
# Prepare for kbit training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: trainable params ~8M / 3B total = ~0.1%


# Step 5: Create Custom Dataset
# Defines a PyTorch Dataset for Assamese VQA that loads images, constructs
# prompts, tokenizes text-image pairs with the processor, prepares model
# inputs/labels, creates train/validation/test datasets, and verifies sample
# sizes and tensor shapes.

from torch.utils.data import Dataset
from PIL import Image
import os

class AssameseVQADataset(Dataset):
    def __init__(self, data, processor, image_dir):
        self.data = data
        self.processor = processor
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = Image.open(
            os.path.join(self.image_dir, sample["image"])
        ).convert("RGB")

        prompt = f"<image> answer en {sample['question']}"

        inputs = self.processor(
            text=prompt,
            images=image,
            suffix=sample["answer"],
            return_tensors="pt",
            padding="max_length",   # ADD
            max_length=384,          # ADD
            truncation=True          # ADD
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "token_type_ids": inputs["token_type_ids"].squeeze(0),
            "labels": inputs["labels"].squeeze(0),
        }


train_dataset = AssameseVQADataset(train_data, processor, IMAGE_DIR)
val_dataset = AssameseVQADataset(val_data, processor, IMAGE_DIR)
test_dataset = AssameseVQADataset(test_data, processor, IMAGE_DIR)

print(f"Train: {len(train_dataset)}")
print(f"Val: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")

sample = train_dataset[0]
for k, v in sample.items():
    print(k, v.shape)
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
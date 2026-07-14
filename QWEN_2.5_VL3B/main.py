# STEP 1 : ENVIRONMENT SETUP
# Install required libraries
!pip install -q -U transformers accelerate peft trl bitsandbytes datasets sentence-transformers qwen-vl-utils

# IMPORT LIBRARIES
import os
import random
import numpy as np
import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from datasets import load_dataset

print("Libraries imported successfully.")

# SET RANDOM SEEDS
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

print(f"Random seed set to {SEED}")

# CHECK GPU

if torch.cuda.is_available():

    device = torch.device("cuda")

    print(f"GPU Name : {torch.cuda.get_device_name(0)}")

    print(f"CUDA Version : {torch.version.cuda}")

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"GPU Memory : {total_memory:.2f} GB")

else:

    raise RuntimeError("CUDA GPU not available.")

# VERIFY GPU COMPATIBILITY

gpu_name = torch.cuda.get_device_name(0)

if "T4" in gpu_name:
    print("T4 GPU detected ✓")

elif "L4" in gpu_name:
    print("L4 GPU detected ✓")

elif "P100" in gpu_name:
    print("Warning: P100 detected. QLoRA may not work reliably with bitsandbytes.")

else:
    print(f"Using GPU: {gpu_name}")

# PRINT PYTORCH INFO
print("PyTorch Version :", torch.__version__)
print("CUDA Available :", torch.cuda.is_available())
print("CUDA Version   :", torch.version.cuda)




#Step 2 — Load & Verify Dataset
#Import Libraries
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

#Dataset Paths
DATASET_ROOT = "/kaggle/input/datasets/bbeastboy/assamese-instrument-vlm-datasetqwen2-5-vl-3b/kaggle_dataset32"

TRAIN_JSON = os.path.join(DATASET_ROOT, "train.jsonl")
VAL_JSON = os.path.join(DATASET_ROOT, "validation.jsonl")
TEST_JSON = os.path.join(DATASET_ROOT, "test.jsonl")

IMAGE_ROOT = os.path.join(DATASET_ROOT, "dataset_32images")

#Load JSONL
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl(TRAIN_JSON)
val_data = load_jsonl(VAL_JSON)
test_data = load_jsonl(TEST_JSON)

#Verify Sample Counts
assert len(train_data) == 1386
assert len(val_data) == 315
assert len(test_data) == 315

print("Dataset sizes are correct.")

#Verify Image Paths
def verify_images(dataset):

    missing = []

    for sample in dataset:

        image_path = sample["messages"][1]["content"][0]["image"]
        full_path = os.path.join(DATASET_ROOT, image_path)

        if not os.path.exists(full_path):
            missing.append(full_path)

    return missing

missing = (
    verify_images(train_data) +
    verify_images(val_data) +
    verify_images(test_data)
)

assert len(missing) == 0

print("All image paths are valid.")

#Display Sample Conversations
def show_sample(dataset, index):

    sample = dataset[index]

    image_path = sample["messages"][1]["content"][0]["image"]
    question = sample["messages"][1]["content"][1]["text"]
    answer = sample["messages"][2]["content"][0]["text"]

    image = Image.open(os.path.join(DATASET_ROOT, image_path))

    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    print("Question:")
    print(question)

    print("\nAnswer:")
    print(answer)

#Visualize Random Samples
show_sample(train_data,0)
show_sample(train_data,150)
show_sample(val_data,10)
show_sample(test_data,20)




#Step 3 — Load Qwen Processor
# Import AutoProcessor
from transformers import AutoProcessor

# Specify the pretrained Qwen2.5-VL model
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Load the processor
processor=AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    min_pixels=224*28*28,
    max_pixels=384*28*28
)

# Display basic processor information
print(f"Model: {MODEL_ID}")
print(f"Processor Loaded: {type(processor).__name__}")

# Check tokenizer vocabulary size
print(f"Vocabulary Size: {processor.tokenizer.vocab_size:,}")

# Display special tokens
print("Special Tokens:")
print(processor.tokenizer.special_tokens_map)

# Verify processor components
print(f"Tokenizer: {type(processor.tokenizer).__name__}")
print(f"Image Processor: {type(processor.image_processor).__name__}")




#Step 4 — Load Qwen2.5-VL-3B (QLoRA)
# Import required libraries
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

# Specify model checkpoint
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load Qwen2.5-VL model in 4-bit
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"":0},
    trust_remote_code=True
)

# Switch model to evaluation mode
model.eval()

# Print model information
print(f"Model Loaded: {MODEL_ID}")
print(f"Model Device: {next(model.parameters()).device}")
print(f"Model Class: {type(model).__name__}")

# Display GPU memory usage
allocated = torch.cuda.memory_allocated()/1024**3
reserved = torch.cuda.memory_reserved()/1024**3

print(f"GPU Memory Allocated : {allocated:.2f} GB")
print(f"GPU Memory Reserved  : {reserved:.2f} GB")
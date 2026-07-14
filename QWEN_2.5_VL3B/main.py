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
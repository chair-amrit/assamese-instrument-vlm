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


# Step 6: Configure Training
# Sets training hyperparameters (batch size, learning rate, scheduler,
# checkpointing, evaluation, mixed precision, and early stopping) and
# initializes the Hugging Face Trainer for QLoRA fine-tuning.

!pip install -q tqdm

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./cultural-paligemma",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=50,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    report_to="none",
    disable_tqdm=False,
    remove_unused_columns=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
print(
    f"Train samples: {len(train_dataset)}"
)
print(
    f"Validation samples: {len(val_dataset)}"
)

import sys
from tqdm.auto import tqdm
tqdm.pandas()

# Train
trainer.train()

# Save best LoRA adapter
model.save_pretrained("./best_checkpoint")
processor.save_pretrained("./best_checkpoint")
print("Best checkpoint saved.")

import shutil

# Zip the checkpoint folder
shutil.make_archive("best_checkpoint", "zip", "./best_checkpoint")

# Also zip the full training logs/checkpoints folder
shutil.make_archive("cultural-paligemma", "zip", "./cultural-paligemma")

print("Zipped files ready in /kaggle/working/:")
print("- best_checkpoint.zip")
print("- cultural-paligemma.zip")


# Step 7: Monitor Training Performance
# Loads training logs from the latest checkpoint, extracts training and
# validation losses, plots the loss curves, saves the visualization, and
# reports the best validation loss achieved during training.

import json
import matplotlib.pyplot as plt
import glob

# Find the latest checkpoint's trainer_state.json
checkpoint_dirs = sorted(glob.glob("./cultural-paligemma/checkpoint-*"), 
                          key=lambda x: int(x.split("-")[-1]))
latest_checkpoint = checkpoint_dirs[-1]
print("Reading from:", latest_checkpoint)

with open(f"{latest_checkpoint}/trainer_state.json") as f:
    state = json.load(f)

logs = state["log_history"]

train_loss = [(x["step"], x["loss"]) for x in logs if "loss" in x and "eval_loss" not in x]
eval_loss  = [(x["step"], x["eval_loss"]) for x in logs if "eval_loss" in x]

train_steps, train_values = zip(*train_loss)
eval_steps, eval_values = zip(*eval_loss)

plt.figure(figsize=(10, 4))
plt.plot(train_steps, train_values, label="Train Loss")
plt.plot(eval_steps, eval_values, label="Validation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Best eval_loss: {min(eval_values):.4f} at step {eval_steps[eval_values.index(min(eval_values))]}")


# Step 8: Run Inference & Save Predictions
# Frees GPU memory, loads the fine-tuned LoRA checkpoint for inference,
# generates answers for the test set, stores predictions alongside ground
# truth, and saves the results as a JSON file for evaluation.
import gc, torch

# Force cleanup
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print("Free VRAM (GB):", torch.cuda.mem_get_info()[0]/1e9)
print("Total VRAM (GB):", torch.cuda.mem_get_info()[1]/1e9)

# List any tensors still on GPU
import gc
count = 0
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.is_cuda:
            count += 1
    except:
        pass
print("Tensors still on GPU:", count)

from peft import PeftModel

# Load best checkpoint
model_inf = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model_inf = PeftModel.from_pretrained(model_inf, "./best_checkpoint")
model_inf.eval()

def generate_answer(image_path, question, max_new_tokens=60):
    image = Image.open(image_path).convert("RGB")
    prompt = f"question: {question}\nanswer:"
    
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        output = model_inf.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )
    
    # Decode and strip prompt
    full_output = processor.tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )
    answer = full_output.split("answer:")[-1].strip()
    return answer

# Run inference on full test set
predictions = []
for sample in test_data:
    img_path = os.path.join(IMAGE_DIR, sample["image"])
    pred = generate_answer(img_path, sample["question"])
    predictions.append({
        "image":        sample["image"],
        "instrument":   sample["instrument"],
        "question":     sample["question"],
        "ground_truth": sample["answer"],
        "prediction":   pred
    })

import os 

# Save predictions
os.makedirs("results", exist_ok=True)
with open("results/predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
print(f"Saved {len(predictions)} predictions.")


# Step 9: Evaluate Model Performance
# Computes semantic similarity (cosine similarity using Sentence Transformers)
# and LAVE scores (Gemini-based judge) for model predictions, reports overall
# and per-category metrics, saves evaluation results, and displays sample
# predictions for qualitative analysis.

#cosine similarity scoring using Sentence Transformers
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_sim(pred, gt):
    e1 = embedder.encode(pred)
    e2 = embedder.encode(gt)
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

# Score all predictions
for p in predictions:
    p["cosine_sim"] = cosine_sim(p["prediction"], p["ground_truth"])

# Overall average
avg_cos = np.mean([p["cosine_sim"] for p in predictions])
print(f"Overall Cosine Similarity: {avg_cos:.4f}")

# Per instrument
from collections import defaultdict
instrument_cos = defaultdict(list)
for p in predictions:
    instrument_cos[p["instrument"]].append(p["cosine_sim"])

print("\nPer Instrument Cosine Similarity:")
for inst, scores in instrument_cos.items():
    print(f"  {inst:15s}: {np.mean(scores):.4f}")

# Per question type (Q1-Q9)
questions_list = [
    "festival", "origin", "material", "parts",
    "sound", "gender", "interaction", "type", "description"
]
question_cos = defaultdict(list)
for p in predictions:
    for i, keyword in enumerate(questions_list):
        if keyword in p["question"].lower():
            question_cos[f"Q{i+1}"].append(p["cosine_sim"])

print("\nPer Question Type Cosine Similarity:")
for q, scores in question_cos.items():
    print(f"  {q}: {np.mean(scores):.4f}")

with open("results/cosine_scores.json", "w") as f:
    json.dump(predictions, f, indent=2)
    
    
#LAVE scoring using Gemini-2.5-flash model
!pip install -q google-generativeaiimport google.generativeai as genai
      
import time
import numpy as np
from collections import defaultdict
import json

from kaggle_secrets import UserSecretsClient

gemini_key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
genai.configure(api_key=gemini_key)

model_judge = genai.GenerativeModel("gemini-2.5-flash")

def lave_score(question, ground_truth, prediction):
    prompt = f"""You are evaluating a Visual Question Answering model on Assamese cultural instruments.

        Question: {question}
        Reference Answer: {ground_truth}
        Model Prediction: {prediction}

        Rate the factual correctness of the model prediction compared to the reference answer.
        Consider semantic equivalence, not exact wording.
        A score of 1.0 means fully correct, 0.0 means completely wrong.
        Return ONLY a number between 0 and 1. Nothing else."""

    try:
        response = model_judge.generate_content(prompt)
        score = float(response.text.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.0

# Score all predictions (with rate limit handling)
for i, p in enumerate(predictions):
    p["lave"] = lave_score(p["question"], p["ground_truth"], p["prediction"])
    time.sleep(1.5)  # free tier rate limit is stricter, ~4-5 req/sec cap but safer slow
    if i % 10 == 0:
        print(f"Progress: {i}/{len(predictions)}")

avg_lave = np.mean([p["lave"] for p in predictions])
print(f"\nOverall LAVE: {avg_lave:.4f}")

instrument_lave = defaultdict(list)
for p in predictions:
    instrument_lave[p["instrument"]].append(p["lave"])

print("\nPer Instrument LAVE:")
for inst, scores in instrument_lave.items():
    print(f"  {inst:15s}: {np.mean(scores):.4f}")

with open("results/lave_scores.json", "w") as f:
    json.dump(predictions, f, indent=2)for p in predictions[:10]:
    print(f"Q: {p['question']}")
    print(f"GT: {p['ground_truth']}")
    print(f"Pred: {p['prediction']}")
    print(f"LAVE: {p['lave']}")
    print("---")



# Step 10: Summarize Results
# Aggregates evaluation metrics by instrument, identifies the weakest question
# for each instrument, generates a final results table (CSV), and selects the
# best prediction examples for qualitative analysis and presentation.

import pandas as pd

instruments = sorted(set(
    p["instrument"]
    for p in predictions
))
rows = []

for inst in instruments:
    inst_preds = [p for p in predictions if p["instrument"] == inst]
    avg_cos  = np.mean([p["cosine_sim"] for p in inst_preds])
    avg_lave = np.mean([p["lave"] for p in inst_preds])
    
    # Find weakest question
    q_scores = defaultdict(list)
    for p in inst_preds:
        q_scores[p["question"]].append(p["cosine_sim"])
    weakest_q = min(q_scores, key=lambda q: np.mean(q_scores[q]))
    
    rows.append({
        "Instrument":     inst,
        "LAVE":           round(avg_lave, 4),
        "Cosine Sim":     round(avg_cos,  4),
        "Weakest Question": weakest_q[:40]
    })

# Add average row
rows.append({
    "Instrument":     "AVERAGE",
    "LAVE":           round(np.mean([p["lave"]       for p in predictions]), 4),
    "Cosine Sim":     round(np.mean([p["cosine_sim"] for p in predictions]), 4),
    "Weakest Question": "-"
})

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("results/final_results.csv", index=False) 

# Show 3 best examples  
import random

seen = set()
good_examples = []

for p in sorted(
    predictions,
    key=lambda x: x["lave"] + x["cosine_sim"],
    reverse=True
):
    if p["instrument"] not in seen:
        good_examples.append(p)
        seen.add(p["instrument"])

    if len(good_examples) == 3:
        break
        
for ex in good_examples:
    print(f"Instrument:   {ex['instrument']}")
    print(f"Question:     {ex['question']}")
    print(f"Ground Truth: {ex['ground_truth']}")
    print(f"Prediction:   {ex['prediction']}")
    print(f"LAVE:         {ex['lave']:.4f}")
    print(f"Cosine Sim:   {ex['cosine_sim']:.4f}")
    print("─" * 60)



# Step 11: Save & Publish Model
# Securely authenticates with GitHub and Hugging Face, uploads the notebook,
# results, and LoRA checkpoint to GitHub, and publishes the trained LoRA
# adapter to the Hugging Face Hub for sharing and reuse.

from kaggle_secrets import UserSecretsClient
import subprocess

# Get GitHub token securely from Kaggle Secrets
github_token = UserSecretsClient().get_secret("GITHUB_TOKEN")
github_user = "chair-amrit"
repo_name = "assamese-instrument-vlm"

# Set remote with token embedded (only in this session, not saved to notebook)
subprocess.run(
    f"git remote set-url origin https://{github_user}:{github_token}@github.com/{github_user}/{repo_name}.git",
    shell=True, cwd="/kaggle/working/assamese-instrument-vlm", check=True
)

%cd /kaggle/working/assamese-instrument-vlm

# Copy notebook itself (adjust filename to yours)
!cp /kaggle/working/*.ipynb .

# Copy results (already done, but re-run safe if you add more)
!mkdir -p results
!cp /kaggle/working/results/*.json results/
!cp /kaggle/working/results/*.csv results/
!cp /kaggle/working/loss_curve.png results/

# Copy best_checkpoint ONLY if small enough (LoRA adapters are usually a few MB - fine for GitHub)
!cp -r /kaggle/working/best_checkpoint .

!git add .
!git commit -m "Add notebook, results, and trained LoRA adapter"
!git push

f

rom huggingface_hub import login
from kaggle_secrets import UserSecretsClient

hf_write_token = UserSecretsClient().get_secret("HF-WRITE")
login(token = hf_write_token)#save model in hugging face
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./best_checkpoint",
    repo_id="IsHereAmrit/paligemma-assamese-instruments-qlora",
    repo_type="model"
)
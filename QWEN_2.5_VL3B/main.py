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




#Step 5 — Build Dataset Class
# Import required libraries
import os
import json
from PIL import Image
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

# Custom Dataset for Qwen2.5-VL
class QwenVQADataset(Dataset):

    # Initialize dataset
    def __init__(self,jsonl_path,dataset_root):
        self.dataset_root=dataset_root
        self.samples=[]

        with open(jsonl_path,"r",encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    # Return total number of samples
    def __len__(self):
        return len(self.samples)

    # Return one sample
    def __getitem__(self,idx):

        sample=self.samples[idx]

        messages=sample["messages"]

        image_path=os.path.join(
            self.dataset_root,
            messages[1]["content"][0]["image"]
        )

        image=Image.open(image_path).convert("RGB")

        # Update image path inside the conversation
        messages[1]["content"][0]["image"]=image_path

        image_inputs,video_inputs=process_vision_info(messages)

        return{
            "messages":messages,
            "image":image,
            "image_inputs":image_inputs,
            "video_inputs":video_inputs
        }

# Create dataset objects
train_dataset=QwenVQADataset(TRAIN_JSON,DATASET_ROOT)
val_dataset=QwenVQADataset(VAL_JSON,DATASET_ROOT)
test_dataset=QwenVQADataset(TEST_JSON,DATASET_ROOT)

# Verify dataset
print(f"Train Samples : {len(train_dataset)}")
print(f"Validation Samples : {len(val_dataset)}")
print(f"Test Samples : {len(test_dataset)}")

sample=train_dataset[0]

print(f"Loaded Image Size : {sample['image'].size}")
print(f"Number of Messages : {len(sample['messages'])}")
print(f"Vision Inputs Created : {sample['image_inputs'] is not None}")






#Step 6 — Build Data Collator
# Import required library
import torch

# Custom data collator for Qwen2.5-VL
class QwenDataCollator:

    # Initialize collator
    def __init__(self,processor):
        self.processor=processor

    # Build one training batch
    def __call__(self,batch):

        # Extract conversations
        conversations=[sample["messages"] for sample in batch]

        # Extract images
        images=[sample["image"] for sample in batch]

        # Apply Qwen chat template
        texts=[
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            for conversation in conversations
        ]

        # Tokenize text and images
        model_inputs=self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )

        # Create labels from input ids
        labels=model_inputs["input_ids"].clone()

        # Ignore padding tokens
        labels[labels==self.processor.tokenizer.pad_token_id]=-100

        # Ignore image tokens
        if hasattr(self.processor,"image_token_id"):
            labels[labels==self.processor.image_token_id]=-100

        # Return training batch
        model_inputs["labels"]=labels

        return model_inputs

# Create collator
data_collator=QwenDataCollator(processor)

# Verify collator
batch=data_collator([
    train_dataset[0],
    train_dataset[1]
])

print(batch.keys())

print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
print(batch["pixel_values"].shape)
print(batch["labels"].shape)





#Step 7 — Configure QLoRA
# Import PEFT utilities
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training

# Prepare model for k-bit training
model=prepare_model_for_kbit_training(model)

# Freeze all vision encoder parameters
for name,param in model.named_parameters():
    if name.startswith("model.visual."):
        param.requires_grad=False

vision_trainable=sum(
    p.numel()
    for n,p in model.named_parameters()
    if n.startswith("model.visual.") and p.requires_grad
)

print(f"Vision trainable parameters: {vision_trainable}")

# Configure LoRA
lora_config=LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj"
    ]
)

# Attach LoRA adapters
model=get_peft_model(model,lora_config)

# Print trainable parameter statistics
model.print_trainable_parameters()

# Verify vision encoder is frozen
vision_trainable=sum(
    p.numel()
    for n,p in model.named_parameters()
    if n.startswith("model.visual.") and p.requires_grad
)

language_trainable=sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Vision Trainable Parameters : {vision_trainable:,}")
print(f"Total Trainable Parameters : {language_trainable:,}")





#Step 8 — Sanity Check
# Import required libraries
from torch.utils.data import DataLoader
import torch

# Create training dataloader
train_loader=DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=data_collator
)

# Create optimizer
optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=2e-4
)

# Enable training mode
model.train()
model.gradient_checkpointing_enable()

# Get one training batch
batch=next(iter(train_loader))

# Move tensors to GPU
batch={k:v.to(model.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}

# Clear previous gradients
optimizer.zero_grad()

# Forward pass
outputs=model(**batch)

# Compute loss
loss=outputs.loss

# Verify loss
assert torch.isfinite(loss),"Loss is NaN or Inf."

print(f"Initial Loss : {loss.item():.4f}")

# Backward pass
loss.backward()

# Verify gradients
grad_found=False

for name,param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad_found=True
        break

assert grad_found,"No gradients were computed."

print("Backward pass successful.")

# Optimizer step
optimizer.step()

# Clear gradients
optimizer.zero_grad()

# Display GPU memory usage
allocated=torch.cuda.memory_allocated()/1024**3
reserved=torch.cuda.memory_reserved()/1024**3

print(f"GPU Memory Allocated : {allocated:.2f} GB")
print(f"GPU Memory Reserved : {reserved:.2f} GB")

print("Sanity check completed successfully.")





#Step 9 — Fine-tune
# Import required libraries
from transformers import TrainingArguments,Trainer,EarlyStoppingCallback

print("train args start")

# Configure training arguments
training_args=TrainingArguments(
    output_dir="./qwen2.5vl_lora_output",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    do_train=True,
    do_eval=True,
    use_cache=False,
    logging_strategy="steps",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    max_grad_norm=1.0
)

model.config.use_cache=False
model.enable_input_require_grads()

print("train args end")

print("trainer start")
# Create trainer
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=processor,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)

print(len(train_dataset),len(val_dataset))
batch=data_collator([train_dataset[0]])
print(batch.keys())

print("trainer end")

print("training start")
# Start fine-tuning
trainer.train()
print("training end")

print("eval best checkpoint start")

# Evaluate best checkpoint on validation set
validation_metrics=trainer.evaluate()

# Display validation metrics
print(validation_metrics)

print("eval best checkpoint end")





#Step 10 — Save Model
# Import required libraries
import os
import json
import torch
import shutil
from transformers import TrainerState

# Create output directory
SAVE_DIR="/kaggle/working/qwen2.5vl_lora"
os.makedirs(SAVE_DIR,exist_ok=True)

# Save LoRA adapter
model.save_pretrained(os.path.join(SAVE_DIR,"adapter"))

# Save processor
processor.save_pretrained(os.path.join(SAVE_DIR,"processor"))

# Save tokenizer
processor.tokenizer.save_pretrained(os.path.join(SAVE_DIR,"tokenizer"))

# Save training arguments
torch.save(
    trainer.args,
    os.path.join(SAVE_DIR, "training_args.pt")
)

# Save trainer state
trainer.state.save_to_json(os.path.join(SAVE_DIR,"trainer_state.json"))

# Save evaluation metrics
validation_metrics=trainer.evaluate()

with open(os.path.join(SAVE_DIR,"validation_metrics.json"),"w") as f:
    json.dump(validation_metrics,f,indent=4)

# Save training log history
with open(os.path.join(SAVE_DIR,"log_history.json"),"w") as f:
    json.dump(trainer.state.log_history,f,indent=4)

# Create compressed backup
shutil.make_archive(
    base_name="/kaggle/working/qwen2.5vl_lora_backup",
    format="zip",
    root_dir=SAVE_DIR
)

print(f"Model saved to: {SAVE_DIR}")
print("Backup ZIP created successfully.")





#Step 11 — Inference
# Import required libraries
import os
import copy
import json
import torch
from tqdm import tqdm
from peft import PeftModel
import pandas as pd
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# Define model and adapter paths
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH="/kaggle/working/qwen2.5vl_lora/adapter"
TEST_JSON="/kaggle/input/datasets/bbeastboy/assamese-instrument-vlm-datasetqwen2-5-vl-3b/kaggle_dataset32/test.jsonl"
OUTPUT_JSON="/kaggle/working/test_predictions.json"
FINAL_VQA_CSV="/kaggle/input/datasets/bbeastboy/assamese-instrument-vlm-datasetqwen2-5-vl-3b/kaggle_dataset32/final_vqa.csv"
IMAGE_ROOT="/kaggle/input/datasets/bbeastboy/assamese-instrument-vlm-datasetqwen2-5-vl-3b/kaggle_dataset32"

# Configure 4-bit inference
bnb_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load processor
processor=AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    min_pixels=224*28*28,
    max_pixels=384*28*28
)

# Load base model
base_model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"":0},
    trust_remote_code=True
)

# Attach LoRA adapter
model=PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

# Switch to evaluation mode
model.eval()

# Load test samples
test_samples=[]

with open(TEST_JSON,"r",encoding="utf-8") as f:
    for line in f:
        test_samples.append(json.loads(line))

# Load final_vqa.csv metadata
metadata_df=pd.read_csv(FINAL_VQA_CSV)

# Keep only test rows
metadata_df=metadata_df[metadata_df["split"]=="test"]

# Build lookup table
metadata_lookup={}

for _,row in metadata_df.iterrows():

    key=(
        row["image_name"],
        row["question"]
    )

    metadata_lookup[key]={
        "instrument":row["instrument"],
        "question_id":row["question_id"],
        "concept":row["concept"]
    }

predictions=[]

# Run inference
with torch.inference_mode():
    
    for sample in tqdm(test_samples):

        # Create a copy so the original JSON is unchanged
        messages=copy.deepcopy(sample["messages"])

        # Convert relative image path to absolute Kaggle path
        messages[1]["content"][0]["image"]=os.path.join(
            IMAGE_ROOT,
            messages[1]["content"][0]["image"]
        )

        image_inputs,video_inputs=process_vision_info(messages)

        prompt=processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs=processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        generated_ids=model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

        generated_ids=generated_ids[:,inputs.input_ids.shape[1]:]

        prediction=processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # Extract sample information
        image_path=messages[1]["content"][0]["image"]

        image_name=os.path.basename(image_path)

        question=messages[1]["content"][1]["text"]

        ground_truth=messages[2]["content"][0]["text"]

        # Get metadata from final_vqa.csv
        metadata=metadata_lookup[(image_name,question)]

        predictions.append({
            "image":image_path,
            "instrument":metadata["instrument"],
            "question_id":metadata["question_id"],
            "concept":metadata["concept"],
            "question":question,
            "ground_truth":ground_truth,
            "prediction":prediction.strip()
        })

# Save predictions
with open(OUTPUT_JSON,"w",encoding="utf-8") as f:
    json.dump(predictions,f,indent=4,ensure_ascii=False)

print(f"Saved {len(predictions)} predictions to {OUTPUT_JSON}")





#Step 12 — LAVE Evaluation
# Import required libraries
import os
import json
import time
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GOOGLE_API_KEY")


# Configure Gemini API
genai.configure(api_key=secret_value_0)
judge_model=genai.GenerativeModel("gemini-3.1-flash-lite")

# Define file paths
PREDICTIONS_JSON="/kaggle/working/test_predictions.json"
OUTPUT_JSON="/kaggle/working/lave_results.json"

# Load prediction file
with open(PREDICTIONS_JSON,"r",encoding="utf-8") as f:
    predictions=json.load(f)

# Store evaluation results
results=[]

# Evaluate every prediction
for sample in tqdm(predictions):

    prompt=f"""
You are an expert evaluator for Visual Question Answering.

Compare the predicted answer with the reference answer based on factual correctness, semantic similarity, and completeness.

Question:
{sample['question']}

Ground Truth:
{sample['ground_truth']}

Predicted Answer:
{sample['prediction']}

Return ONLY valid JSON in this exact format:

{{
  "score": 0.95,
  "reason": "Short explanation."
}}

Rules:
- score must be between 0.0 and 1.0
- 1.0 = perfectly correct
- 0.8 = mostly correct
- 0.5 = partially correct
- 0.2 = mostly incorrect
- 0.0 = completely incorrect

Do not output anything except the JSON object.
"""

    while True:

        try:

            response=judge_model.generate_content(prompt)

            text=response.text.strip()

            if text.startswith("```"):
                text=text.replace("```json","").replace("```","").strip()

            result=json.loads(text)

            score=float(result["score"])

            score=max(0.0,min(1.0,score))

            reason=result["reason"]

            break

        except Exception as e:

            if "429" in str(e):

                print("Rate limit reached. Waiting 35 seconds...")

                time.sleep(35)

                continue

            score=None

            reason=str(e)

            break

    if score is None:
        continue

    results.append({
        "instrument":sample["instrument"],
        "question_id":sample["question_id"],
        "concept":sample["concept"],
        "question":sample["question"],
        "ground_truth":sample["ground_truth"],
        "prediction":sample["prediction"],
        "lave_score":score,
        "reason":reason
    })

    time.sleep(5)

# Save detailed LAVE results
with open(OUTPUT_JSON,"w",encoding="utf-8") as f:
    json.dump(results,f,indent=4,ensure_ascii=False)

# Create dataframe
df=pd.DataFrame(results)

# Compute overall LAVE
overall_lave=df["lave_score"].mean()

# Compute instrument-wise LAVE
instrument_lave=df.groupby("instrument")["lave_score"].mean().sort_values(ascending=False)

# Compute concept-wise LAVE
concept_lave=df.groupby("concept")["lave_score"].mean().sort_values(ascending=False)

# Display results
print(f"Overall LAVE Score : {overall_lave:.4f}")

print("\nInstrument-wise LAVE")
print(instrument_lave)

print("\nConcept-wise LAVE")
print(concept_lave)
"""
Training configuration.
"""

# Output
OUTPUT_DIR = "./qwen2.5vl_lora_output"

# Training
NUM_EPOCHS = 5

TRAIN_BATCH_SIZE = 1

EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.01

MAX_GRAD_NORM = 1.0

# Scheduler
LR_SCHEDULER = "cosine"

WARMUP_STEPS = 50

# Mixed Precision
FP16 = True

GRADIENT_CHECKPOINTING = True

USE_CACHE = False

# Evaluation
DO_TRAIN = True

DO_EVAL = True

EVAL_STRATEGY = "epoch"

SAVE_STRATEGY = "epoch"

SAVE_TOTAL_LIMIT = 2

LOAD_BEST_MODEL = True

BEST_MODEL_METRIC = "eval_loss"

GREATER_IS_BETTER = False

# Logging
LOGGING_STRATEGY = "steps"

LOGGING_STEPS = 10

REPORT_TO = "none"

# DataLoader
NUM_WORKERS = 2

PIN_MEMORY = True

REMOVE_UNUSED_COLUMNS = False

# Early Stopping
EARLY_STOPPING_PATIENCE = 2

# Reproducibility
SEED = 42
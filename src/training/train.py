"""
Training pipeline for QLoRA fine-tuning of Qwen2.5-VL-3B.
"""

import torch

from torch.utils.data import DataLoader

from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from configs.lora_config import lora_config


def prepare_qlora_model(model):
    """Prepare the quantized model and attach LoRA adapters."""

    from peft import (
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    model = prepare_model_for_kbit_training(model)

    # Freeze all vision encoder parameters
    for name, param in model.named_parameters():
        if name.startswith("model.visual."):
            param.requires_grad = False

    model = get_peft_model(model, lora_config)

    # Verify vision encoder is frozen
    vision_trainable = sum(
        p.numel()
        for n, p in model.named_parameters()
        if n.startswith("model.visual.") and p.requires_grad
    )

    language_trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Vision Trainable Parameters : {vision_trainable:,}")
    print(f"Total Trainable Parameters : {language_trainable:,}")

    model.print_trainable_parameters()

    return model


def sanity_check(model, train_dataset, data_collator):
    """Run one forward/backward pass before full training."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=data_collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
    )

    model.train()
    model.gradient_checkpointing_enable()

    batch = next(iter(train_loader))

    batch = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    optimizer.zero_grad()

    outputs = model(**batch)

    loss = outputs.loss

    assert torch.isfinite(loss), "Loss is NaN or Inf."

    print(f"Initial Loss : {loss.item():.4f}")

    loss.backward()

    grad_found = False

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_found = True
            break

    assert grad_found, "No gradients were computed."

    print("Backward pass successful.")

    optimizer.step()
    optimizer.zero_grad()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"GPU Memory Allocated : {allocated:.2f} GB")
    print(f"GPU Memory Reserved : {reserved:.2f} GB")

    print("Sanity check completed successfully.")


def create_training_arguments():
    """Create the TrainingArguments used in the master notebook."""

    training_args = TrainingArguments(
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
        max_grad_norm=1.0,
    )

    return training_args


def create_trainer(
    model,
    training_args,
    train_dataset,
    val_dataset,
    data_collator,
    processor,
):
    """Create the Hugging Face Trainer."""

    trainer = Trainer(
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
        ],
    )

    return trainer


def train_model(
    model,
    train_dataset,
    val_dataset,
    data_collator,
    processor,
):
    """Prepare the model and run QLoRA fine-tuning."""

    model = prepare_qlora_model(model)

    sanity_check(
        model,
        train_dataset,
        data_collator,
    )

    training_args = create_training_arguments()

    model.config.use_cache = False
    model.enable_input_require_grads()

    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        data_collator=data_collator,
        processor=processor,
    )

    print(len(train_dataset), len(val_dataset))

    batch = data_collator([train_dataset[0]])
    print(batch.keys())

    trainer.train()

    return model, trainer
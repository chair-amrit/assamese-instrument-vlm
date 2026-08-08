"""
Inference utilities for Qwen2.5-VL-3B on the Assamese Instrument VQA dataset.
"""

import torch
from PIL import Image


def generate_answer(
    model,
    processor,
    image_path,
    question,
    max_new_tokens=128,
):
    """
    Generate an answer for a single image-question pair.
    """

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    model.eval()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    input_ids_length = inputs["input_ids"].shape[1]

    generated_ids = generated_ids[:, input_ids_length:]

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return answer.strip()


def run_inference(
    model,
    processor,
    test_dataset,
    image_root,
    max_new_tokens=128,
):
    """
    Run inference over the test dataset.
    """

    predictions = []

    model.eval()

    for sample in test_dataset:

        image_path = image_root / sample["image"]

        question = sample["question"]

        answer = generate_answer(
            model=model,
            processor=processor,
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
        )

        predictions.append(
            {
                "image": sample["image"],
                "question": question,
                "ground_truth": sample["answer"],
                "prediction": answer,
            }
        )

    return predictions
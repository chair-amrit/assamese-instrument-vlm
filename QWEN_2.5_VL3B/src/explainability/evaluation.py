"""
Aggregation and validation for cross-modal explainability results.

Combines per-sample attention/attribution results (Steps 13-15) into
per-failure-category and per-instrument summaries, and provides the
Integrated Gradients stability validation (n_steps sensitivity check).
"""

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients


FAILURE_CATEGORIES = [
    "Correct",
    "Question Misunderstanding",
    "Hallucination",
    "Partial Answer / Incomplete Answer",
    "Truncation",
    "Repetition",
    "Mixed Attribute",
]


def load_results(attention_path, attribution_path):
    """Load and filter out failed samples from both result JSON files."""
    with open(attention_path) as f:
        attention_results = json.load(f)
    with open(attribution_path) as f:
        attribution_results = json.load(f)

    attention_results = [r for r in attention_results if "error" not in r]
    attribution_results = [r for r in attribution_results if "error" not in r]

    print(f"Attention results: {len(attention_results)}")
    print(f"Attribution results: {len(attribution_results)}")

    return attention_results, attribution_results


def aggregate_by_category(attention_results, attribution_results,
                           categories=FAILURE_CATEGORIES, save_path=None):
    """
    Compute average attention and attribution coverage per failure category.

    Returns:
        pd.DataFrame with columns [category, n_samples, avg_attention_coverage,
        avg_attribution_coverage]
    """
    category_stats = []
    for cat in categories:
        attn_cat = [r for r in attention_results if r["failure_category"] == cat]
        attr_cat = [r for r in attribution_results if r["failure_category"] == cat]

        if not attn_cat:
            continue

        avg_coverage = np.mean([r["image_coverage_ratio"] for r in attn_cat])
        avg_attr_coverage = (
            np.mean([r["attribution_coverage_ratio"] for r in attr_cat])
            if attr_cat else 0
        )

        category_stats.append({
            "category": cat,
            "n_samples": len(attn_cat),
            "avg_attention_coverage": round(avg_coverage, 4),
            "avg_attribution_coverage": round(avg_attr_coverage, 4),
        })

    df = pd.DataFrame(category_stats)
    print(df.to_string(index=False))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    return df


def aggregate_by_instrument(attention_results, save_path=None):
    """
    Compute average attention coverage and failure-category distribution
    per instrument.

    Returns:
        pd.DataFrame with columns [instrument, n_samples, avg_attention_coverage,
        failure_distribution]
    """
    instruments = sorted(set(r["instrument"] for r in attention_results))
    instrument_stats = []

    for inst in instruments:
        attn_inst = [r for r in attention_results if r["instrument"] == inst]
        avg_coverage = np.mean([r["image_coverage_ratio"] for r in attn_inst])

        cat_dist = defaultdict(int)
        for r in attn_inst:
            cat_dist[r["failure_category"]] += 1

        instrument_stats.append({
            "instrument": inst,
            "n_samples": len(attn_inst),
            "avg_attention_coverage": round(avg_coverage, 4),
            "failure_distribution": dict(cat_dist),
        })

    df = pd.DataFrame(instrument_stats)
    print(df.to_string(index=False))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    return df


def run_stability_validation(model, processor, samples, test_dir,
                              n_steps_baseline=10, n_steps_reference=50,
                              stability_threshold=0.9, save_path=None):
    """
    Validate Integrated Gradients stability by comparing attribution maps
    computed at a low n_steps (baseline) vs a high n_steps (reference) on
    one representative sample per failure category, using cosine similarity.

    This is a required check before trusting attribution results - IG
    accuracy is sensitive to interpolation step count, and low step counts
    can produce unstable, unreliable attribution maps (see docs/protocol.md).

    Returns:
        list of dicts with per-sample cosine similarity and stability verdict
    """
    import gc
    from qwen_vl_utils import process_vision_info

    validation_samples = samples.groupby("failure_category").first().reset_index()
    print(f"Validating on {len(validation_samples)} samples")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for param in model.parameters():
        param.requires_grad_(False)

    stability_results = []

    for idx, sample in validation_samples.iterrows():
        print(f"\n[{idx+1}/{len(validation_samples)}] {sample['instrument']} — {sample['failure_category']}")

        image_path = os.path.join(test_dir, sample["instrument"], sample["image"])
        question = sample["question"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                            padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        gen_ids = gen_ids[:, inputs.input_ids.shape[1]:]
        prediction = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        target_token_id = processor.tokenizer(prediction, add_special_tokens=False).input_ids[0]

        def forward_fn(pixel_values):
            outputs = model(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                image_grid_thw=inputs.image_grid_thw, pixel_values=pixel_values,
                return_dict=True,
            )
            return outputs.logits[:, -1, target_token_id]

        pixel_values = inputs.pixel_values.clone().detach().requires_grad_(True)
        ig = IntegratedGradients(forward_fn)

        attr_baseline = ig.attribute(pixel_values, baselines=torch.zeros_like(pixel_values),
                                      n_steps=n_steps_baseline, internal_batch_size=1)
        map_baseline = attr_baseline.squeeze(0).abs().sum(dim=-1).detach().cpu().float().numpy()

        del attr_baseline
        gc.collect()
        torch.cuda.empty_cache()

        attr_reference = ig.attribute(pixel_values, baselines=torch.zeros_like(pixel_values),
                                       n_steps=n_steps_reference, internal_batch_size=1)
        map_reference = attr_reference.squeeze(0).abs().sum(dim=-1).detach().cpu().float().numpy()

        cos_sim = np.dot(map_baseline, map_reference) / (
            np.linalg.norm(map_baseline) * np.linalg.norm(map_reference) + 1e-8
        )

        stability_results.append({
            "image": sample["image"],
            "instrument": sample["instrument"],
            "failure_category": sample["failure_category"],
            f"cosine_similarity_{n_steps_baseline}_vs_{n_steps_reference}": float(cos_sim),
            "stable": bool(cos_sim > stability_threshold),
        })
        verdict = "STABLE" if cos_sim > stability_threshold else "UNSTABLE"
        print(f"  Cosine similarity (n={n_steps_baseline} vs n={n_steps_reference}): {cos_sim:.4f} — {verdict}")

        del attr_reference, pixel_values
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Stability Validation Summary")
    print("=" * 60)
    sim_key = f"cosine_similarity_{n_steps_baseline}_vs_{n_steps_reference}"
    for r in stability_results:
        print(f"{r['instrument']:12s} {r['failure_category']:30s} "
              f"cos_sim={r[sim_key]:.4f}  {'✓ STABLE' if r['stable'] else '✗ UNSTABLE'}")

    avg_stability = np.mean([r[sim_key] for r in stability_results])
    print(f"\nAverage stability: {avg_stability:.4f}")
    print(f"Verdict: {'n_steps=' + str(n_steps_baseline) + ' is SUFFICIENT' if avg_stability > stability_threshold else f'n_steps={n_steps_reference} recommended as final'}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(stability_results, f, indent=2)

    return stability_results
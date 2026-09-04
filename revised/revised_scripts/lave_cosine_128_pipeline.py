"""
Final evaluation script: Binary LAVE + Cosine Similarity
Assamese Instrument VLM — 128-token revised predictions (315 samples)
"""

import os
import json
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(r"D:\InternshipGU\Assamese_instrument_VLM\revised\128_tokens")
INPUT_JSON = BASE_DIR / "revised_test_predictions.json"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"

REQUIRED_FIELDS = ["instrument", "question_id", "concept", "question", "ground_truth", "prediction"]
EXPECTED_SAMPLE_COUNT = 315


# ---------------------------------------------------------------------------
# Step 0 — Data integrity checks
# ---------------------------------------------------------------------------

def load_and_validate_predictions():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSON}")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if not isinstance(predictions, list):
        raise ValueError("Input JSON must be a list of samples.")

    if len(predictions) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} samples, found {len(predictions)}. "
            f"Aborting — this is the final evaluation set."
        )

    for i, sample in enumerate(predictions):
        missing = [f for f in REQUIRED_FIELDS if f not in sample]
        if missing:
            raise ValueError(f"Sample {i} is missing required fields: {missing}")

    for i, sample in enumerate(predictions):
        sample["sample_id"] = i

    print(f"Loaded and validated {len(predictions)} samples from {INPUT_JSON}")
    return predictions


def check_environment():
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. Set it in your .env file as GOOGLE_API_KEY=<your_key>"
        )
    if not MODEL_NAME:
        raise EnvironmentError("MODEL_NAME is empty. Set the Gemini model name before running.")
    print(f"Gemini configured with model: {MODEL_NAME}")


# ---------------------------------------------------------------------------
# Step 1 — Binary LAVE evaluation
# ---------------------------------------------------------------------------

LAVE_PROMPT_TEMPLATE = """You are an expert evaluator for Visual Question Answering.

Judge ONLY the prediction actually given below. Do not invent, assume, or add
information that is not present in the prediction.

Question: {question}
Ground Truth: {ground_truth}
Predicted Answer: {prediction}

Decide whether the prediction is correct and semantically aligned with the
question and ground truth.

Return ONLY valid JSON in this exact format:
{{
  "score": 1,
  "reason": "Short explanation."
}}

Rules:
- score must be exactly 1 or 0 (integer, no other values)
- 1 = prediction is correct and semantically aligned with the question and ground truth
- 0 = prediction is incorrect, answers a different concept/question, or is not
      sufficiently correct
- Do not output anything except the JSON object.
"""


def run_lave_evaluation(predictions, client, max_retries=6):
    results = []

    for sample in tqdm(predictions, desc="Running LAVE"):
        prompt = LAVE_PROMPT_TEMPLATE.format(
            question=sample["question"],
            ground_truth=sample["ground_truth"],
            prediction=sample["prediction"],
        )

        score, reason = None, None
        attempt = 0

        while attempt < max_retries:
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                    ),
                )
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.replace("```json", "").replace("```", "").strip()

                result = json.loads(text)
                score = int(result["score"])
                if score not in (0, 1):
                    raise ValueError(f"Invalid score returned: {score}")
                reason = result.get("reason", "")
                break

            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower() or "quota" in err.lower():
                    wait = 35
                    print(f"Rate limit hit. Waiting {wait}s...")
                    time.sleep(wait)
                    attempt += 1
                    continue
                elif attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"Error: {err}. Retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    attempt += 1
                    continue
                else:
                    raise RuntimeError(
                        f"LAVE evaluation failed after {max_retries} attempts for sample "
                        f"(instrument={sample['instrument']}, question_id={sample['question_id']}): {err}"
                    )

        results.append({
            "sample_id": sample["sample_id"],
            "instrument": sample["instrument"],
            "question_id": sample["question_id"],
            "concept": sample["concept"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "prediction": sample["prediction"],
            "lave_score": score,
            "reason": reason,
        })

        time.sleep(5)

    if len(results) != len(predictions):
        raise RuntimeError(
            f"LAVE result count mismatch: got {len(results)}, expected {len(predictions)}"
        )

    return results


# ---------------------------------------------------------------------------
# Step 2 — Cosine similarity evaluation
# ---------------------------------------------------------------------------

def run_cosine_evaluation(predictions):
    print("Loading sentence embedding model (all-mpnet-base-v2)...")
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    ground_truth_list = [s["ground_truth"] for s in predictions]
    prediction_list = [s["prediction"] for s in predictions]

    gt_embeddings = embedding_model.encode(
        ground_truth_list, batch_size=32, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    pred_embeddings = embedding_model.encode(
        prediction_list, batch_size=32, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=True,
    )

    results = []
    for sample, gt_emb, pred_emb in zip(predictions, gt_embeddings, pred_embeddings):
        cosine_score = cos_sim(gt_emb, pred_emb).item()
        cosine_score = max(0.0, min(1.0, cosine_score))

        results.append({
            "sample_id": sample["sample_id"],
            "instrument": sample["instrument"],
            "question_id": sample["question_id"],
            "concept": sample["concept"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "prediction": sample["prediction"],
            "cosine_similarity": cosine_score,
        })

    if len(results) != len(predictions):
        raise RuntimeError(
            f"Cosine result count mismatch: got {len(results)}, expected {len(predictions)}"
        )

    return results


# ---------------------------------------------------------------------------
# Step 3 — Merge, verify, save
# ---------------------------------------------------------------------------

def merge_results(lave_results, cosine_results):
    lave_df = pd.DataFrame(lave_results)
    cosine_df = pd.DataFrame(cosine_results)

    merged = pd.merge(
        lave_df, cosine_df, on="sample_id", how="inner", suffixes=("", "_cosine_dup")
    )
    dup_cols = [c for c in merged.columns if c.endswith("_cosine_dup")]
    merged = merged.drop(columns=dup_cols)

    if len(merged) != len(lave_df) or len(merged) != len(cosine_df):
        raise RuntimeError(
            f"Merge integrity check failed: lave={len(lave_df)}, "
            f"cosine={len(cosine_df)}, merged={len(merged)}"
        )

    print(f"Merge verified: {len(merged)} samples aligned across LAVE and cosine results.")
    return lave_df, cosine_df, merged


def save_outputs(lave_results, cosine_results, lave_df, cosine_df, merged_df):
    (BASE_DIR).mkdir(parents=True, exist_ok=True)

    with open(BASE_DIR / "lave_results.json", "w", encoding="utf-8") as f:
        json.dump(lave_results, f, indent=4, ensure_ascii=False)
    lave_df.to_csv(BASE_DIR / "lave_results.csv", index=False, encoding="utf-8-sig")

    with open(BASE_DIR / "cosine_results.json", "w", encoding="utf-8") as f:
        json.dump(cosine_results, f, indent=4, ensure_ascii=False)
    cosine_df.to_csv(BASE_DIR / "cosine_results.csv", index=False, encoding="utf-8-sig")

    print("Saved: lave_results.json/csv, cosine_results.json/csv")
    return merged_df


# ---------------------------------------------------------------------------
# Step 4 — Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(merged_df):
    overall_lave = merged_df["lave_score"].mean()
    overall_cosine = merged_df["cosine_similarity"].mean()

    summary_df = pd.DataFrame([{
        "overall_lave_accuracy": round(overall_lave, 4),
        "overall_cosine_similarity": round(overall_cosine, 4),
        "n_samples": len(merged_df),
    }])
    summary_df.to_csv(BASE_DIR / "evaluation_summary.csv", index=False)

    instrument_metrics = merged_df.groupby("instrument").agg(
        lave_accuracy=("lave_score", "mean"),
        cosine_similarity=("cosine_similarity", "mean"),
        n_samples=("lave_score", "count"),
    ).round(4).sort_values("lave_accuracy", ascending=False)
    instrument_metrics.to_csv(BASE_DIR / "instrument_metrics.csv")

    concept_metrics = merged_df.groupby("concept").agg(
        lave_accuracy=("lave_score", "mean"),
        cosine_similarity=("cosine_similarity", "mean"),
        n_samples=("lave_score", "count"),
    ).round(4).sort_values("lave_accuracy", ascending=False)
    concept_metrics.to_csv(BASE_DIR / "concept_metrics.csv")

    print("\n" + "=" * 60)
    print(f"Overall LAVE Accuracy      : {overall_lave:.4f}")
    print(f"Overall Cosine Similarity  : {overall_cosine:.4f}")
    print("=" * 60)
    print("\nInstrument-wise metrics")
    print(instrument_metrics.to_string())
    print("\nConcept-wise metrics")
    print(concept_metrics.to_string())

    return summary_df, instrument_metrics, concept_metrics


# ---------------------------------------------------------------------------
# Step 5 — Figures
# ---------------------------------------------------------------------------

def make_figures(summary_df, instrument_metrics, concept_metrics):
    plt.rcParams.update({"font.size": 11})

    # Overall comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    metrics = ["overall_lave_accuracy", "overall_cosine_similarity"]
    labels = ["LAVE Accuracy", "Cosine Similarity"]
    values = [summary_df[m].iloc[0] for m in metrics]
    bars = ax.bar(labels, values, color=["steelblue", "coral"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "overall_metric_comparison.png", dpi=300)
    plt.close()

    # Instrument-wise
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(instrument_metrics))
    width = 0.35
    ax.bar([i - width / 2 for i in x], instrument_metrics["lave_accuracy"],
           width, label="LAVE Accuracy", color="steelblue")
    ax.bar([i + width / 2 for i in x], instrument_metrics["cosine_similarity"],
           width, label="Cosine Similarity", color="coral")
    ax.set_xticks(list(x))
    ax.set_xticklabels(instrument_metrics.index, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Instrument-wise Evaluation Metrics")
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "instrument_metric_comparison.png", dpi=300)
    plt.close()

    # Concept-wise
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(concept_metrics))
    ax.bar([i - width / 2 for i in x], concept_metrics["lave_accuracy"],
           width, label="LAVE Accuracy", color="steelblue")
    ax.bar([i + width / 2 for i in x], concept_metrics["cosine_similarity"],
           width, label="Cosine Similarity", color="coral")
    ax.set_xticks(list(x))
    ax.set_xticklabels(concept_metrics.index, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Concept-wise Evaluation Metrics")
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "concept_metric_comparison.png", dpi=300)
    plt.close()

    print("\nSaved: overall_metric_comparison.png, instrument_metric_comparison.png, "
          "concept_metric_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    check_environment()
    predictions = load_and_validate_predictions()

    client = genai.Client(api_key=GOOGLE_API_KEY)

    print("\n--- Running LAVE evaluation ---")
    lave_results = run_lave_evaluation(predictions, client)

    print("\n--- Running cosine similarity evaluation ---")
    cosine_results = run_cosine_evaluation(predictions)

    lave_df, cosine_df, merged_df = merge_results(lave_results, cosine_results)
    save_outputs(lave_results, cosine_results, lave_df, cosine_df, merged_df)

    summary_df, instrument_metrics, concept_metrics = compute_summary(merged_df)
    make_figures(summary_df, instrument_metrics, concept_metrics)

    print(f"\nAll outputs saved to: {BASE_DIR}")


if __name__ == "__main__":
    main()
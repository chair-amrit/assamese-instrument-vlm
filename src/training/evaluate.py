"""
Evaluation utilities for the fine-tuned Qwen2.5-VL model.

Implements:
1. LAVE evaluation using Gemini
2. Cosine similarity evaluation using Sentence-Transformers
3. Result aggregation by instrument and question concept
"""

import json
import time

import pandas as pd
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm


# =====================================================
# LAVE Evaluation
# =====================================================

def evaluate_lave(
    predictions_path,
    output_path,
    api_key,
):
    """
    Evaluate predictions using Gemini as an LLM judge.
    """

    genai.configure(api_key=api_key)

    judge_model = genai.GenerativeModel(
        "gemini-3.1-flash-lite"
    )

    with open(
        predictions_path,
        "r",
        encoding="utf-8",
    ) as f:
        predictions = json.load(f)

    results = []

    for sample in tqdm(predictions):

        prompt = f"""
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

                response = judge_model.generate_content(
                    prompt
                )

                text = response.text.strip()

                if text.startswith("```"):
                    text = (
                        text
                        .replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )

                result = json.loads(text)

                score = float(result["score"])

                score = max(
                    0.0,
                    min(1.0, score),
                )

                reason = result["reason"]

                break

            except Exception as e:

                if "429" in str(e):

                    print(
                        "Rate limit reached. "
                        "Waiting 35 seconds..."
                    )

                    time.sleep(35)

                    continue

                score = None
                reason = str(e)

                break

        if score is None:
            continue

        results.append(
            {
                "instrument": sample["instrument"],
                "question_id": sample["question_id"],
                "concept": sample["concept"],
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "prediction": sample["prediction"],
                "lave_score": score,
                "reason": reason,
            }
        )

        time.sleep(5)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
        )

    df = pd.DataFrame(results)

    overall_lave = df["lave_score"].mean()

    instrument_lave = (
        df.groupby("instrument")["lave_score"]
        .mean()
        .sort_values(ascending=False)
    )

    concept_lave = (
        df.groupby("concept")["lave_score"]
        .mean()
        .sort_values(ascending=False)
    )

    print(
        f"Overall LAVE Score : {overall_lave:.4f}"
    )

    print("\nInstrument-wise LAVE")
    print(instrument_lave)

    print("\nConcept-wise LAVE")
    print(concept_lave)

    return (
        df,
        overall_lave,
        instrument_lave,
        concept_lave,
    )


# =====================================================
# Cosine Similarity Evaluation
# =====================================================

def evaluate_cosine_similarity(
    predictions_path,
    output_path,
    csv_output_path=None,
):
    """
    Evaluate semantic similarity using
    Sentence-Transformer embeddings.
    """

    with open(
        predictions_path,
        "r",
        encoding="utf-8",
    ) as f:
        predictions = json.load(f)

    embedding_model = SentenceTransformer(
        "all-mpnet-base-v2"
    )

    ground_truth_list = [
        sample["ground_truth"]
        for sample in predictions
    ]

    prediction_list = [
        sample["prediction"]
        for sample in predictions
    ]

    gt_embeddings = embedding_model.encode(
        ground_truth_list,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    pred_embeddings = embedding_model.encode(
        prediction_list,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    results = []

    for (
        sample,
        gt_embedding,
        pred_embedding,
    ) in zip(
        predictions,
        gt_embeddings,
        pred_embeddings,
    ):

        cosine_score = cos_sim(
            gt_embedding,
            pred_embedding,
        ).item()

        cosine_score = max(
            0.0,
            min(1.0, cosine_score),
        )

        results.append(
            {
                "instrument": sample["instrument"],
                "question_id": sample["question_id"],
                "concept": sample["concept"],
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "prediction": sample["prediction"],
                "cosine_similarity": cosine_score,
            }
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
        )

    df = pd.DataFrame(results)

    if csv_output_path is not None:
        df.to_csv(
            csv_output_path,
            index=False,
        )

    overall_cosine = (
        df["cosine_similarity"].mean()
    )

    instrument_cosine = (
        df.groupby("instrument")[
            "cosine_similarity"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    concept_cosine = (
        df.groupby("concept")[
            "cosine_similarity"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    print(
        f"Overall Cosine Similarity : "
        f"{overall_cosine:.4f}"
    )

    print(
        "\nInstrument-wise Cosine Similarity"
    )
    print(instrument_cosine)

    print(
        "\nConcept-wise Cosine Similarity"
    )
    print(concept_cosine)

    return (
        df,
        overall_cosine,
        instrument_cosine,
        concept_cosine,
    )


# =====================================================
# Combined Evaluation
# =====================================================

def evaluate_model(
    predictions_path,
    lave_output_path,
    cosine_output_path,
    cosine_csv_path=None,
    api_key=None,
):
    """
    Run both LAVE and cosine-similarity evaluation.
    """

    lave_results = evaluate_lave(
        predictions_path=predictions_path,
        output_path=lave_output_path,
        api_key=api_key,
    )

    cosine_results = evaluate_cosine_similarity(
        predictions_path=predictions_path,
        output_path=cosine_output_path,
        csv_output_path=cosine_csv_path,
    )

    return {
        "lave": lave_results,
        "cosine": cosine_results,
    }
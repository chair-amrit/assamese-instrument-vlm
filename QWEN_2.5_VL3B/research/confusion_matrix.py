import os
import json
import time
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv

from google import genai
from google.genai import types

# LOAD ENVIRONMENT
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

MODEL = "gemini-3.1-flash-lite"

# FILE PATHS
INPUT_CSV = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\research\categorized_preds.csv"
OUTPUT_CSV = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\research\categorized_preds_with_attribute.csv"
CONFUSION_CSV = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\research\confusion_matrix.csv"
CONFUSION_NORM_CSV = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\research\confusion_matrix_normalized.csv"
CONFUSION_PNG = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\research\confusion_matrix.png"
CONFUSION_NORM_PNG = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\research\confusion_matrix_normalized.png"

# LOAD CSV
df = pd.read_csv(INPUT_CSV)

print("Total rows:", len(df))

# Only incorrect predictions need Gemini
wrong_df = df[df["failure_category"] != "Correct"].copy()

print("Incorrect rows:", len(wrong_df))

# New columns
wrong_df["predicted_attribute"] = ""
wrong_df["attribute_confidence"] = 0.0
wrong_df["attribute_reason"] = ""

def get_predicted_attribute(question, ground_truth, prediction):

    prompt = f"""
You are an expert researcher analyzing prediction errors in a Vision-Language Model (VLM) for Visual Question Answering (VQA).

Your task is NOT to judge whether the prediction is correct.

Instead, determine which ATTRIBUTE the prediction is primarily attempting to answer.

The allowed attributes are EXACTLY:

- festival
- origin
- material
- parts
- playing_method
- sound
- traditional_player
- instrument_type
- description
- none

Attribute definitions:

festival
The prediction discusses a festival or celebration where the instrument is played.

origin
The prediction discusses the geographical, historical, or cultural origin of the instrument.

material
The prediction describes the materials from which the instrument is made.

parts
The prediction describes the physical components or structure of the instrument.

playing_method
The prediction explains how the instrument is played.

sound
The prediction describes the sound produced by the instrument.

traditional_player
The prediction identifies who traditionally plays the instrument.

instrument_type
The prediction classifies the instrument (e.g., wind instrument, percussion instrument, idiophone).

description
The prediction provides a general description of the instrument rather than focusing on one specific attribute.

none
The prediction does not clearly express any of the above attributes.

Rules:

1. Consider BOTH the Question and the Prediction.
2. Ignore whether the prediction is factually correct.
3. Determine which attribute the prediction is primarily attempting to answer.
4. Return EXACTLY ONE label from the allowed list.
5. Do NOT invent new labels or modify the spelling.
6. If the prediction is truncated but it is still obvious which attribute it is attempting to answer, return that attribute.
7. If the prediction mainly gives a general explanation instead of answering a specific attribute, return "description".
8. If no attribute can reasonably be identified, return "none".
9. Return ONLY valid JSON.

Question:
{question}

Ground Truth:
{ground_truth}

Prediction:
{prediction}

Return ONLY this JSON format:

{{
  "predicted_attribute": "material",
  "confidence": 0.95,
  "reason": "The prediction primarily discusses the material."
}}
"""

    try:

        response = client.models.generate_content(

            model=MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json"
            )

        )

        response["predicted_attribute"] = response.get("predicted_attribute", "none").strip().lower()
        allowed = {
            "festival",
            "origin",
            "material",
            "parts",
            "playing_method",
            "sound",
            "traditional_player",
            "instrument_type",
            "description",
            "none"
        }

        if response["predicted_attribute"] not in allowed:
            response["predicted_attribute"] = "none"

        response.setdefault("predicted_attribute", "none")
        response.setdefault("confidence", 0.0)
        response.setdefault("reason", "")

        return response

    except Exception as e:

        return {

            "predicted_attribute": "none",

            "confidence": 0.0,

            "reason": str(e)

        }
    
for idx in tqdm(wrong_df.index):

    row = wrong_df.loc[idx]

    response = get_predicted_attribute(

        row["question"],
        row["ground_truth"],
        row["prediction"]

    )

    wrong_df.at[idx, "predicted_attribute"] = response["predicted_attribute"]

    wrong_df.at[idx, "attribute_confidence"] = response["confidence"]

    wrong_df.at[idx, "attribute_reason"] = response["reason"]

    time.sleep(4.5)


# Copy values back into the original dataframe
for col in [
    "predicted_attribute",
    "attribute_confidence",
    "attribute_reason"
]:
    df.loc[wrong_df.index, col] = wrong_df[col]

df.to_csv(

    OUTPUT_CSV,

    index=False,

    encoding="utf-8-sig"

)
print("Saved")
print(OUTPUT_CSV)

#create confusion matrix    
confusion = pd.crosstab(

    wrong_df["concept"],

    wrong_df["predicted_attribute"]

)
confusion.to_csv(CONFUSION_CSV)
print(confusion.to_string())

#normalize confusion matrix 
confusion_norm = pd.crosstab(

    wrong_df["concept"],

    wrong_df["predicted_attribute"],

    normalize="index"

)
confusion_norm = confusion_norm.round(3)
confusion_norm.to_csv(CONFUSION_NORM_CSV)

#heatmaps
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,7))
sns.heatmap(

    confusion,

    annot=True,

    fmt="d",

    cmap="Blues"

)
plt.title("Attribute Confusion Matrix")
plt.ylabel("Actual Attribute")
plt.xlabel("Predicted Attribute")
plt.tight_layout()
plt.savefig(CONFUSION_PNG,dpi=300)
plt.close()

plt.figure(figsize=(12,7))
sns.heatmap(

    confusion_norm,

    annot=True,

    fmt=".2f",

    cmap="Blues"

)
plt.title("Normalized Attribute Confusion Matrix")
plt.ylabel("Actual Attribute")
plt.xlabel("Predicted Attribute")
plt.tight_layout()
plt.savefig(CONFUSION_NORM_PNG,dpi=300)
plt.close()
print("Finished.")
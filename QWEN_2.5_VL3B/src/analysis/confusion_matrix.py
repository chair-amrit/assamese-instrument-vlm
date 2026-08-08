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
You are an expert evaluator performing attribute-level error analysis for a Vision-Language Model (VLM).

Your task is to identify the PRIMARY ATTRIBUTE expressed in the prediction.

Inputs:
- Question: identifies the intended attribute.
- Ground Truth: serves only as a semantic reference for what that attribute means. Do NOT compare it with the prediction to judge correctness.
- Prediction: this is the text you must classify.

Choose EXACTLY ONE label:

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

Instructions:
- Determine which attribute is primarily expressed in the prediction.
- Use the question and ground truth only to understand the meaning of the intended attribute.
- Do NOT classify based on whether the prediction matches the ground truth.
- If the prediction contains multiple attributes, choose the dominant one.
- If the prediction is truncated, classify the attribute it is clearly attempting to describe.
- Use "description" only when the prediction is an overall description rather than a specific attribute.
- Use "none" only if no attribute can be identified.
- Return exactly one label from the list above.

Question:
{question}

Ground Truth:
{ground_truth}

Prediction:
{prediction}

Return ONLY valid JSON:

{{
  "predicted_attribute": "<label>",
  "confidence": 0.00,
  "reason": "<one short sentence>"
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
        response = json.loads(response.text) 

        response.setdefault("predicted_attribute", "none")
        response.setdefault("confidence", 0.0)
        response.setdefault("reason", "")

        try:
            response["confidence"] = float(response["confidence"])
        except:
            response["confidence"] = 0.0

        response["predicted_attribute"] = str(
            response.get("predicted_attribute", "none")
        ).strip().lower()

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
all_attributes = [
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
]

confusion = pd.crosstab(
    wrong_df["concept"],
    wrong_df["predicted_attribute"]
).reindex(columns=all_attributes, fill_value=0)

confusion.to_csv(CONFUSION_CSV)

print(confusion.to_string())

#normalize confusion matrix 
confusion_norm = pd.crosstab(
    wrong_df["concept"],
    wrong_df["predicted_attribute"],
    normalize="index"
).reindex(columns=all_attributes, fill_value=0)

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
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
You are an expert researcher analyzing prediction errors in a Vision-Language Model.

Your task is NOT to judge correctness.

Instead, determine which ATTRIBUTE the prediction is primarily answering.

Possible attributes (return EXACTLY one):

festival
origin
material
parts
playing_method
sound
traditional_player
instrument_type
description
none

Definitions

festival
The prediction talks about festivals or celebrations.

origin
The prediction talks about geographical or cultural origin.

material
The prediction talks about materials used to make the instrument.

parts
The prediction talks about components or physical parts.

playing_method
The prediction explains how the instrument is played.

sound
The prediction describes the produced sound.

traditional_player
The prediction identifies who traditionally plays it.

instrument_type
The prediction classifies the instrument.

description
The prediction gives a general description.

none
The prediction clearly answers none of these.

Rules

1. Ignore whether the answer is correct.
2. Ignore the question being asked.
3. Decide ONLY what attribute the prediction is actually describing.
4. If truncated but clearly discussing an attribute, choose that attribute.
5. Return ONLY valid JSON.

Question:
{question}

Ground Truth:
{ground_truth}

Prediction:
{prediction}

Return JSON only

{{
"predicted_attribute":"material",
"confidence":0.95,
"reason":"The prediction describes the material."
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

        return response

    except Exception as e:

        return {

            "predicted_attribute": "none",

            "confidence": 0.0,

            "reason": str(e)

        }
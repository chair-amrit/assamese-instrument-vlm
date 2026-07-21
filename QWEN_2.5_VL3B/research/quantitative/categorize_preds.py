import os
import json
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types
import json
import time
from dotenv import load_dotenv

load_dotenv() 

#initialize client
client=genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL="gemini-3.1-flash-lite"

# Input and output paths
INPUT_JSON=r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\research\test_predictions.json"
OUTPUT_CSV=r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\research\categorized_preds.csv"

# Load predictions
with open(INPUT_JSON,"r",encoding="utf-8") as f:
    predictions=json.load(f)

results=[]

# Classify every prediction
for sample in tqdm(predictions):

    question=sample["question"]
    ground_truth=sample["ground_truth"]
    prediction=sample["prediction"]

    # prompt for LLM    
    prompt=f"""
You are an expert researcher analyzing Vision-Language Model (VLM) predictions for Visual Question Answering (VQA).

Your task is to classify each prediction into EXACTLY ONE primary failure category.

Categories:

1. Correct – Prediction correctly answers the question.
2. Question Misunderstanding – Answers a different question than the one asked.
3. Attribute Swapping – Correct instrument but retrieves the wrong attribute (e.g., festival instead of playing method).
4. Hallucination – Invents facts unsupported by the reference.
5. Wrong Instrument Identification – Describes another instrument.
6. Partial Answer / Incomplete Answer – Only partially answers the question.
7. Truncation – Response is cut off before completion.
8. Repetition – Repeats information unnecessarily.
9. Over-Generalization / Generic Answer – Gives a vague answer applicable to many instruments.
10. Wrong Cultural Knowledge – Incorrect festival, origin, or traditional knowledge.
11. Contradictory Answer – Contains internally conflicting information.
12. Mixed Attribute – Mixes correct facts with incorrect facts; neither alone explains the error.

Instructions:
- Compare the prediction with BOTH the question and the reference answer.
- Assign ONLY ONE category: choose the dominant reason for failure.
- If the answer is fully correct (minor wording differences are acceptable), return "Correct".
- Do NOT judge writing style; judge factual correctness and question relevance.
- Confidence must be a decimal between 0.00 and 1.00.
- Reason must be one concise sentence (≤20 words).

Question:
{question}

Ground Truth:
{ground_truth}

Prediction:
{prediction}

Return ONLY valid JSON:

{{
  "failure_category":"<one category exactly as listed>",
  "confidence":0.00,
  "reason":"<short explanation>"
}}
"""
    try:
        response=client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json"
            )
        )

        response=json.loads(response.text)

        response.setdefault("failure_category","API Error")
        response.setdefault("confidence",0.0)
        response.setdefault("reason","No reason returned.")

    except Exception as e:

        response={
            "failure_category":"API Error",
            "confidence":0.0,
            "reason":str(e)
        }

    time.sleep(4.5)

    results.append({
        **sample,
        "failure_category":response["failure_category"],
        "confidence":response["confidence"],
        "reason":response["reason"]
    })

# Save CSV
df=pd.DataFrame(results)
df.to_csv(OUTPUT_CSV,index=False,encoding="utf-8-sig")

print(f"Saved to: {OUTPUT_CSV}")
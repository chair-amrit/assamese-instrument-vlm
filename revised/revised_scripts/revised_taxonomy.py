"""
src/analysis/revised_taxonomy.py

Implements the axis-first VQA failure taxonomy defined in
docs/methodology/01_mathematical_foundation.md through 04_taxonomy_algorithm.md.

Gemini is used only for claim extraction and atomic per-claim/per-attribute
judgments. All axis aggregation and category derivation (delta(T)) is
deterministic Python, per 02/03/04.
"""

import os
import re
import json
import time
from collections import Counter

import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

INPUT_JSON = r"D:\InternshipGU\Assamese_instrument_VLM\revised\128_tokens\revised_test_predictions.json"
TEST_OUTPUT_CSV = r"D:\InternshipGU\Assamese_instrument_VLM\revised\128_tokens\analysis\revised_taxonomy_128.csv"

TEST_LIMIT = None  

# ---------------------------------------------------------------------------
# Fixed template table — 01 Section 6
# ---------------------------------------------------------------------------

TEMPLATE_TABLE = {
    "q1": {"concept": "festival", "attrs": ["festival"]},
    "q2": {"concept": "origin", "attrs": ["origin"]},
    "q3": {"concept": "material", "attrs": ["material"]},
    "q4": {"concept": "parts", "attrs": ["parts"]},
    "q5": {"concept": "sound", "attrs": ["sound"]},
    "q6": {"concept": "traditional_player", "attrs": ["traditional_player"]},
    "q7": {"concept": "playing_method", "attrs": ["playing_method"]},
    "q8": {"concept": "instrument_type", "attrs": ["instrument_type"]},
    "q9": {"concept": "description", "attrs": ["cultural_significance", "role_in_assamese_music"]},
}

CONCEPT_TO_QID = {v["concept"]: k for k, v in TEMPLATE_TABLE.items()}

# Full attribute universe (all 9 templates' attributes), used for Axis 1
# misaligned-vs-indeterminate discrimination.
ALL_ATTRS = sorted({a for v in TEMPLATE_TABLE.values() for a in v["attrs"]})

# ---------------------------------------------------------------------------
# Axis 2b — fixed, versioned hedge/refusal lexicon (deterministic, no LLM)
# ---------------------------------------------------------------------------

HEDGE_LEXICON_VERSION = "v1"
HEDGE_PATTERNS = [
    r"\bi don'?t know\b",
    r"\bi'?m not sure\b",
    r"\bnot certain\b",
    r"\bmay be\b",
    r"\bmight be\b",
    r"\bcannot determine\b",
    r"\bcan'?t determine\b",
    r"\bi don'?t have (enough )?information\b",
    r"\bunable to (determine|answer|identify)\b",
    r"\bno information (is )?available\b",
    r"\buncertain\b",
    r"\bpossibly\b",
    r"\bi'?m unable\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), flags=re.IGNORECASE)


def detect_axis2b(prediction_text: str) -> str:
    return "yes" if HEDGE_RE.search(prediction_text or "") else "no"


# ---------------------------------------------------------------------------
# Axis 6 — deterministic degenerate-repetition heuristic (no LLM)
# ---------------------------------------------------------------------------

def detect_axis6(prediction_text: str, ngram_size: int = 4, min_repeats: int = 2) -> str:
    """
    Flags 'present' only for degenerate looping (same n-gram repeating
    consecutively/near-consecutively without new information), not natural
    lexical repetition. Deterministic n-gram scan.
    """
    tokens = re.findall(r"\w+", (prediction_text or "").lower())
    if len(tokens) < ngram_size * min_repeats:
        return "absent"

    ngram_counts = Counter()
    for i in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[i:i + ngram_size])
        ngram_counts[ngram] += 1

    for ngram, count in ngram_counts.items():
        if count >= min_repeats:
            positions = [
                i for i in range(len(tokens) - ngram_size + 1)
                if tuple(tokens[i:i + ngram_size]) == ngram
            ]
            gaps = [positions[j + 1] - positions[j] for j in range(len(positions) - 1)]
            if any(gap <= ngram_size + 2 for gap in gaps):
                return "present"

    return "absent"


# ---------------------------------------------------------------------------
# Concept resolution
# ---------------------------------------------------------------------------

def resolve_concept(sample: dict):
    for field in ("question_id", "q_id", "template", "template_id"):
        val = sample.get(field)
        if val and str(val).lower() in TEMPLATE_TABLE:
            return str(val).lower(), TEMPLATE_TABLE[str(val).lower()]

    for field in ("concept", "question_concept", "attribute"):
        val = sample.get(field)
        if val and val in CONCEPT_TO_QID:
            qid = CONCEPT_TO_QID[val]
            return qid, TEMPLATE_TABLE[qid]

    question_text = (sample.get("question") or "").lower()
    KEYWORD_FALLBACK = {
        "festival": "q1", "origin": "q2", "material": "q3", "parts": "q4",
        "sound": "q5", "gender": "q6", "traditionally play": "q6",
        "interaction region": "q7", "type of": "q8", "description": "q9",
        "cultural significance": "q9", "role in assamese music": "q9",
    }
    for kw, qid in KEYWORD_FALLBACK.items():
        if kw in question_text:
            return qid, TEMPLATE_TABLE[qid]

    return None, None


# ---------------------------------------------------------------------------
# Gemini extraction — claims, attribute tagging, comparability, support
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are extracting structured evidence for a formal VQA
evaluation taxonomy. Do NOT classify the prediction into any failure category.
Only extract and judge atomic claims as instructed.

Concept asked (K): {concept}
Required attributes for this question (A_set(K)): {attrs}
Full known attribute universe (for tagging any claim, not just required ones):
{all_attrs}

Question: {question}
Ground truth (G): {ground_truth}
Prediction (P): {prediction}

Step 1 — Extract every minimal, checkable claim from the Prediction.
Step 2 — For each claim, tag it with the SINGLE closest attribute from the
full attribute universe above, or "none" if it does not correspond to any
known attribute (e.g. filler, greeting, unrelated remark).
Step 3 — For each required attribute in A_set(K), determine:
  - G_value: the value/content for that attribute as stated in G, or null if
    G does not independently realize that attribute.
  - P_value: the value/content for that attribute as stated in P's claims,
    or null if no claim addresses it.
  - verdict: one of "consistent" (P_value matches G_value in meaning),
    "contradictory" (P_value conflicts with G_value), or "indeterminate"
    (cannot be reliably judged even though both exist), only when both
    G_value and P_value are non-null. If either is null, set verdict to null.
  - coverage: one of "covered" (P fully and correctly conveys G_value for
    this attribute), "not_covered", or "indeterminate", only when G_value is
    non-null. If G_value is null, set coverage to null.
Step 4 — For EVERY claim extracted in Step 1 (regardless of attribute tag),
determine supported_by_G: true only if the claim is explicitly stated in G or
directly logically entailed by G. Real-world truth outside G does NOT count
as support. If false, the claim is unsupported.

Step 5 — Determine termination_status from the Prediction text only:
  - "truncated" = the response clearly ends mid-thought, mid-clause, or before
    completing its final statement.
  - "intact" = the response reaches a complete linguistic stopping point,
    regardless of whether it ends with punctuation.
  - "indeterminate" = the ending cannot be reliably judged.
Do NOT use the presence or absence of a full stop alone as evidence of truncation.
Do NOT judge semantic correctness here; judge only whether the response itself
appears cut off.

Return ONLY valid JSON in exactly this schema:
{{
  "claims": [
    {{"text": "<claim text>", "attribute": "<attribute name or none>", "supported_by_G": true}}
  ],
  "attribute_evaluation": {{
    "<attribute name>": {{"G_value": "<value or null>", "P_value": "<value or null>", "verdict": "<consistent|contradictory|indeterminate|null>", "coverage": "<covered|not_covered|indeterminate|null>"}}
  }},
  "termination_status": "<intact|truncated|indeterminate>"
}}
"""


def call_gemini_extraction(
    question,
    concept,
    attrs,
    ground_truth,
    prediction,
    retries=5
):
    prompt = EXTRACTION_PROMPT.format(
        concept=concept,
        attrs=attrs,
        all_attrs=ALL_ATTRS,
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

            # Make sure Gemini actually returned something
            if not response.text:
                raise ValueError("Gemini returned an empty response")

            return json.loads(response.text)

        except Exception as e:
            error = str(e)

            # Permanent authentication/access errors
            if "401" in error or "UNAUTHENTICATED" in error:
                print(f"    ❌ 401 authentication error")
                return {
                    "error": error,
                    "claims": [],
                    "attribute_evaluation": {}
                }

            if "403" in error or "PERMISSION_DENIED" in error:
                print(f"    ❌ 403 permission/access error")
                return {
                    "error": error,
                    "claims": [],
                    "attribute_evaluation": {}
                }

            # Retry transient errors
            print(
                f"    ⚠️ Gemini error on attempt "
                f"{attempt}/{retries}: {error[:180]}"
            )

            if attempt < retries:
                wait_time = min(5 * attempt, 30)
                print(f"    🔄 Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("    ❌ All retries failed")

                return {
                    "error": error,
                    "claims": [],
                    "attribute_evaluation": {}
                }

# ---------------------------------------------------------------------------
# Axis computation — deterministic aggregation per 02
# ---------------------------------------------------------------------------

def compute_axis2a(claims):
    return "yes" if len(claims) > 0 else "no"


def compute_axis1(claims, attrs, ax2a):
    if ax2a == "no":
        return "indeterminate", None

    p_k = [c for c in claims if c.get("attribute") in attrs]
    p_bar_k = [c for c in claims if c.get("attribute") not in attrs]

    if p_k:
        return "aligned", None

    other_known = [c for c in p_bar_k if c.get("attribute") not in (None, "none")]
    if other_known:
        return "misaligned", other_known[0]["attribute"]

    return "indeterminate", None


def compute_axis3(attr_eval, attrs):
    comparable = [
        attr_eval[a]["verdict"] for a in attrs
        if a in attr_eval and attr_eval[a].get("verdict") in
        ("consistent", "contradictory", "indeterminate")
    ]
    if not comparable:
        return "indeterminate"
    has_consistent = "consistent" in comparable
    has_contradictory = "contradictory" in comparable
    if has_consistent and has_contradictory:
        return "mixed"
    if all(v == "contradictory" for v in comparable):
        return "incorrect"
    if all(v == "consistent" for v in comparable):
        return "correct"
    return "indeterminate"


def compute_axis4(attr_eval, attrs):
    applicable = [a for a in attrs if attr_eval.get(a, {}).get("G_value") is not None]
    if not applicable:
        return "indeterminate"

    coverages = [attr_eval[a].get("coverage") for a in applicable]
    if any(c == "indeterminate" for c in coverages):
        return "indeterminate"

    n = len(attrs)
    m = sum(1 for c in coverages if c == "covered")
    return "complete" if m == n else "partial"


def compute_axis7(claims, ax2a):
    if ax2a == "no":
        return "none"
    return "present" if any(c.get("supported_by_G") is False for c in claims) else "none"


# ---------------------------------------------------------------------------
# Category derivation — delta(T), 03 Section 5
# ---------------------------------------------------------------------------

def derive_category(ax2a, ax1, ax3, ax4):
    if ax2a == "no":
        return "Non-Answer / Abstention"

    if ax1 == "misaligned":
        return "Question Misunderstanding"

    if ax1 == "indeterminate":
        return "C_review"

    if ax3 == "incorrect":
        return "Hallucination"

    if ax3 == "correct":
        if ax4 == "indeterminate":
            return "C_review"
        if ax4 == "complete":
            return "Correct"
        return "Partial Answer"

    if ax3 == "mixed":
        if ax4 == "indeterminate":
            return "C_review"
        return "Mixed Attribute"

    return "C_review"

# ---------------------------------------------------------------------------
# Per-sample classification
# ---------------------------------------------------------------------------

def classify_one(sample: dict) -> dict:
    question = sample.get("question", "")
    ground_truth = sample.get("ground_truth", "")
    prediction = sample.get("prediction", "")

    qid, template = resolve_concept(sample)
    if template is None:
        return {**sample, "taxonomy_error": "concept_unresolved"}

    concept = template["concept"]
    attrs = template["attrs"]

    extraction = call_gemini_extraction(question, concept, attrs, ground_truth, prediction)
    if "error" in extraction:
        return {**sample, "taxonomy_error": f"gemini_error: {extraction['error']}"}

    claims = extraction.get("claims", [])
    attr_eval = extraction.get("attribute_evaluation", {})

    # Branch A short-circuit: no content => forced values, no attr_eval needed
    ax2a = compute_axis2a(claims)
    ax2b = detect_axis2b(prediction)

    if ax2a == "no":
        ax1, misaligned_target = "indeterminate", None
        ax3, ax4, ax7 = "not_applicable", "not_applicable", "none"
        ax6 = "absent"
    else:
        ax1, misaligned_target = compute_axis1(claims, attrs, ax2a)
        if ax1 == "misaligned":
            ax3, ax4 = "not_applicable", "not_applicable"
        elif ax1 == "indeterminate":
            ax3, ax4 = "indeterminate", "indeterminate"
        else:
            ax3 = compute_axis3(attr_eval, attrs)
            ax4 = compute_axis4(attr_eval, attrs) if ax3 in ("correct", "mixed") else "not_applicable"
        ax6 = detect_axis6(prediction)
        ax7 = compute_axis7(claims, ax2a)

    ax5 = extraction.get("termination_status", "indeterminate")

    category = derive_category(ax2a, ax1, ax3, ax4)
    is_review = category == "C_review"

    return {
        **sample,
        "concept": concept,
        "axis_2a_content": ax2a,
        "axis_2b_hedge": ax2b,
        "axis_1_alignment": ax1,
        "axis_1_misaligned_target": misaligned_target,
        "axis_3_correctness": ax3,
        "axis_4_completeness": ax4,
        "axis_5_termination": ax5,
        "axis_6_repetition": ax6,
        "axis_7_unsupported": ax7,
        "core_category": category,
        "needs_review": is_review,
        "toka_q9_limitation": (
            sample.get("instrument", "").lower() == "toka" and concept == "description"
        ),
    }


# ---------------------------------------------------------------------------
# Test harness — TEST_LIMIT samples only, do not run full dataset yet
# ---------------------------------------------------------------------------

from tqdm import tqdm

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    test_samples = predictions if TEST_LIMIT is None else predictions[:TEST_LIMIT]
    results = []

    pbar = tqdm(
        test_samples,
        desc="Testing revised taxonomy",
        dynamic_ncols=True
    )

    for idx, sample in enumerate(pbar, start=1):
        result = classify_one(sample)
        results.append(result)

        pd.DataFrame(results).to_csv(
            TEST_OUTPUT_CSV,
            index=False,
            encoding="utf-8-sig"
        )

        pbar.set_postfix_str(
            f"Sample {idx}/{len(test_samples)} | ✅ Gemini responded | 💾 Saved",
            refresh=True
        )

        time.sleep(4.5)

    print(f"\n✅ Finished {len(results)} samples")
    print(f"Output: {TEST_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
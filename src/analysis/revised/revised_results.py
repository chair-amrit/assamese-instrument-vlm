"""
src/analysis/generate_taxonomy_artifacts.py

Generates result artifacts for the revised axis-first taxonomy (01-04),
using the 315-row classified CSV as the single source of truth.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = r"D:\InternshipGU\Assamese_instrument_VLM\results\failure_analysis\quantitative_revised\revised_taxonomy - Sheet1.csv"
OUT_DIR = r"D:\InternshipGU\Assamese_instrument_VLM\results\failure_analysis\quantitative_revised\artifacts"

CATEGORIES = [
    "Correct",
    "Question Misunderstanding",
    "Incoherent Response",
    "Hallucination",
    "Partial Answer",
    "Mixed Attribute",
    "Non-Answer / Abstention",
]

FLAG_COLS = {
    "axis_5_termination": "truncated",
    "axis_6_repetition": "present",
    "axis_7_unsupported": "present",
}


def load(path):
    df = pd.read_csv(path)
    df["category"] = pd.Categorical(df["core_category"], categories=CATEGORIES)
    return df


# ---------------------------------------------------------------------------
# 1. Category summary
# ---------------------------------------------------------------------------

def category_summary(df, out_dir):
    n = len(df)
    counts = df["core_category"].value_counts().reindex(CATEGORIES, fill_value=0)
    pct = (counts / n * 100).round(2)

    rows = []
    for cat in CATEGORIES:
        cat_df = df[df["core_category"] == cat]
        row = {
            "category": cat,
            "count": int(counts[cat]),
            "percentage": pct[cat],
        }
        for col, flag_val in FLAG_COLS.items():
            row[f"{col}_{flag_val}_count"] = int((cat_df[col] == flag_val).sum())
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    overall_flags = {
        "axis_5_truncated_total": int((df["axis_5_termination"] == "truncated").sum()),
        "axis_6_repetition_total": int((df["axis_6_repetition"] == "present").sum()),
        "axis_7_unsupported_total": int((df["axis_7_unsupported"] == "present").sum()),
        "total_samples": n,
    }
    overall_df = pd.DataFrame([overall_flags])

    crosstabs = {}
    for col, flag_val in FLAG_COLS.items():
        ct = pd.crosstab(df["core_category"], df[col]).reindex(CATEGORIES, fill_value=0)
        crosstabs[col] = ct

    os.makedirs(out_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(out_dir, "category_summary.csv"), index=False)
    overall_df.to_csv(os.path.join(out_dir, "flag_summary_overall.csv"), index=False)
    for col, ct in crosstabs.items():
        ct.to_csv(os.path.join(out_dir, f"crosstab_category_x_{col}.csv"))

    print("=" * 60)
    print("Category Summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()
    print("Overall Flags")
    print(overall_df.to_string(index=False))

    return summary_df, overall_df, crosstabs


# ---------------------------------------------------------------------------
# 2. Failure-distribution figures
# ---------------------------------------------------------------------------

def distribution_figures(df, out_dir):
    counts = df["core_category"].value_counts().reindex(CATEGORIES, fill_value=0)

    plt.figure(figsize=(10, 6))
    ax = counts.plot(kind="bar", color="steelblue")
    ax.bar_label(ax.containers[0], padding=3)
    plt.title("Revised Core-Category Distribution")
    plt.xlabel("Core Category")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "category_distribution.png"), dpi=300)
    plt.close()

    for col, flag_val in FLAG_COLS.items():
        vc = df[col].value_counts()
        plt.figure(figsize=(6, 5))
        ax = vc.plot(kind="bar", color="coral")
        ax.bar_label(ax.containers[0], padding=3)
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}_distribution.png"), dpi=300)
        plt.close()

    print(f"\nSaved: category_distribution.png, "
          f"{', '.join(f'{c}_distribution.png' for c in FLAG_COLS)}")


# ---------------------------------------------------------------------------
# 3. Per-instrument / per-concept summary
# ---------------------------------------------------------------------------

def instrument_and_concept_summary(df, out_dir):
    instrument_ct = pd.crosstab(df["instrument"], df["core_category"]).reindex(
        columns=CATEGORIES, fill_value=0
    )
    instrument_pct = (instrument_ct.div(instrument_ct.sum(axis=1), axis=0) * 100).round(2)

    instrument_ct.to_csv(os.path.join(out_dir, "instrument_category_counts.csv"))
    instrument_pct.to_csv(os.path.join(out_dir, "instrument_category_percentages.csv"))

    concept_ct = pd.crosstab(df["concept"], df["core_category"]).reindex(
        columns=CATEGORIES, fill_value=0
    )
    concept_pct = (concept_ct.div(concept_ct.sum(axis=1), axis=0) * 100).round(2)

    concept_ct.to_csv(os.path.join(out_dir, "concept_category_counts.csv"))
    concept_pct.to_csv(os.path.join(out_dir, "concept_category_percentages.csv"))

    print("\nInstrument x Category (counts)")
    print(instrument_ct.to_string())
    print("\nConcept x Category (counts)")
    print(concept_ct.to_string())

    return instrument_ct, concept_ct


# ---------------------------------------------------------------------------
# 4. Sample-level taxonomy table
# ---------------------------------------------------------------------------

def sample_level_table(df, out_dir):
    cols = [
        "instrument", "question_id", "prediction",
        "core_category", "axis_5_termination",
        "axis_6_repetition", "axis_7_unsupported",
    ]
    table = df[cols].copy()
    table.to_csv(os.path.join(out_dir, "sample_level_taxonomy.csv"), index=False)
    print(f"\nSaved sample_level_taxonomy.csv ({len(table)} rows)")
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load(INPUT_CSV)
    os.makedirs(OUT_DIR, exist_ok=True)

    category_summary(df, OUT_DIR)
    distribution_figures(df, OUT_DIR)
    instrument_and_concept_summary(df, OUT_DIR)
    sample_level_table(df, OUT_DIR)

    print(f"\nAll artifacts saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
"""
Generate the complete final taxonomy-results package from the locked
315-row revised_taxonomy_128.csv.

This script does NOT reclassify anything. It reads the final taxonomy CSV
exactly as-is and produces summary tables, cross-tabs, figures, and
paper-ready tables from the existing columns.
"""

import sys
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

INPUT_CSV = r"D:\InternshipGU\Assamese_instrument_VLM\revised\128_tokens\analysis\revised_taxonomy_128.csv"
OUTPUT_DIR = r"D:\InternshipGU\Assamese_instrument_VLM\revised\128_tokens\analysis\artifacts"

EXPECTED_ROWS = 315

REQUIRED_COLUMNS = [
    "image", "instrument", "question_id", "concept", "question",
    "ground_truth", "prediction",
    "axis_2a_content", "axis_2b_hedge", "axis_1_alignment",
    "axis_1_misaligned_target", "axis_3_correctness", "axis_4_completeness",
    "axis_5_termination", "axis_6_repetition", "axis_7_unsupported",
    "core_category", "needs_review", "toka_q9_limitation",
]

# Frozen category space (order matters for consistent reporting; includes
# zero-count categories such as C_review-resolved labels that may not
# appear in this particular run).
CORE_CATEGORIES = [
    "Correct",
    "Partial Answer",
    "Hallucination",
    "Question Misunderstanding",
    "Mixed Attribute",
    "Incoherent Response",
    "Non-Answer / Abstention",
    "C_review",
]

# Frozen per-axis label spaces (full domain, per 01_mathematical_foundation.md
# and 02_decision_functions.md), used so zero-count labels still appear.
AXIS_DOMAINS = {
    "axis_1_alignment": ["aligned", "misaligned", "indeterminate"],
    "axis_2a_content": ["yes", "no"],
    "axis_2b_hedge": ["yes", "no"],
    "axis_3_correctness": ["correct", "incorrect", "mixed", "not_applicable", "indeterminate"],
    "axis_4_completeness": ["complete", "partial", "not_applicable", "indeterminate"],
    "axis_5_termination": ["intact", "truncated", "indeterminate"],
    "axis_6_repetition": ["absent", "present"],
    "axis_7_unsupported": ["none", "present"],
}

AXIS_LABELS = {
    "axis_1_alignment": "Axis 1 — Question/Semantic Alignment",
    "axis_2a_content": "Axis 2a — Substantive Content Present",
    "axis_2b_hedge": "Axis 2b — Uncertainty/Refusal Marker",
    "axis_3_correctness": "Axis 3 — Semantic Correctness",
    "axis_4_completeness": "Axis 4 — Completeness",
    "axis_5_termination": "Axis 5 — Termination Integrity",
    "axis_6_repetition": "Axis 6 — Repetition",
    "axis_7_unsupported": "Axis 7 — Unsupported Content",
}

DPI = 300


# ----------------------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"FATAL: missing required columns: {missing_cols}")

    if len(df) != EXPECTED_ROWS:
        sys.exit(f"FATAL: expected {EXPECTED_ROWS} rows, found {len(df)}")

    needs_review = to_bool(df["needs_review"])

    if needs_review.isna().any():
        sys.exit("FATAL: invalid values in needs_review column")

    if needs_review.sum() != 0:
        n = int(needs_review.sum())
        sys.exit(f"FATAL: expected 0 needs_review rows, found {n}")

    unknown_cats = set(df["core_category"].unique()) - set(CORE_CATEGORIES)
    if unknown_cats:
        sys.exit(f"FATAL: unrecognized core_category values not in the frozen "
                  f"taxonomy: {unknown_cats}")

    for axis_col, domain in AXIS_DOMAINS.items():
        unknown_vals = set(df[axis_col].dropna().unique()) - set(domain)
        if unknown_vals:
            sys.exit(f"FATAL: unrecognized values in {axis_col} not in the "
                      f"frozen axis domain: {unknown_vals}")

    print(f"Validation passed: {len(df)} rows, 0 needs_review, "
          f"all category/axis values within frozen taxonomy.")


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def to_bool(series):
    return series.astype(str).str.strip().str.lower().map({
        "true": True,
        "false": False
    })


def out(path: str) -> str:
    return os.path.join(OUTPUT_DIR, path)


def count_pct(series: pd.Series, domain: list, total: int) -> pd.DataFrame:
    """Count + percentage for every label in `domain`, including zero-count ones."""
    counts = series.value_counts()
    rows = []
    for label in domain:
        c = int(counts.get(label, 0))
        pct = round(100 * c / total, 2) if total else 0.0
        rows.append({"label": label, "count": c, "percentage": pct})
    return pd.DataFrame(rows)


def bar_chart(df_counts: pd.DataFrame, title: str, xlabel: str, filename: str,
              rotate: bool = True, color: str = "#4C72B0"):
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(df_counts) + 2), 5))
    bars = ax.bar(df_counts["label"], df_counts["count"], color=color, edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    else:
        plt.setp(ax.get_xticklabels(), fontsize=9)
    for b, (_, row) in zip(bars, df_counts.iterrows()):
        h = b.get_height()
        ax.annotate(f"{row['count']}\n({row['percentage']}%)",
                     xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out(filename), dpi=DPI)
    plt.close(fig)


def crosstab_bar(df: pd.DataFrame, group_col: str, cat_col: str, title: str,
                  xlabel: str, filename: str):
    ct = pd.crosstab(df[group_col], df[cat_col])
    ct = ct.reindex(columns=[c for c in CORE_CATEGORIES if c in ct.columns])
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(ct) + 2), 6))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", edgecolor="black", linewidth=0.3)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    ax.legend(title="Core Category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out(filename), dpi=DPI)
    plt.close(fig)


def crosstab_binary_bar(df: pd.DataFrame, flag_col: str, present_value: str,
                         title: str, filename: str):
    ct = pd.crosstab(df["core_category"], df[flag_col])
    ct = ct.reindex(index=[c for c in CORE_CATEGORIES if c in ct.index])
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="black", linewidth=0.3)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Core Category", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.legend(title=flag_col, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out(filename), dpi=DPI)
    plt.close(fig)


# ----------------------------------------------------------------------
# SECTIONS
# ----------------------------------------------------------------------

def section_category_summary(df: pd.DataFrame, total: int):
    cs = count_pct(df["core_category"], CORE_CATEGORIES, total)
    cs = cs.rename(columns={"label": "core_category"})
    cs.to_csv(out("category_summary.csv"), index=False)
    bar_chart(cs.rename(columns={"core_category": "label"}),
              "Core Category Distribution (N=%d)" % total,
              "Core Category", "category_distribution.png")
    return cs


def section_axis_distributions(df: pd.DataFrame, total: int):
    for axis_col, domain in AXIS_DOMAINS.items():
        ac = count_pct(df[axis_col], domain, total)
        ac.to_csv(out(f"{axis_col}_distribution.csv"), index=False)
        bar_chart(ac, f"{AXIS_LABELS[axis_col]} (N={total})",
                   AXIS_LABELS[axis_col], f"{axis_col}_distribution.png",
                   rotate=(len(domain) > 3))


def section_crosstabs(df: pd.DataFrame):
    # 10. category_by_instrument
    ct = pd.crosstab(df["instrument"], df["core_category"])
    ct = ct.reindex(columns=[c for c in CORE_CATEGORIES if c in ct.columns], fill_value=0)
    ct.to_csv(out("category_by_instrument.csv"))
    crosstab_bar(df, "instrument", "core_category",
                 "Core Category by Instrument", "Instrument",
                 "category_by_instrument.png")

    # 11. category_by_concept
    ct = pd.crosstab(df["concept"], df["core_category"])
    ct = ct.reindex(columns=[c for c in CORE_CATEGORIES if c in ct.columns], fill_value=0)
    ct.to_csv(out("category_by_concept.csv"))
    crosstab_bar(df, "concept", "core_category",
                 "Core Category by Concept", "Concept",
                 "category_by_concept.png")

    # 12. category_by_axis5
    ct = pd.crosstab(df["core_category"], df["axis_5_termination"])
    ct = ct.reindex(index=[c for c in CORE_CATEGORIES if c in ct.index], fill_value=0)
    ct.to_csv(out("category_by_axis5.csv"))
    crosstab_binary_bar(df, "axis_5_termination", "truncated",
                         "Core Category by Axis 5 (Termination Integrity)",
                         "category_by_axis5.png")

    # 13. category_by_axis6
    ct = pd.crosstab(df["core_category"], df["axis_6_repetition"])
    ct = ct.reindex(index=[c for c in CORE_CATEGORIES if c in ct.index], fill_value=0)
    ct.to_csv(out("category_by_axis6.csv"))
    crosstab_binary_bar(df, "axis_6_repetition", "present",
                         "Core Category by Axis 6 (Repetition)",
                         "category_by_axis6.png")

    # 14. category_by_axis7
    ct = pd.crosstab(df["core_category"], df["axis_7_unsupported"])
    ct = ct.reindex(index=[c for c in CORE_CATEGORIES if c in ct.index], fill_value=0)
    ct.to_csv(out("category_by_axis7.csv"))
    crosstab_binary_bar(df, "axis_7_unsupported", "present",
                         "Core Category by Axis 7 (Unsupported Content)",
                         "category_by_axis7.png")


def _group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_val, sub in df.groupby(group_col, sort=True):
        n = len(sub)
        row = {group_col: group_val, "total_samples": n}
        for cat in CORE_CATEGORIES:
            c = int((sub["core_category"] == cat).sum())
            row[f"{cat}_count"] = c
            row[f"{cat}_pct"] = round(100 * c / n, 2) if n else 0.0
        for axis_col, present_label in [
            ("axis_5_termination", "truncated"),
            ("axis_6_repetition", "present"),
            ("axis_7_unsupported", "present"),
        ]:
            c = int((sub[axis_col] == present_label).sum())
            row[f"{axis_col}_{present_label}_count"] = c
            row[f"{axis_col}_{present_label}_pct"] = round(100 * c / n, 2) if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def section_instrument_concept_summaries(df: pd.DataFrame):
    inst = _group_summary(df, "instrument")
    inst.to_csv(out("instrument_summary.csv"), index=False)

    conc = _group_summary(df, "concept")
    conc.to_csv(out("concept_summary.csv"), index=False)

    return inst, conc


def section_sample_level_and_failures(df: pd.DataFrame):
    sample_cols = [
        "image", "instrument", "question_id", "concept", "prediction",
        "axis_1_alignment", "axis_2a_content", "axis_2b_hedge",
        "axis_3_correctness", "axis_4_completeness", "axis_5_termination",
        "axis_6_repetition", "axis_7_unsupported",
        "core_category", "needs_review", "toka_q9_limitation",
    ]
    sample_df = df[sample_cols].copy()
    sample_df.to_csv(out("sample_level_taxonomy.csv"), index=False)

    failure_df = df[df["core_category"] != "Correct"].copy()
    failure_df.to_csv(out("failure_cases.csv"), index=False)

    unsupported_df = df[df["axis_7_unsupported"] == "present"].copy()
    unsupported_df.to_csv(out("unsupported_cases.csv"), index=False)

    repetitive_df = df[df["axis_6_repetition"] == "present"].copy()
    repetitive_df.to_csv(out("repetitive_cases.csv"), index=False)

    return failure_df, unsupported_df, repetitive_df


def section_paper_tables(cat_summary: pd.DataFrame, inst_summary: pd.DataFrame,
                          conc_summary: pd.DataFrame):
    # 21. paper_category_table
    paper_cat = cat_summary.rename(columns={
        "core_category": "Category", "count": "N", "percentage": "Percent (%)"
    })
    paper_cat.to_csv(out("paper_category_table.csv"), index=False)

    # 22. paper_instrument_table  (N + Correct% + Partial% + Hallucination% + QM%)
    keep_cats = ["Correct", "Partial Answer", "Hallucination", "Question Misunderstanding"]
    paper_inst = inst_summary[["instrument", "total_samples"] +
                               [f"{c}_pct" for c in keep_cats]].copy()
    paper_inst.columns = ["Instrument", "N"] + [f"{c} (%)" for c in keep_cats]
    paper_inst.to_csv(out("paper_instrument_table.csv"), index=False)

    # 23. paper_concept_table
    paper_conc = conc_summary[["concept", "total_samples"] +
                               [f"{c}_pct" for c in keep_cats]].copy()
    paper_conc.columns = ["Concept", "N"] + [f"{c} (%)" for c in keep_cats]
    paper_conc.to_csv(out("paper_concept_table.csv"), index=False)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    if not os.path.isfile(INPUT_CSV):
        sys.exit(f"FATAL: input CSV not found at {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    validate(df)
    ensure_output_dir()

    total = len(df)

    cat_summary = section_category_summary(df, total)
    section_axis_distributions(df, total)
    section_crosstabs(df)
    inst_summary, conc_summary = section_instrument_concept_summaries(df)
    failure_df, unsupported_df, repetitive_df = section_sample_level_and_failures(df)
    section_paper_tables(cat_summary, inst_summary, conc_summary)

    # ------------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL TAXONOMY REPORT")
    print("=" * 60)
    print(f"Total rows processed: {total}")

    needs_review = to_bool(df["needs_review"])

    if needs_review.isna().any():
        sys.exit("FATAL: invalid values in needs_review column")

    print(f"needs_review = True rows: {int(needs_review.sum())}")
    toka_limit = to_bool(df["toka_q9_limitation"])

    if toka_limit.isna().any():
        sys.exit("FATAL: invalid values in toka_q9_limitation column")

    print(f"toka_q9_limitation = True rows: {int(toka_limit.sum())} "
        f"(annotation flag, not a failure)")
    
    print("\nCore category distribution:")
    for _, row in cat_summary.iterrows():
        print(f"  {row['core_category']:<30} {row['count']:>4}  ({row['percentage']:>5.2f}%)")

    print("\nCross-cutting flag totals:")
    for axis_col, present_label, name in [
        ("axis_5_termination", "truncated", "Truncated"),
        ("axis_6_repetition", "present", "Repetitive"),
        ("axis_7_unsupported", "present", "Unsupported content"),
    ]:
        c = int((df[axis_col] == present_label).sum())
        pct = round(100 * c / total, 2) if total else 0.0
        print(f"  {name:<20} {c:>4}  ({pct:>5.2f}%)")

    print(f"\nFailure cases (core_category != Correct): {len(failure_df)}")
    print(f"Unsupported-content cases: {len(unsupported_df)}")
    print(f"Repetitive cases: {len(repetitive_df)}")
    print(f"\nAll artifacts written to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
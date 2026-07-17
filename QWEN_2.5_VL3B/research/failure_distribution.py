import pandas as pd
import matplotlib.pyplot as plt

# PATH
CSV_PATH = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\research\categorized_preds.csv"


df = pd.read_csv(CSV_PATH)

# Overall Accuracy
total = len(df)
correct = (df["failure_category"] == "Correct").sum()
accuracy = correct / total * 100

print("=" * 60)
print(f"Overall Accuracy : {accuracy:.2f}%")
print(f"Correct : {correct}")
print(f"Total   : {total}")
print("=" * 60)

# 1. FAILURE DISTRIBUTION
failure_counts = (
    df["failure_category"]
    .value_counts()
)

failure_percent = (
    df["failure_category"]
    .value_counts(normalize=True)
    * 100
).round(2)

failure_summary = pd.DataFrame({
    "Count": failure_counts,
    "Percentage (%)": failure_percent
})

print("\nFailure Distribution")
print(failure_summary)

failure_summary.to_csv(
    "failure_distribution.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(10,6))
ax = failure_counts.plot(kind="bar")
ax.bar_label(ax.containers[0], padding=3)
plt.title("Failure Category Distribution")
plt.xlabel("Failure Category")
plt.ylabel("Count")
plt.ylim(0, 190)
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("failure_distribution.png", dpi=300)
plt.close()

# 2. QUESTION TYPE ANALYSIS
concept_rows = []

for concept, group in df.groupby("concept"):

    total_q = len(group)

    correct_q = (group["failure_category"] == "Correct").sum()

    accuracy_q = correct_q / total_q * 100

    failures = group[group["failure_category"] != "Correct"]

    if len(failures) == 0:
        common_failure = "None"
    else:
        common_failure = failures["failure_category"].mode()[0]

    concept_rows.append({

        "Concept": concept,
        "Total Questions": total_q,
        "Correct": correct_q,
        "Accuracy (%)": round(accuracy_q,2),
        "Most Common Failure": common_failure

    })

concept_df = pd.DataFrame(concept_rows)

concept_df = concept_df.sort_values(
    by="Accuracy (%)",
    ascending=False
)

print("\nQuestion Type Analysis")
print(concept_df)

concept_df.to_csv(
    "question_type_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10,6))
bars = plt.bar(concept_df["Concept"], concept_df["Accuracy (%)"])
plt.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

plt.title("Accuracy by Question Type")
plt.xlabel("Question Type")
plt.ylabel("Accuracy (%)")
plt.yticks(range(0, 101, 5))
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("question_type_accuracy.png", dpi=300)
plt.close()

# 3. INSTRUMENT ANALYSIS
instrument_rows = []

for instrument, group in df.groupby("instrument"):

    total_i = len(group)

    correct_i = (group["failure_category"] == "Correct").sum()

    accuracy_i = correct_i / total_i * 100

    failures = group[group["failure_category"] != "Correct"]

    if len(failures)==0:
        common_failure="None"
    else:
        common_failure=failures["failure_category"].mode()[0]

    instrument_rows.append({

        "Instrument": instrument,
        "Total Questions": total_i,
        "Correct": correct_i,
        "Accuracy (%)": round(accuracy_i,2),
        "Most Common Failure": common_failure

    })

instrument_df = pd.DataFrame(instrument_rows)

instrument_df = instrument_df.sort_values(
    by="Accuracy (%)",
    ascending=False
)

print("\nInstrument Analysis")
print(instrument_df)

instrument_df.to_csv(
    "instrument_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10,6))
bars = plt.bar(instrument_df["Instrument"], instrument_df["Accuracy (%)"])
plt.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

plt.title("Accuracy by Instrument")
plt.xlabel("Instrument")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 75)
plt.yticks(range(0, 76, 5))
plt.xticks(rotation=35)
plt.tight_layout()
plt.savefig("instrument_accuracy.png", dpi=300)
plt.close()

# 4. FAILURE DISTRIBUTION BY QUESTION TYPE
concept_failure = pd.crosstab(
    df["concept"],
    df["failure_category"]
)

concept_failure.to_csv(
    "question_type_failure_distribution.csv",
    encoding="utf-8-sig"
)

# 5. FAILURE DISTRIBUTION BY INSTRUMENT
instrument_failure = pd.crosstab(
    df["instrument"],
    df["failure_category"]
)

instrument_failure.to_csv(
    "instrument_failure_distribution.csv",
    encoding="utf-8-sig"
)

print("\nAnalysis Complete!\n")

print("Generated Files:")
print("-------------------------------")
print("failure_distribution.csv")
print("failure_distribution.png")
print("question_type_analysis.csv")
print("question_type_accuracy.png")
print("instrument_analysis.csv")
print("instrument_accuracy.png")
print("question_type_failure_distribution.csv")
print("instrument_failure_distribution.csv")
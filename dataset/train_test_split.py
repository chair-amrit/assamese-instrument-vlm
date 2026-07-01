import json
from collections import Counter
from sklearn.model_selection import train_test_split

# Load dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total samples:", len(data))

# Stratified Split (70 / 15 / 15)
labels = [sample["instrument"] for sample in data]

# 70% train, 30% temp
train_data, temp_data = train_test_split(
    data,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

# Split remaining 30% into 15% val + 15% test
temp_labels = [sample["instrument"] for sample in temp_data]

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42
)

# Save Splits
with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open("val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

with open("test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

# Overall Counts

print("\nSAMPLE COUNTS:")
print(f"Train      : {len(train_data)}")
print(f"Validation : {len(val_data)}")
print(f"Test       : {len(test_data)}")

# Instrument Distribution
def print_distribution(name, dataset):

    counts = Counter(
        sample["instrument"]
        for sample in dataset
    )

    print(f"\n=== {name.upper()} DISTRIBUTION ===")

    for instrument in sorted(counts):
        print(f"{instrument:10s}: {counts[instrument]}")

    print(f"Unique instruments: {len(counts)}")

print_distribution("Train", train_data)
print_distribution("Validation", val_data)
print_distribution("Test", test_data)

# Verify All 7 Instruments Present
required = {
    "bahi",
    "bihudhol",
    "gogona",
    "khutitaal",
    "pepa",
    "toka",
    "xutuli"
}

for name, dataset in [
    ("Train", train_data),
    ("Validation", val_data),
    ("Test", test_data)
]:

    present = {
        sample["instrument"]
        for sample in dataset
    }

    if required.issubset(present):
        print(f"\n{name}: PASS")
    else:
        print(f"\n{name}: FAIL")
        print("Missing:", required - present)

print("\ntrain.json saved")
print("val.json saved")
print("test.json saved")
import json
import os
import re

# Load dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Add instrument field
for sample in data:
    filename = os.path.basename(sample["image"]).lower()

    instrument = re.match(r"[a-z]+", filename).group()

    sample["instrument"] = instrument

# Save updated dataset
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Updated {len(data)} samples.")
print("Added instrument field to all entries.")
print("\nExample:")
print(data[0])
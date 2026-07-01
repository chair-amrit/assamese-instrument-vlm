from PIL import Image
import json
import os
from collections import Counter, defaultdict

IMAGE_DIR = "images"

# Normalize images + lowercase filenames

for filename in os.listdir(IMAGE_DIR):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    old_path = os.path.join(IMAGE_DIR, filename)

    new_filename = filename.lower()
    new_path = os.path.join(IMAGE_DIR, new_filename)

    if old_path != new_path:
        os.rename(old_path, new_path)

    image = Image.open(new_path)

    if image.mode != "RGB":
        image = image.convert("RGB")
        image.save(new_path)

print("Image normalization complete.")

# Load dataset

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total samples:", len(data))

# Empty field check

for i, sample in enumerate(data):

    if not sample["image"]:
        print("Missing image:", i)

    if not sample["instrument"]:
        print("Missing instrument:", i)

    if not sample["question"]:
        print("Missing question:", i)

    if not sample["answer"]:
        print("Missing answer:", i)

print("Empty field check complete.")

# Missing image check

missing = []

for sample in data:

    image_path = sample["image"].lower()

    if not os.path.exists(image_path):
        missing.append(image_path)

print("Missing images:", len(missing))

# Verify 7 instruments

instrument_counts = Counter(
    sample["instrument"]
    for sample in data
)

print("\nInstrument Distribution")

for instrument, count in sorted(instrument_counts.items()):
    print(f"{instrument}: {count}")

print("Unique instruments:", len(instrument_counts))

# Verify 9 questions per instrument

questions_per_instrument = defaultdict(set)

for sample in data:
    questions_per_instrument[
        sample["instrument"]
    ].add(sample["question"])

print("\nQuestions per Instrument")

for instrument, questions in sorted(
    questions_per_instrument.items()
):
    print(
        f"{instrument}: {len(questions)}"
    )

    if len(questions) != 9:
        print(
            f"ERROR: {instrument} has {len(questions)} questions"
        )
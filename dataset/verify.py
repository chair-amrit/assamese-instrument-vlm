# Normalize all images to RGB mode (handle RGBA conversion)
from PIL import Image
import os

IMAGE_DIR = "images"

for filename in os.listdir(IMAGE_DIR):

    old_path = os.path.join(IMAGE_DIR, filename)

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    new_filename = filename.lower()
    new_path = os.path.join(IMAGE_DIR, new_filename)

    if old_path != new_path:
        os.rename(old_path, new_path)

    image = Image.open(new_path)

    if image.mode != "RGB":
        image = image.convert("RGB")
        image.save(new_path)

print("Image normalization complete.")

#Verify dataset
import json
from collections import Counter

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total samples:", len(data))

#Check empty fields
for i, sample in enumerate(data):

    if not sample["image"]:
        print("Missing image:", i)

    if not sample["question"]:
        print("Missing question:", i)

    if not sample["answer"]:
        print("Missing answer:", i)

print("Empty field check complete.")

#Verify 7 instruments
import os
import re

instrument_counts = Counter()

for sample in data:
    filename = os.path.basename(sample["image"]).lower()

    instrument = re.match(r"[a-z]+", filename).group()

    instrument_counts[instrument] += 1

print("Number of questions per instrument:", instrument_counts)
print("Unique instruments:", len(instrument_counts))

#Verify 9 questions per instrument
from collections import defaultdict

questions_per_instrument = defaultdict(set)

for sample in data:
    filename = os.path.basename(sample["image"]).lower()

    instrument = re.match(r"[a-z]+", filename).group()

    questions_per_instrument[instrument].add(
        sample["question"]
    )

print("\nQuestions per instrument:")

for instrument, questions in questions_per_instrument.items():
    print(f"{instrument}: {len(questions)}")
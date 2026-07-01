import json
import os
from sklearn.model_selection import train_test_split

with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

#number of images per instrument
from collections import Counter

image_instrument_map = {}

for sample in dataset:
    img = sample["image"]

    if img not in image_instrument_map:
        image_instrument_map[img] = sample["instrument"]

image_counts = Counter(image_instrument_map.values())

print("\n=== IMAGE COUNT PER INSTRUMENT ===")

for instrument, count in sorted(image_counts.items()):
    print(f"{instrument:10s}: {count}")

print("Total Images:", len(image_instrument_map))

# Step 1: Get unique images per instrument
image_instrument_map = {}
for sample in dataset:
    img = sample["image"]
    if img not in image_instrument_map:
        image_instrument_map[img] = sample["instrument"]

images = list(image_instrument_map.keys())
labels = [image_instrument_map[img] for img in images]

# Step 2: Split images (not samples), stratified by instrument
train_imgs, temp_imgs, _, temp_labels = train_test_split(
    images, labels, test_size=0.30, stratify=labels, random_state=42
)
val_imgs, test_imgs = train_test_split(
    temp_imgs, test_size=0.50, stratify=temp_labels, random_state=42
)

# Step 3: Assign ALL 9 QA pairs of each image to its split
train_data = [s for s in dataset if s["image"] in train_imgs]
val_data   = [s for s in dataset if s["image"] in val_imgs]
test_data  = [s for s in dataset if s["image"] in test_imgs]

print(f"Train images: {len(train_imgs)}, QA pairs: {len(train_data)}")
print(f"Val images:   {len(val_imgs)}, QA pairs: {len(val_data)}")
print(f"Test images:  {len(test_imgs)}, QA pairs: {len(test_data)}")

# Step 4: Verify zero image overlap
assert not set(train_imgs) & set(val_imgs)
assert not set(train_imgs) & set(test_imgs)
assert not set(val_imgs)   & set(test_imgs)
print("No leakage confirmed.")

# Step 5: Save
for name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
    with open(f"{name}.json", "w") as f:
        json.dump(split, f, indent=2)

def print_image_distribution(name, image_list):

    counts = Counter(
        image_instrument_map[img]
        for img in image_list
    )

    print(f"\n=== {name.upper()} IMAGE DISTRIBUTION ===")

    for instrument, count in sorted(counts.items()):
        print(f"{instrument:10s}: {count}")

    print("Total Images:", len(image_list))

print_image_distribution("Train", train_imgs)
print_image_distribution("Validation", val_imgs)
print_image_distribution("Test", test_imgs)
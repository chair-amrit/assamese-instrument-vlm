import os
import json
import cv2
import albumentations as A
from copy import deepcopy

DATASET_JSON   = "dataset.json"          
IMAGES_DIR     = "images"                
OUTPUT_JSON    = "dataset_augmented.json"
AUGMENTED_DIR = "images/augmented"
os.makedirs(AUGMENTED_DIR, exist_ok=True)


# 2 augmentation pipelines (one per augmented copy)
aug1 = A.Compose([
    A.Rotate(limit=8, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    A.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.05, hue=0.0, p=1.0),
])

aug2 = A.Compose([
    A.Rotate(limit=(-8, -4), border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.0, p=1.0),
])

augmentations = [("aug1", aug1), ("aug2", aug2)]



#load datatset
with open(DATASET_JSON, "r") as f:
    data = json.load(f)



# get unique images preserving order
seen = set()
unique_images = []
for entry in data:
    if entry["image"] not in seen:
        seen.add(entry["image"])
        unique_images.append(entry["image"])

print(f"Unique images found: {len(unique_images)}")



# augment
new_entries = []

for img_path in unique_images:
    full_path = os.path.join(IMAGES_DIR, os.path.basename(img_path))

    if not os.path.exists(full_path):
        print(f"  [SKIP] not found: {full_path}")
        continue

    image = cv2.imread(full_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get all QA pairs for this image
    qa_pairs = [e for e in data if e["image"] == img_path]

    for suffix, pipeline in augmentations:
        augmented      = pipeline(image=image)["image"]
        aug_image      = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        # build new filename e.g. bihudhol1_aug1.jpg
        base           = os.path.splitext(os.path.basename(img_path))[0]
        ext            = os.path.splitext(img_path)[1]
        new_filename   = f"{base}_{suffix}{ext}"
        new_img_path = os.path.join(AUGMENTED_DIR, new_filename)

        cv2.imwrite(new_img_path, aug_image)
        print(f"  Saved: {new_img_path}")

        # duplicate QA pairs pointing to new image
        for qa in qa_pairs:
            new_entry = deepcopy(qa)
            new_entry["image"] = f"images/augmented/{new_filename}"
            new_entries.append(new_entry)



# save final json : originals first  then  augmented
final_data = data + new_entries  


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)
print(f"\nDone.")
print(f"  Original entries : {len(data)}")
print(f"  Augmented entries: {len(new_entries)}")
print(f"  Total entries    : {len(final_data)}")
print(f"  Saved to         : {OUTPUT_JSON}")
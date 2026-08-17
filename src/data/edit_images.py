import os
from pathlib import Path

# folder path   
folder_path = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\dataset_32images\xutuli\train"   
base_name = "xutuli_train"                                        
# Example output:
# instrument1.jpg
# instrument2.jpg
# instrument3.jpg

folder = Path(folder_path)

# Get all .jpg files (case insensitive)
jpg_files = sorted(
    [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"]
)

# Count images
print(f"Found {len(jpg_files)} JPG images.")

if len(jpg_files) == 0:
    print("No JPG images found.")
    exit()

# Rename images
for i, file in enumerate(jpg_files, start=1):
    new_name = f"{base_name}{i}.jpg"
    new_path = folder / new_name

    os.rename(file, new_path)
    print(f"{file.name}  -->  {new_name}")

print("\nDone!")
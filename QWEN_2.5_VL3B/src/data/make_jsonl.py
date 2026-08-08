import pandas as pd
import json
import os

# PATHS
CSV_PATH = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\final_vqa.csv"
OUTPUT_DIR = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\jsonl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD CSV
df = pd.read_csv(CSV_PATH)

# GENERATE JSONL FILES (portable relative paths)
for split in ["train", "validation", "test"]:
    split_df = df[df["split"] == split]
    output_file = os.path.join(OUTPUT_DIR, f"{split}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in split_df.iterrows():

            # Relative path — works on both local PC and Kaggle
            image_path = "/".join([
                "dataset_32images",
                row["instrument"],
                row["split"],
                row["image_name"]
            ])

            local_image = os.path.join(
                r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\dataset_32images",
                row["instrument"],
                row["split"],
                row["image_name"]
            )

            if not os.path.exists(local_image):
                raise FileNotFoundError(f"Missing image: {local_image}")

            sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an expert on Assamese musical instruments."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path
                            },
                            {
                                "type": "text",
                                "text": row["question"]
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": row["answer"]
                            }
                        ]
                    }
                ]
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"{split}.jsonl created with {len(split_df)} samples.")

print("\nDone!")
print(f"JSONL files saved to: {OUTPUT_DIR}")
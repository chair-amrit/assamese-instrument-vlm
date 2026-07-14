import pandas as pd
import random
import os

# ============ FILL THESE PATHS ============
ANSWERS_TABLE_PATH = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\Assamese_Musical_Instrument_answers.csv"   # <- 1. path to your answers spreadsheet (csv/xlsx)
QUESTIONS_TABLE_PATH = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\Assamese_Musical_Instrument_questions.csv" # <- 2. path to your questions spreadsheet (csv/xlsx)
IMAGES_ROOT_PATH = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\dataset_32images"     # <- 3. path to root "Images" folder containing 7 instrument folders
OUTPUT_CSV_PATH = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\final_vqa.csv"       # <- 4. where to save the final generated table

# ============ CONFIG ============
CONCEPTS = [
    "festival", "origin", "material", "parts", "sound",
    "traditional_player", "playing_method", "instrument_type", "description"
]
# 5. Make sure these concept names EXACTLY match your questions/answers table column names

INSTRUMENTS = ["bahi", "toka", "khutitaal", "xutuli", "bihu_dhol", "pepa", "gogona"]
# 6. Confirm these match your folder names exactly (case-sensitive)

SPLITS = ["train", "validation", "test"]
# 7. Confirm your image filenames follow pattern: <instrument>_<split><number>.<ext>
#    e.g., bahi_train1.jpg, bahi_val1.jpg, bahi_test1.jpg

IMAGE_EXT = ".jpg"  # 8. change if your images are .png or mixed

random.seed(42)  # 9. change/remove seed if you want different randomization each run


def load_tables():
    answers_df = pd.read_csv(ANSWERS_TABLE_PATH) if ANSWERS_TABLE_PATH.endswith(".csv") else pd.read_excel(ANSWERS_TABLE_PATH)
    questions_df = pd.read_csv(QUESTIONS_TABLE_PATH) if QUESTIONS_TABLE_PATH.endswith(".csv") else pd.read_excel(QUESTIONS_TABLE_PATH)
    return answers_df, questions_df


def build_lookup_dicts(answers_df, questions_df):
    # answers_df: rows = instruments, columns = concepts
    answers_dict = answers_df.set_index("instrument").to_dict(orient="index")

    # questions_df: rows = concepts, columns = questionid, concept, phrase_1, phrase_2, phrase_3
    questions_dict = questions_df.set_index("concept").to_dict(orient="index")

    return answers_dict, questions_dict


def get_image_files(instrument_folder, split):
    split_folder = os.path.join(instrument_folder, split)
    # 10. Confirm this matches your actual folder nesting (Images/instrument/split/*.jpg)
    files = [f for f in os.listdir(split_folder) if f.lower().endswith(IMAGE_EXT)]
    return sorted(files)


def assign_phrase_groups(concepts):
    """Randomly split 9 concepts into two roughly equal groups for phrase_1/phrase_2."""
    shuffled = concepts.copy()
    random.shuffle(shuffled)
    mid = len(shuffled) // 2  # 4 or 5
    group_a = shuffled[:mid]
    group_b = shuffled[mid:]
    return group_a, group_b


def generate_table():
    answers_df, questions_df = load_tables()
    answers_dict, questions_dict = build_lookup_dicts(answers_df, questions_df)

    rows = []

    for instrument in INSTRUMENTS:
        instrument_folder = os.path.join(IMAGES_ROOT_PATH, instrument)
        # 11. Confirm instrument_folder path resolves correctly

        for split in SPLITS:
            image_files = get_image_files(instrument_folder, split)

            group_a, group_b = [], []
            for idx, image_file in enumerate(image_files):
                image_name = image_file

                if split == "train":
                    # Randomly assign each image's 9 concepts into phrase_1-group / phrase_2-group
                    # Alternate which group gets phrase_1 vs phrase_2 per image for extra balance
                    if idx % 2 == 0:
                        group_a, group_b = assign_phrase_groups(CONCEPTS)
                        phrase_1_concepts, phrase_2_concepts = group_a, group_b
                    else:
                        phrase_1_concepts, phrase_2_concepts = group_b, group_a

                    for concept in CONCEPTS:
                        phrase_used = "phrase_1" if concept in phrase_1_concepts else "phrase_2"
                        question_text = questions_dict[concept][phrase_used]
                        answer_text = answers_dict[instrument][concept]

                        rows.append({
                            "image_name": image_name,
                            "instrument": instrument,
                            "split": split,
                            "concept": concept,
                            "question_id": questions_dict[concept]["question_id"],
                            "phrase_used": phrase_used,
                            "question": question_text,
                            "answer": answer_text
                        })

                else:
                    # val/test always use phrase_3 (original)
                    for concept in CONCEPTS:
                        question_text = questions_dict[concept]["phrase_3"]
                        answer_text = answers_dict[instrument][concept]

                        rows.append({
                            "image_name": image_name,
                            "instrument": instrument,
                            "split": split,
                            "concept": concept,
                            "question_id": questions_dict[concept]["question_id"],
                            "phrase_used": "phrase_3",
                            "question": question_text,
                            "answer": answer_text
                        })

    final_df = pd.DataFrame(rows)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"Final table saved: {len(final_df)} rows -> {OUTPUT_CSV_PATH}")
    return final_df


# ============ RUN ============
final_df = generate_table()
print(final_df.head(20))
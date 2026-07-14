import pandas as pd

df = pd.read_csv(r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\final_vqa.csv")

print("=== BASIC INFO ===")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\n=== SPLIT COUNTS ===")
print(df['split'].value_counts())

print("\n=== ROWS PER INSTRUMENT ===")
print(df['instrument'].value_counts())

print("\n=== ROWS PER CONCEPT ===")
print(df['concept'].value_counts())

print("\n=== PHRASE USAGE (train only) ===")
train_df = df[df['split'] == 'train']
print(train_df['phrase_used'].value_counts())

print("\n=== VAL/TEST PHRASE CHECK (should be phrase_3 only) ===")
non_train = df[df['split'] != 'train']
print(non_train['phrase_used'].value_counts())

print("\n=== NULL CHECK ===")
print(df.isnull().sum())

print("\n=== SAMPLE ROWS ===")
print(df.sample(5).to_string())

print("\n=== CHECK: EVERY IMAGE HAS EXACTLY 9 ROWS ===")
rows_per_image = df.groupby("image_name").size()

if (rows_per_image == 9).all():
    print("PASS: Every image has exactly 9 rows.")
else:
    print("FAIL: Some images do not have exactly 9 rows.")
    print(rows_per_image[rows_per_image != 9])


print("\n=== CHECK: (image_name, question_id) UNIQUENESS ===")
duplicates = df.duplicated(subset=["image_name", "question_id"], keep=False)

if not duplicates.any():
    print("PASS: Every (image_name, question_id) pair is unique.")
else:
    print("FAIL: Duplicate (image_name, question_id) pairs found.")
    print(df.loc[duplicates, ["image_name", "question_id"]]
          .sort_values(["image_name", "question_id"]))


print("\n=== CHECK: TRAINING IMAGES HAVE BOTH PHRASE_1 AND PHRASE_2 ===")

phrase_check = (
    train_df.groupby("image_name")["phrase_used"]
    .apply(lambda x: set(x))
)

bad_images = phrase_check[
    phrase_check.apply(lambda s: not {"phrase_1", "phrase_2"}.issubset(s))
]

if len(bad_images) == 0:
    print("PASS: Every training image contains both Phrase 1 and Phrase 2.")
else:
    print("FAIL: Some training images are missing Phrase 1 or Phrase 2.")
    print(bad_images)


print("\n=== CHECK: QUESTION IDs PER IMAGE ===")

qid_check = (
    df.groupby("image_name")["question_id"]
    .nunique()
)

if (qid_check == 9).all():
    print("PASS: Every image has all 9 unique question IDs.")
else:
    print("FAIL: Some images have missing or duplicate question IDs.")
    print(qid_check[qid_check != 9])
import pandas as pd
import json

df = pd.read_csv("dataset/Assamese_Musical_Instrument_Dataset.csv")

dataset = []

for _,row in df.iterrows():
    sample = {
        "image": f"images/{row['Image_Name']}",
        "question": row["Question"],
        "answer": row["Answer"]
    }

    dataset.append(sample)

with open("dataset.json" , "w" , encoding="utf-8") as f:
    json.dump(dataset, f , indent=2 , ensure_ascii=False)

print("Done")
print("Samples:", len(dataset))
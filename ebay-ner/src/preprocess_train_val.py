import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

INPUT_PATH = Path("data/raw/Tagged_Titles_Train.tsv.gz")
OUTPUT_DIR = Path("data/processed")

def split_train_val(df):
    print(" Columns:", df.columns.tolist())
    if "RecordNumber" not in df.columns:
        raise ValueError(" 'RecordNumber' column not found in input!")
    splitter = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["RecordNumber"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

def fill_tags(tag_list):
    filled = []
    last_tag = None
    for tag in tag_list:
        if tag == "":
            filled.append(last_tag)
        else:
            filled.append(tag)
            last_tag = tag
    return filled

def convert_to_bio_format(df):
    records = []
    grouped = df.groupby("RecordNumber")
    for record_id, group in grouped:
        tokens = group["Token"].tolist()
        raw_tags = fill_tags(group["Tag"].tolist())

        bio_tags = []
        prev_tag = "O"
        for tag in raw_tags:
            if tag == "O" or tag is None or tag == "":
                bio_tags.append("O")
                prev_tag = "O"
            elif tag != prev_tag:
                bio_tags.append(f"B-{tag}")
                prev_tag = tag
            else:
                bio_tags.append(f"I-{tag}")

        records.append({
            "record_id": int(record_id),
            "tokens": tokens,
            "tags": bio_tags
        })
    return records

def main():
    print(" Loading tagged data ...")
    df = pd.read_csv(INPUT_PATH, sep="\t", compression="gzip", keep_default_na=False)
    df = df.rename(columns={"Record Number": "RecordNumber"})  # Ensure column is named properly

    print(f" Loaded {len(df)} rows")
    print(" Splitting into train and val ...")
    train_df, val_df = split_train_val(df)

    print(" Converting to BIO format ...")
    train_bio = convert_to_bio_format(train_df)
    val_bio = convert_to_bio_format(val_df)

    print(f" Saving to {OUTPUT_DIR} ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "train.bio.json", "w", encoding="utf-8") as f:
        json.dump(train_bio, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "val.bio.json", "w", encoding="utf-8") as f:
        json.dump(val_bio, f, ensure_ascii=False, indent=2)

    print(f" Done. Train: {len(train_bio)} samples | Val: {len(val_bio)} samples")

if __name__ == "__main__":
    main()

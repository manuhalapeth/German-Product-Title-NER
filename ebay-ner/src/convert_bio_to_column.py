import json
from pathlib import Path

def convert(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            tokens = record["tokens"]
            tags = record["tags"]
            for token, tag in zip(tokens, tags):
                f.write(f"{token} {tag}\n")
            f.write("\n")

if __name__ == "__main__":
    base_input = Path("data/processed")
    base_output = Path("data/corpus")
    base_output.mkdir(parents=True, exist_ok=True)

    print(" Converting train.bio.json → train.txt")
    convert(base_input / "train.bio.json", base_output / "train.txt")

    print(" Converting val.bio.json → dev.txt")
    convert(base_input / "val.bio.json", base_output / "dev.txt")

    print(" Done: Files saved in data/corpus/")

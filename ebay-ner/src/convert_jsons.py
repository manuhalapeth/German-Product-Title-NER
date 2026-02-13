import os
import json
from pathlib import Path

"""
    Convert a JSON-formatted NER dataset to Flair text format.

    Args:
        json_path (str): Path to input JSON file containing {"tokens": [...], "tags": [...]}.
        output_path (str): Path to write the output Flair-formatted text file.

    Output format per line:
        token tag
    Empty line separates sentences.
    """
def convert_to_flair(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for entry in entries:
            tokens = entry["tokens"]
            tags = entry["tags"]
            for tok, tag in zip(tokens, tags):
                out_f.write(f"{tok} {tag}\n")
            out_f.write("\n")

def convert_all(train_json, val_json, test_json, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    convert_to_flair(train_json, os.path.join(output_folder, "train.txt"))
    convert_to_flair(val_json, os.path.join(output_folder, "dev.txt"))
    convert_to_flair(test_json, os.path.join(output_folder, "test.txt"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="data/corpus")

    args = parser.parse_args()
    convert_all(args.train_json, args.val_json, args.test_json, args.output_folder)

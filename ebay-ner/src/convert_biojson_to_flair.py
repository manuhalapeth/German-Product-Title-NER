import json
from pathlib import Path

def convert_biojson_to_flair(input_json_path, output_txt_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for entry in data:
            tokens = entry["tokens"]
            tags = entry["tags"]
            for token, tag in zip(tokens, tags):
                f.write(f"{token} {tag}\n")
            f.write("\n")

# Create directory if not exists
Path("data/corpus").mkdir(parents=True, exist_ok=True)

#  Update with correct paths here:
convert_biojson_to_flair("data/processed/train.bio.json", "data/corpus/train.txt")
convert_biojson_to_flair("data/processed/val.bio.json", "data/corpus/dev.txt")

print(" Conversion complete! Files saved to data/corpus/")

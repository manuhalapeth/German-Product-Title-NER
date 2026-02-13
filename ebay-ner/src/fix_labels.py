import json
from pathlib import Path

def convert_tags_to_bio_format(samples):
    cleaned_samples = []
    for record in samples:
        tokens = record.get("tokens", [])
        tags = record.get("tags", [])
        if len(tokens) != len(tags):
            print(f" Skipping record with mismatched lengths: {record.get('record_id')}")
            continue

        bio_tags = []
        prev_tag = "O"
        for tag in tags:
            if tag == "":
                if prev_tag == "O":
                    bio_tags.append("O")
                else:
                    bio_tags.append("I-" + prev_tag)
            elif tag == "O":
                bio_tags.append("O")
                prev_tag = "O"
            else:
                bio_tags.append("B-" + tag)
                prev_tag = tag

        record["tags"] = bio_tags
        cleaned_samples.append(record)

    return cleaned_samples

def save_bio_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f" Saved: {output_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    input_train = base_dir / "data" / "processed" / "train.bio.json"
    input_val = base_dir / "data" / "processed" / "val.bio.json"
    output_train = base_dir / "data" / "processed" / "train.bio.json"   # overwrite
    output_val = base_dir / "data" / "processed" / "val.bio.json"       # overwrite

    for infile, outfile in [(input_train, output_train), (input_val, output_val)]:
        with open(infile, "r", encoding="utf-8") as f:
            data = json.load(f)
        cleaned = convert_tags_to_bio_format(data)
        save_bio_json(cleaned, outfile)


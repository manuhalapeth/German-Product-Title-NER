import json
from pathlib import Path

# Paths
INPUT_PATH = Path("data/predictions/quiz_pred.json")
OUTPUT_PATH = Path("data/submissions/submission.tsv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load predictions
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

submission = []

# Process each record
for entry in data:
    record_id = str(entry["record_id"])
    category_id = str(entry["category_id"])
    tokens = entry["tokens"]
    tags = entry["tags"]

    current_aspect = None
    current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-") or tag.startswith("S-"):
            # Flush previous
            if current_aspect and current_tokens:
                submission.append([record_id, category_id, current_aspect, " ".join(current_tokens)])
            current_aspect = tag.split("-", 1)[1]
            current_tokens = [token]

        elif tag.startswith("I-") or tag.startswith("E-"):
            if current_aspect:
                current_tokens.append(token)
            else:
                # Edge case: I- without B-
                current_aspect = tag.split("-", 1)[1]
                current_tokens = [token]

        else:  # tag == "O" or unrecognized
            if current_aspect and current_tokens:
                submission.append([record_id, category_id, current_aspect, " ".join(current_tokens)])
            current_aspect = None
            current_tokens = []

    # Final flush
    if current_aspect and current_tokens:
        submission.append([record_id, category_id, current_aspect, " ".join(current_tokens)])

# Save to TSV
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for row in submission:
        f.write("\t".join(row) + "\n")

print(f" Submission file written to {OUTPUT_PATH}")

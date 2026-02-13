import json
from collections import Counter

def analyze_tag_distribution(json_path): # Analyze and print the frequency of each tag in a JSON dataset.
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tag_counter = Counter() # Count tags across all samples
    for sample in data:
        for tag in sample["tags"]:
            tag_counter[tag] += 1

    print("Tag Frequencies (sorted):") # Print frequencies sorted from most to least common
    for tag, count in tag_counter.most_common():
        print(f"{tag}: {count}")

if __name__ == "__main__":
    analyze_tag_distribution("data/processed/train.bio.json")

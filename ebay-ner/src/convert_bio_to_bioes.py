import os
import sys
from typing import List

def bio_to_bioes(tags: List[str]) -> List[str]:
    """
    Convert a list of BIO tags to BIOES format.

    Args:
        tags (List[str]): List of BIO tags (e.g., ['B-PER', 'I-PER', 'O']).

    Returns:
        List[str]: Corresponding list of BIOES tags.
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
            continue

        tag_type = tag[2:]
        prefix = tag[:1]

        if prefix == 'B':
            if i + 1 != len(tags) and tags[i + 1][2:] == tag_type and tags[i + 1][:1] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append('S-' + tag_type)
        elif prefix == 'I':
            if i + 1 != len(tags) and tags[i + 1][2:] == tag_type and tags[i + 1][:1] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append('E-' + tag_type)
        else:
            raise Exception(f"Invalid BIO tag found: {tag}")
    return new_tags



    """
    Convert a BIO-formatted file to BIOES format line-by-line.

    Args:
        input_path (str): Path to the input BIO file.
        output_path (str): Path to write the BIOES-formatted output.
    """
def convert_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    sentences = []
    current_tokens = []
    current_tags = []

    for line in lines:
        line = line.strip()
        if line == "":
            if current_tokens:
                bioes_tags = bio_to_bioes(current_tags)
                for token, tag in zip(current_tokens, bioes_tags):
                    sentences.append(f"{token} {tag}\n")
                sentences.append("\n")
                current_tokens = []
                current_tags = []
        else:
            try:
                token, tag = line.split()
            except ValueError:
                print(f"Skipping malformed line: {line}")
                continue
            current_tokens.append(token)
            current_tags.append(tag)

    # Handle last sentence
    if current_tokens:
        bioes_tags = bio_to_bioes(current_tags)
        for token, tag in zip(current_tokens, bioes_tags):
            sentences.append(f"{token} {tag}\n")
        sentences.append("\n")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(sentences)

    print(f" Converted {input_path} to BIOES format at {output_path}")

 
    """
    Convert all splits (train, dev, test) from BIO to BIOES format
    in the `data/corpus` directory and save to `data/corpus_bioes`.
    """
def main():
    base_input_dir = "data/corpus"
    base_output_dir = "data/corpus_bioes"

    os.makedirs(base_output_dir, exist_ok=True)

    for split in ["train.txt", "dev.txt", "test.txt"]:
        input_file = os.path.join(base_input_dir, split)
        output_file = os.path.join(base_output_dir, split)
        if os.path.exists(input_file):
            convert_file(input_file, output_file)
        else:
            print(f" File not found: {input_file}")


if __name__ == "__main__":
    main()


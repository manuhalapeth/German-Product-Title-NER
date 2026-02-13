# src/convert_space_to_tab.py

import os
# Files to convert: mapping of dataset split name → input file path
FILES_TO_CONVERT = {
    "train": "data/corpus_bioes/train.txt",
    "dev": "data/corpus_bioes/dev.txt",
    "test": "data/corpus_bioes/test.txt",
}
# Output directory for tab-separated files
OUTPUT_DIR = "src/corpus_bioes"

"""
    Convert a text file from space-separated to tab-separated format.
    
    Args:
        input_path (str): Path to the input space-separated file.
        output_path (str): Path to save the tab-separated output file.
    
    Behavior:
        - Each line should have a token and its tag separated by a space.
        - Converts the first space in each line to a tab.
        - Preserves empty lines as sentence separators.
        - Malformed lines (no space) are written as-is? (double check it)
    """
def convert_space_to_tab(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            stripped = line.strip()
            if stripped == "":
                outfile.write("\n")
                continue
            parts = stripped.split(" ", 1)
            if len(parts) == 2:
                outfile.write(parts[0] + "\t" + parts[1] + "\n")
            else:
                # If line has no space (only token or malformed), write as is
                outfile.write(line)
    print(f" Converted {input_path} to {output_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, input_file in FILES_TO_CONVERT.items():
        output_file = os.path.join(OUTPUT_DIR, f"{name}_tab.txt")
        convert_space_to_tab(input_file, output_file)

if __name__ == "__main__":
    main()

import os

def fix_ner_txt_format(input_file, output_file):
    fixed_lines = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    prev_tag = None
    last_line_was_blank = True  # so first line doesn't trigger extra break

    for line in lines:
        stripped = line.strip()

        # If it's an empty line, just add it if not already added
        if not stripped:
            if not last_line_was_blank:
                fixed_lines.append('\n')
                last_line_was_blank = True
            continue

        token_parts = stripped.rsplit(' ', 1)

        # Token and tag extraction
        if len(token_parts) == 2:
            token, tag = token_parts
        else:
            token = token_parts[0]
            tag = "O"

        # Add break if:
        # - tag starts with 'B-' (beginning of a new entity),
        # - previous tag was not I-<same_type> (not continuation),
        # - last line wasn’t already blank
        if tag.startswith('B-'):
            if prev_tag is not None:
                prev_type = prev_tag[2:] if prev_tag.startswith('I-') else None
                curr_type = tag[2:]

                if prev_tag != f"I-{curr_type}" and not last_line_was_blank:
                    fixed_lines.append('\n')
                    last_line_was_blank = True

        fixed_lines.append(f"{token} {tag}\n")
        prev_tag = tag
        last_line_was_blank = False

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(fixed_lines)

if __name__ == "__main__":
    os.makedirs("data/corpus", exist_ok=True)

    for split in ['train', 'dev', 'test']:
        path = os.path.join("data", "corpus", f"{split}.txt")
        if os.path.exists(path):
            print(f" Fixing: {path}")
            fix_ner_txt_format(path, path)
        else:
            print(f" File not found: {path}")

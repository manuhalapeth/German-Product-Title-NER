from transformers import AutoTokenizer
# Initialize the tokenizer from the XLM-RoBERTa large model
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

print("🔍 Checking for token → subtoken splits...\n")
# Counters to track how many tokens are split into multiple subtokens
split_count = 0
total = 0
"""
    Reads a token-per-line file and checks how many tokens are split
    into multiple subtokens by the tokenizer.

    Args:
        file_path (str): Path to the tokenized training file. Each line
                         should contain a token (optionally followed by labels).

    Returns:
        None. Prints out tokens that split and a summary at the end.
    """
with open("src/corpus_bioes/train_tab.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        token = line.split()[0]# assume the first element is the token
        subtokens = tokenizer.tokenize(token)  # tokenize using XLM-RoBERTa
        total += 1
        if len(subtokens) > 1:
            print(f" Token '{token}' splits into {subtokens}")
            split_count += 1

print(f"\n Done. {split_count}/{total} tokens split into subtokens.")

# predict_train.py

from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

# Load the tagger model
try:
    tagger = SequenceTagger.load("data/outputs/flair-de-ner/best-model.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'data/outputs/flair-de-ner/best-model.pt' exists and is a valid Flair model.")
    exit()

# Read and parse the training file
try:
    with open("src/corpus_bioes/train_tab.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
except FileNotFoundError:
    print("Error: 'src/corpus_bioes/train_tab.txt' not found.")
    exit()
except Exception as e:
    print(f"Error reading train_tab.txt: {e}")
    exit()

# Process into sentence strings
sentences_text = []
current_tokens_list = []
for line in lines:
    line = line.strip()
    if line == "":
        if current_tokens_list:
            sentences_text.append(" ".join(current_tokens_list))
            current_tokens_list = []
        continue
    parts = line.split("\t")
    # We only need the token text for creating the Sentence object for prediction
    if len(parts) >= 1: # Ensure there's at least a token part
        current_tokens_list.append(parts[0])

# Add last sentence if file doesn’t end with a blank line
if current_tokens_list:
    sentences_text.append(" ".join(current_tokens_list))

# Predict and write output
try:
    with open("train_predictions.tsv", "w", encoding="utf-8") as out_f:
        for text in tqdm(sentences_text, desc="Predicting"):
            sent = Sentence(text)
            tagger.predict(sent)

            for token in sent.tokens: # Iterate through the token objects within the sentence
                tag_value = "O" # Default tag value if no specific tag is found

                # --- START OF CRITICAL CORRECTION ---
                # Attempt to get the 'ner' tag value using different methods
                # based on common Flair versions, given 'get_tag()' is failing.

                # Method 1: Direct 'tag' attribute (common in older versions, and often used by taggers)
                if hasattr(token, 'tag') and token.tag is not None:
                    # Check if 'token.tag' is a Tag object or just a string
                    if hasattr(token.tag, 'value'): # It's a Tag object
                        tag_value = token.tag.value
                    else: # It's already the string value
                        tag_value = token.tag
                # Method 2: 'tags' dictionary access (sometimes used for multiple tag types)
                elif hasattr(token, 'tags') and 'ner' in token.tags and token.tags['ner'] is not None:
                    # Check if token.tags['ner'] is a Tag object or just a string
                    if hasattr(token.tags['ner'], 'value'):
                        tag_value = token.tags['ner'].value
                    else:
                        tag_value = token.tags['ner']
                # --- END OF CRITICAL CORRECTION ---

                out_f.write(f"{tag_value}\t{token.text}\n")

except Exception as e:
    print(f"An error occurred during prediction or writing: {e}")
    # Print more specific error details if possible for debugging
    # For example, if 'token' is available:
    # print(f"Error occurred at token: {token.text}")

print(" Prediction complete. Results written to train_predictions.tsv")


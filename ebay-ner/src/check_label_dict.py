from flair.models import SequenceTagger

# Load model directly
tagger = SequenceTagger.load("data/outputs/flair-de-ner/best-model.pt")

# Access the label dictionary used for NER
label_dict = tagger.label_dictionary

print(" Tags used in the trained model:")
for tag in label_dict.get_items():
    print(tag)

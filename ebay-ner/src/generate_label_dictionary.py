from flair.datasets import ColumnCorpus
from flair.data import Dictionary
from pathlib import Path

def main():
    data_folder = Path("data/corpus")
    label_type = "ner"

    # Define the column structure of your CoNLL file
    columns = {0: "text", 1: label_type}

    # Load your corpus
    corpus = ColumnCorpus(data_folder, columns, train_file="train.txt", dev_file="dev.txt")

    # Create the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    # Save it to disk
    output_path = Path("resources/taggers/ner_tags.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_dict.save(output_path)
    print(f" Saved label dictionary to {output_path}")

if __name__ == "__main__":
    main()

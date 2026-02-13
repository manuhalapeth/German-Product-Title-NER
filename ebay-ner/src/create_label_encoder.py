from flair.datasets import ColumnCorpus
from flair.data import Dictionary

columns = {0: 'text', 1: 'ner'}
data_folder = 'data/corpus'
label_type = 'ner'

corpus = ColumnCorpus(data_folder, columns, train_file='train.txt', dev_file='dev.txt', test_file=None)

# Create label dictionary
label_dict: Dictionary = corpus.make_label_dictionary(label_type=label_type)

# Add 'O' label if missing
if 'O' not in label_dict.get_items():
    label_dict.add_item('O')

# Save label dictionary
label_dict.save('resources/taggers/ner_tags.pkl')
print("Label dictionary saved to resources/taggers/ner_tags.pkl")
print(label_dict)


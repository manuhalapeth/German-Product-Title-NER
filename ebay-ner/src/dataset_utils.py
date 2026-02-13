import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

"""
    PyTorch Dataset for Named Entity Recognition (NER).

    Reads a JSON file containing tokenized text and corresponding BIO/BIOES labels,
    and prepares it for transformer-based models like BERT.

    Args:
        path (str): Path to the JSON file. Each entry should have 'tokens' and 'tags'.
        tokenizer (PreTrainedTokenizerFast): HuggingFace tokenizer for encoding tokens.
        label2id (dict): Mapping from label string to integer ID.
        max_len (int, optional): Maximum sequence length for padding/truncation. Default: 128
    """
class NERDataset(Dataset):
    def __init__(self, path, tokenizer, label2id, max_len=128):
        with open(path, "r") as f:
            self.samples = json.load(f)

        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)


        """
        Returns a single encoded sample from the dataset.

        Returns:
            dict: {
                "input_ids": Tensor of token IDs,
                "attention_mask": Tensor of attention mask,
                "labels": Tensor of label IDs aligned to tokens
            }
        """
    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        labels = sample["tags"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Align labels to tokenized input
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(self.label2id[labels[word_idx]])
            else:
                # For subwords, we can optionally assign the same tag or -100
                label_ids.append(self.label2id[labels[word_idx]])
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids)
        }

"""
    Custom collate function to combine samples into a batch for DataLoader.

    Args:
        batch (list[dict]): List of samples returned by NERDataset.__getitem__

    Returns:
        dict: Batched tensors for 'input_ids', 'attention_mask', and 'labels'
    """
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

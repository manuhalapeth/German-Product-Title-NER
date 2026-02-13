import torch.nn as nn
from transformers import BertModel

class BertTagger(nn.Module):
    """
    BERT-based sequence tagging model for Named Entity Recognition (NER) or similar token-level tasks.

    Architecture:
        - Pretrained BERT encoder
        - Dropout layer for regularization
        - Linear classifier projecting hidden states to label space

    Args:
        model_name (str): Name or path of the pretrained BERT model (HuggingFace).
        num_labels (int): Number of output labels/classes.
    """
    def __init__(self, model_name, num_labels):
        super(BertTagger, self).__init__()
        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(model_name)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Linear classifier to predict labels for each token
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for sequence tagging.

        Args:
            input_ids (torch.Tensor): Token IDs, shape [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask, shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Logits for each token, shape [batch_size, seq_len, num_labels]
        """
        # Get hidden states from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]

        # Apply dropout for regularization
        sequence_output = self.dropout(sequence_output)

        # Linear layer to project to label space
        logits = self.classifier(sequence_output)  # shape: [batch_size, seq_len, num_labels]
        return logits


def build_model(config):
    """
    Helper function to instantiate BertTagger using a config dictionary.

    Args:
        config (dict): Configuration dictionary containing at least 'label2id' mapping and 'model_name'.

    Returns:
        BertTagger: Initialized model ready for training.
    """
    num_labels = len(config["label2id"])
    model_name = config.get("model_name", "bert-base-uncased")
    return BertTagger(model_name=model_name, num_labels=num_labels)

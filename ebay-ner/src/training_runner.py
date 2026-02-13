# ===========================
# training_running.py
# ===========================

import torch
from pathlib import Path
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import (
    TransformerWordEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
    StackedEmbeddings,
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import os
import shutil

# ===========================
# 1️⃣ Training Function
# ===========================

def run_training(config, trial):
    """
    Train a Flair SequenceTagger using a given configuration and trial information.

    Args:
        config (dict): Dictionary containing dataset paths, model parameters, and training settings.
        trial (object): Trial object (used for tracking/trial-specific output naming).

    Returns:
        float: F1-score on the development set based on the configured evaluation metric.
    """
    # --- Load corpus ---
    data_folder = config["data"]["corpus_dir"]
    train_file = config["data"]["train_file"]
    test_file = config["data"]["test_file"]
    dev_file = config["data"]["dev_file"]
    tag_type = config["data"]["label_type"]

    # Column mapping for Flair ColumnCorpus
    column_format = {0: "text", 1: tag_type}

    corpus = ColumnCorpus(
        data_folder,
        column_format,
        train_file=train_file,
        test_file=test_file,
        dev_file=dev_file,
    )

    # Create tag dictionary
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    # --- Set up embeddings ---
    embeddings_list = []
    for emb in config["model"]["embeddings"]:
        if emb.startswith("flair:"):
            embeddings_list.append(FlairEmbeddings(emb.replace("flair:", "")))
        elif emb == "char":
            embeddings_list.append(CharacterEmbeddings())
        else:
            embeddings_list.append(TransformerWordEmbeddings(model=emb))

    embeddings = StackedEmbeddings(embeddings=embeddings_list)

    # --- Initialize the SequenceTagger ---
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=config["model"].get("use_crf", True),
        use_rnn=config["model"].get("use_rnn", False),
        word_dropout=config["model"].get("word_dropout", 0.05),
        locked_dropout=config["model"].get("locked_dropout", 0.1),
        reproject_embeddings=True,
    )

    # --- Output directory for this trial ---
    output_dir = Path("data/outputs") / f"{trial.number}_best_0.000198"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize trainer ---
    trainer = ModelTrainer(tagger, corpus)

    # Map user-friendly monitor keys to Flair evaluation metrics
    monitor_map = {
        "micro_F1_score": ("micro avg", "f1-score"),
        "macro_F1_score": ("macro avg", "f1-score"),
        "weighted_F1_score": ("weighted avg", "f1-score"),
    }
    monitor_key = config["trainer"].get("monitor", "micro_F1_score")
    main_eval_metric = monitor_map.get(monitor_key, ("micro avg", "f1-score"))

    # --- Train the model ---
    trainer.train(
        base_path=config["output"]["output_dir"],
        learning_rate=config["trainer"]["learning_rate"],
        mini_batch_size=config["trainer"]["mini_batch_size"],
        mini_batch_chunk_size=config["trainer"]["mini_batch_chunk_size"],
        max_epochs=config["trainer"]["max_epochs"],
        patience=config["trainer"]["patience"],
        embeddings_storage_mode=config["trainer"]["embeddings_storage_mode"],
        checkpoint=config["trainer"]["checkpoint"],
        save_final_model=config["trainer"]["save_final_model"],
        train_with_dev=False,
        monitor_train=False,
        monitor_test=False,
        main_evaluation_metric=("macro avg", "f1-score"),
    )

    # --- Load the best model ---
    model_path = output_dir / "best-model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {model_path}. Training may not have finished successfully."
        )

    tagger = SequenceTagger.load(model_path)

    # --- Evaluate on the dev set ---
    result = tagger.evaluate(corpus.dev)

    # Return F1-score for the chosen evaluation metric
    return result.main_evaluation_result.classification_report[main_eval_metric[0]][main_eval_metric[1]]

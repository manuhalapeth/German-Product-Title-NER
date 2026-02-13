"""
Flair NER Training Pipeline with Optuna Hyperparameter Tuning

This script trains a Flair SequenceTagger model for sequence labeling
(NER) using a configurable YAML file. It supports:

- ColumnCorpus loading
- Transformer + Flair + Character embeddings
- CRF / RNN configuration
- Optuna hyperparameter tuning
- Final retraining with best parameters

"""
import os
import sys
import yaml
import torch
import optuna
import numpy as np
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


def load_config(path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        path (str): Path to YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_model(config: dict) -> float:
    """
    Train a Flair SequenceTagger model and return best dev F1 score.

    Steps:
        1. Load corpus from ColumnCorpus
        2. Build label dictionary
        3. Construct embeddings stack
        4. Initialize SequenceTagger
        5. Train model
        6. Extract best dev F1

    Args:
        config (dict): Configuration dictionary.

    Returns:
        float: Best development F1 score achieved.
    """
    # --- Load corpus ---
    corpus_dir = config["data"]["corpus_dir"]
    column_format = {0: "text", 1: config["data"]["label_type"]}
    corpus: Corpus = ColumnCorpus(
        corpus_dir,
        column_format,
        train_file=config["data"]["train_file"],
        dev_file=config["data"]["dev_file"],
        test_file=config["data"]["test_file"],
        in_memory=True,
    )
    print(f"Loaded {len(corpus.train)} train / {len(corpus.dev)} dev / {len(corpus.test)} test sentences")

    # --- Labels & embeddings ---
    label_type = config["data"]["label_type"]
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    emb_list = []
    for emb in config["model"]["embeddings"]:
        if emb.startswith("flair:"):
            emb_list.append(FlairEmbeddings(emb.replace("flair:", "")))
        elif emb == "char":
            emb_list.append(CharacterEmbeddings())
        else:
            emb_list.append(
                TransformerWordEmbeddings(
                    model=emb,
                    fine_tune=config["model"].get("fine_tune", True),
                    subtoken_pooling="first",
                )
            )
    embeddings = StackedEmbeddings(emb_list)

    # --- Tagger ---
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type=label_type,
        use_crf=config["model"].get("use_crf", True),
        use_rnn=config["model"].get("use_rnn", False),
        word_dropout=config["model"].get("word_dropout", 0.05),
        locked_dropout=config["model"].get("locked_dropout", 0.1),
        reproject_embeddings=True,
    )

    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Trainer ---
    trainer = ModelTrainer(tagger, corpus)

    # --- Only optimizer-safe parameters! ---
    trainer_result = trainer.train(
        base_path=output_dir,
        learning_rate=config["trainer"]["learning_rate"],
        mini_batch_size=config["trainer"]["mini_batch_size"],
        max_epochs=config["trainer"]["max_epochs"],
        patience=config["trainer"]["patience"],
        optimizer=torch.optim.AdamW,
        embeddings_storage_mode="cpu",
        use_amp=False,
    )

    # --- Extract best F1 ---
    best_f1 = 0.0
    if "dev_score_history" in trainer_result:
        best_f1 = max(trainer_result["dev_score_history"])
    elif "dev_score" in trainer_result:
        best_f1 = max(trainer_result["dev_score"]) if isinstance(trainer_result["dev_score"], list) else trainer_result["dev_score"]

    print(f" Training done. Best dev F1 = {best_f1:.4f}")
    return best_f1


def main():
    cfg_path = "config/training_config.yaml"
    if not os.path.isfile(cfg_path):
        print(f" Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(cfg_path)
    config["trainer"]["learning_rate"] = float(config["trainer"]["learning_rate"])
    config["trainer"]["mini_batch_size"] = int(config["trainer"]["mini_batch_size"])

    # --- Optuna tuning ---
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(n_startup_trials=10))

    def objective(trial: optuna.Trial) -> float:
        tcfg = dict(config)
        tcfg["trainer"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        tcfg["trainer"]["mini_batch_size"] = trial.suggest_categorical("mini_batch_size", [8, 16])
        tcfg["model"]["word_dropout"] = trial.suggest_float("word_dropout", 0.01, 0.2)
        tcfg["model"]["locked_dropout"] = trial.suggest_float("locked_dropout", 0.0, 0.5)

        outdir = Path(config["output"]["output_dir"]) / f"trial_{trial.number}"
        outdir.mkdir(parents=True, exist_ok=True)
        tcfg["output"]["output_dir"] = str(outdir)

        print(f"[Trial {trial.number}] lr={tcfg['trainer']['learning_rate']:.2e}, "
              f"batch={tcfg['trainer']['mini_batch_size']}, "
              f"wd={tcfg['model']['word_dropout']:.2f}, "
              f"ld={tcfg['model']['locked_dropout']:.2f}")

        try:
            return train_model(tcfg)
        except Exception as e:
            print(f"[Trial {trial.number}] failed → {e}")
            return 0.0

    print("🔍 Starting hyperparameter tuning ...")
    study.optimize(objective, n_trials=25)
    print(f" Best params: {study.best_trial.params}")
    print(f" Best dev F1: {study.best_value:.4f}")

    # --- Final model training ---
    best = study.best_trial.params
    config["trainer"]["learning_rate"] = best.get("learning_rate", config["trainer"]["learning_rate"])
    config["trainer"]["mini_batch_size"] = best.get("mini_batch_size", config["trainer"]["mini_batch_size"])
    config["model"]["word_dropout"] = best.get("word_dropout", config["model"]["word_dropout"])
    config["model"]["locked_dropout"] = best.get("locked_dropout", config["model"]["locked_dropout"])

    final_dir = Path(config["output"]["output_dir"]) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    config["output"]["output_dir"] = str(final_dir)

    print(" Retraining final model with best params ...")
    final_score = train_model(config)
    print(f" Final model complete → Best F1 = {final_score:.4f}")
    print(f" Models saved to {config['output']['output_dir']}")


if __name__ == "__main__":
    main()

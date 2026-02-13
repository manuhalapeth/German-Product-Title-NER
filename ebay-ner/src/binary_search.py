# bayesian_search.py
import os
import sys
import yaml
import pprint
import shutil
from copy import deepcopy
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

# Import run_training from the random search script
from train_model import run_training

# === Config ===
CONFIG_PATH = "config/training_config.yaml"       # my config path
STUDY_NAME = "random_baseline"                    # must match the random-search study name
STORAGE_URL = "sqlite:///optuna_study.db"         # same DB used by random search
N_TRIALS = 25                                     # number of Bayesian trials to run

# Load YAML config
if not os.path.exists(CONFIG_PATH):
    print(f" Config not found at: {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH, "r") as f:
    base_config = yaml.safe_load(f)

pprint.pprint(base_config)

OUTPUT_BASE = base_config["output"]["output_dir"]  # base output dir from config
os.makedirs(OUTPUT_BASE, exist_ok=True)


def objective(trial: optuna.trial.Trial): #Objective function for Bayesian hyperparameter optimization.
    """
    Main routine to load an existing Optuna study and perform Bayesian optimization.

    Steps:
    1. Load existing study (TPE sampler) from SQLite storage.
    2. Run N_TRIALS Bayesian optimization trials.
    3. Print F1 scores for all trials.
    4. Identify and copy the best model to the base output directory.
    """
    trial_config = deepcopy(base_config)

    # Suggest hyperparameters
    trial_config["trainer"]["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    trial_config["trainer"]["mini_batch_size"] = trial.suggest_categorical("mini_batch_size", [8, 16, 32])
    trial_config["model"]["word_dropout"] = trial.suggest_float("word_dropout", 0.01, 0.2)
    trial_config["model"]["locked_dropout"] = trial.suggest_float("locked_dropout", 0.0, 0.5)
    trial_config["model"]["use_crf"] = trial.suggest_categorical("use_crf", [True, False])

    # Keep output_dir pointing to base; run_training will create trial_{n}
    trial_config["output"]["output_dir"] = OUTPUT_BASE

    try:
        return run_training(trial_config, trial)
    except Exception as e:
        print(f"[⚠️ Trial {trial.number}] Exception during training: {e}")
        return None


def main(): # Main routine to load an existing Optuna study and perform Bayesian optimization.
    # Load existing study with TPE sampler
    try:
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=STORAGE_URL,
            sampler=TPESampler()
        )
        print(f" Loaded existing study: '{STUDY_NAME}'")
    except Exception as e:
        print(f" Could not load study '{STUDY_NAME}' from '{STORAGE_URL}': {e}")
        sys.exit(1)

    # Run Bayesian optimization
    print(f"\n Starting {N_TRIALS} Bayesian (TPE) trials...")
    study.optimize(objective, n_trials=N_TRIALS)

    # Print F1 scores
    print("\n=== All trial F1 scores (trial_number: f1) ===")
    for t in sorted(study.trials, key=lambda x: x.number):
        print(f"Trial {t.number}: F1 = {t.value}")

    # Save best model
    best = study.best_trial
    print(f"\n🎯 Best trial: {best.number} — F1 = {best.value}")
    print("Params:")
    pprint.pprint(best.params)

    best_trial_dir = os.path.join(OUTPUT_BASE, f"trial_{best.number}")
    candidate_paths = [
        os.path.join(best_trial_dir, "best-model.pt"),
        os.path.join(best_trial_dir, "final-model.pt"),
        os.path.join(best_trial_dir, "final-model", "best-model.pt"),
    ]

    src = next((p for p in candidate_paths if os.path.exists(p)), None)
    if not src:
        print(f" Could not find model file for trial {best.number}")
        sys.exit(1)

    dst = os.path.join(OUTPUT_BASE, "best-model.pt")
    shutil.copy(src, dst)
    print(f" Best model copied to: {dst}")


if __name__ == "__main__":
    main()
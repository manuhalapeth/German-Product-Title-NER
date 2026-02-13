import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
from allowed_by_category import is_allowed

# ===========================================================
# 1. MODEL CONFIGURATION
# ===========================================================
tagger_paths = [
    # Only my strongest model (F1 ≈ 0.90)
    "/Users/manuhalapeth/Downloads/eBay_ML_Challenge_2025/"
    "ebay-ner/data/outputs/flair-de-ner/bayesian_search/trial_0/trial_1/final-model.pt"
]

taggers = [SequenceTagger.load(path) for path in tagger_paths]
print(f" Loaded {len(taggers)} model(s).")

# ===========================================================
# 2. LOAD QUIZ DATA
# ===========================================================
with open("data/processed/quiz_tokenized.json", "r", encoding="utf-8") as f:
    records = json.load(f)
print(f" Loaded {len(records)} quiz records.")

# ===========================================================
# 3. ASPECT-SPECIFIC CONFIDENCE THRESHOLDS
# ===========================================================
# Looser thresholds boost recall for high-F1 models
per_aspect_thresholds = {
    "Hersteller": 0.45,
    "Produktart": 0.45,
    "Kompatible_Fahrzeug_Marke": 0.40,
    "Kompatibles_Fahrzeug_Modell": 0.45,
    "Einbauposition": 0.40,
    "Anzahl_Der_Einheiten": 0.40,
    "Im_Lieferumfang_Enthalten": 0.40,
    "Bremsscheiben-Aussendurchmesser": 0.45,
    "Herstellernummer": 0.45,
    "default": 0.40
}
DEFAULT_THRESHOLD = 0.40  # recall-friendly baseline

# ===========================================================
# 4. ENSEMBLE-LIKE PREDICTION (adapted for single model)
# ===========================================================
def predict_weighted(title, category_id, taggers):
    """
    Ensemble-style prediction (keeps weighting and threshold logic)
    but without strict voting, optimized for single strong model.
    """
    span_scores = defaultdict(list)

    for tagger in taggers:
        sentence = Sentence(title, use_tokenizer=True)
        tagger.predict(sentence)
        for span in sentence.get_spans("ner"):
            aspect_name = span.tag[2:] if span.tag.startswith(("B-", "I-", "S-", "E-")) else span.tag
            aspect_value = span.text.strip()
            span_scores[(aspect_name, aspect_value)].append(span.score)

    merged_spans = []
    for (aspect_name, aspect_value), scores in span_scores.items():
        avg_score = sum(scores) / len(scores)
        threshold = per_aspect_thresholds.get(aspect_name, DEFAULT_THRESHOLD)

        if avg_score >= threshold and is_allowed(aspect_name, category_id):
            merged_spans.append((aspect_name, aspect_value))

    return merged_spans

# ===========================================================
# 5. RUN PREDICTIONS
# ===========================================================
output_file = "Fifth_quiz_predictions_trial1.tsv"

print("\n Running predictions...")
with open(output_file, "w", encoding="utf-8") as out:
    for record in tqdm(records, desc="Tagging Quiz Data"):
        record_id = record["record_id"]
        category_id = record["category_id"]
        title = record["title"]

        predicted_spans = predict_weighted(title, category_id, taggers)
        for aspect_name, aspect_value in predicted_spans:
            out.write(f"{record_id}\t{category_id}\t{aspect_name}\t{aspect_value}\n")

print(f"\n Predictions written to {output_file}")

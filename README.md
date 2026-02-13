# German Product Title NER System

A production-grade Named Entity Recognition system for extracting structured product attributes from German e-commerce listing titles. Originally developed for a large-scale automotive e-commerce NER challenge.

## Overview

Extracts 29 product attributes from German product titles: manufacturer info, vehicle compatibility, technical specs, and product identifiers. Combines multilingual transformer embeddings with contextual language models and CRF-based sequence labeling.

## Architecture (Check ARCHITECTURE.MD for more context) 

```
Input Title → Tokenization → Stacked Embeddings → CRF Sequence Tagger → Post-Processing → Structured Output
```

**Embedding Stack:**
- XLM-RoBERTa-large (multilingual transformer, fine-tuned)
- Flair German forward/backward contextual embeddings

**Sequence Tagger:**
- 256-dim hidden layer with CRF output
- BIOES tagging scheme
- Per-aspect confidence thresholding

| Component | Value |
|-----------|-------|
| Base Model | xlm-roberta-large |
| Hidden Size | 256 |
| CRF | Enabled |
| Fine-tuning | Enabled |

## Extracted Attributes

**29 aspects across 6 categories:**

- **Manufacturing**: Manufacturer, part numbers, OEM references
- **Product Classification**: Product type, product line, model, technology
- **Vehicle Compatibility**: Brand, model, year
- **Technical Specs**: Dimensions, viscosity grades, tooth counts
- **Physical Properties**: Size, length, width, thickness, material, color
- **Logistics**: Installation position, included items, unit counts

## Training

**Hyperparameter Optimization:**
- Optuna TPE sampler: 25 random + 25 Bayesian trials
- Search: learning rate (1e-5 to 5e-4), batch size, dropout rates, CRF toggle
- Optimizer: AdamW with early stopping (patience=3)
- Data split: 90/10 GroupShuffleSplit by record (prevents leakage)

## Inference Pipeline

1. Flair sentence tokenization
2. Forward pass through SequenceTagger
3. Span extraction with confidence scores
4. Per-aspect threshold filtering (0.40-0.45 range)
5. Category-aspect compatibility gating
6. Aspect-specific normalization (viscosity → `5W40`, diameter → `Ø300mm`)
7. Fuzzy deduplication (94% similarity threshold)

## Design Tradeoffs

**Why CRF sequence labeling?**
Product titles have strong sequential dependencies. A token labeled as a vehicle brand strongly predicts the next token is a model name. CRF captures these constraints at the sequence level rather than making independent per-token decisions. This matters more than raw accuracy since it produces coherent entity spans.

**Why per-aspect confidence thresholds?**
A single global threshold fails because aspects have different base rates and model confidence distributions. Manufacturer names are high-confidence, while technical specs often have lower scores due to ambiguity. Tuning thresholds per aspect (0.40-0.45) lets us optimize precision/recall independently for each attribute type.

**Why category gating after prediction?**
The model learns general patterns across all categories, but some aspects only apply to specific product categories. Applying category-aspect rules post-prediction (rather than training separate models) keeps the architecture simple while enforcing domain constraints. High-impact aspects like manufacturer bypass gating to preserve recall.

**Key performance tradeoffs:**
- XLM-RoBERTa-large over smaller models: Slower inference, but significantly better on German compound words and code-switching
- CRF over softmax: ~15% slower training, but eliminates invalid tag sequences
- Post-processing over end-to-end: Adds pipeline complexity, but allows rapid iteration on normalization rules without retraining
- Fuzzy dedup at 94%: Aggressive enough to catch variants, conservative enough to preserve legitimate near-duplicates

## Reproducibility

### Dependencies

```
torch==2.1.0
flair==0.13.1
transformers==4.41.2
optuna
pandas==2.2.2
scikit-learn==1.5.0
```

### Training

```bash
python src/preprocess_train_val.py
python src/convert_biojson_to_flair.py
python src/convert_bio_to_bioes.py
python src/convert_space_to_tab.py
python src/train_model.py
```

### Inference

```bash
python src/predict.py
python src/post_process_predictions_recall_tuned.py \
    --input predictions.tsv \
    --output final_submission.tsv
```

## Project Structure

```
ebay-ner/
├── config/
│   ├── training_config.yaml      # Model + trainer + data config
│   ├── inference_config.yaml     # Inference settings
│   └── eval_config.yaml          # Evaluation settings
│
├── src/                          # 25 Python modules
│   ├── preprocess_train_val.py   # Raw TSV → BIO JSON + train/val split
│   ├── preprocess_quiz.py        # Quiz data normalization
│   ├── convert_biojson_to_flair.py
│   ├── convert_bio_to_bioes.py   # BIO → BIOES tagging
│   ├── convert_space_to_tab.py   # Format for Flair ColumnCorpus
│   ├── train_model.py            # Training + Optuna HPO
│   ├── binary_search.py          # Bayesian HPO continuation
│   ├── predict.py                # Inference with thresholding
│   ├── post_process_predictions_recall_tuned.py
│   ├── allowed_by_category.py    # Category-aspect gating rules
│   ├── dataset_utils.py          # PyTorch Dataset for transformers
│   └── model_utils.py            # BERT tagger definition
│
├── data/
│   ├── raw/                      # Original gzipped TSV input
│   ├── processed/                # BIO JSON (train.bio.json, val.bio.json)
│   ├── corpus/                   # Space-separated Flair format
│   ├── corpus_bioes/             # Tab-separated BIOES (final training input)
│   ├── outputs/                  # Trained models + Optuna trials
│   ├── predictions/              # Raw model outputs
│   └── submissions/              # Post-processed final outputs
│
├── scripts/                      # Shell scripts for pipeline orchestration
├── all_tags.txt                  # Complete NER tag inventory (48 tags)
├── optuna_study.db               # Persistent HPO trial storage
└── requirements.txt
```

**Data flow through directories:**
```
data/raw/ → data/processed/ → data/corpus/ → data/corpus_bioes/ → data/outputs/
                                                                        ↓
                                              data/submissions/ ← data/predictions/
```

## Future Work

- Multi-model ensemble with weighted confidence aggregation
- Category-specific fine-tuning for low-frequency aspects
- Knowledge distillation for inference latency reduction
- Streaming inference API with adaptive batching

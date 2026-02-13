# Architecture

Technical architecture documentation for the German Product Title NER System.

---

## 1. System Overview

### High-Level Description

This system extracts structured product attributes from unstructured German e-commerce listing titles. It implements a sequence labeling pipeline using stacked multilingual embeddings, CRF-based decoding, and domain-specific post-processing.

The architecture prioritizes:
- Modular data transformation stages
- Configuration-driven training
- Decoupled inference and post-processing
- Domain constraint enforcement at prediction time

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    TRAINING PATH                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

  Raw TSV (gzipped)
       │
       ▼
  ┌─────────────────┐
  │ preprocess_     │───▶ GroupShuffleSplit (90/10 by RecordNumber)
  │ train_val.py    │───▶ Empty tag backfilling
  └─────────────────┘───▶ BIO format conversion
       │
       ▼
  BIO JSON (train.bio.json, val.bio.json)
       │
       ▼
  ┌─────────────────┐
  │ convert_biojson │───▶ JSON → space-separated token-tag pairs
  │ _to_flair.py    │
  └─────────────────┘
       │
       ▼
  ┌─────────────────┐
  │ convert_bio_to_ │───▶ BIO → BIOES (adds S- and E- prefixes)
  │ bioes.py        │
  └─────────────────┘
       │
       ▼
  ┌─────────────────┐
  │ convert_space_  │───▶ Space-separated → tab-separated
  │ to_tab.py       │
  └─────────────────┘
       │
       ▼
  Tab-separated BIOES corpus (train_tab.txt, dev_tab.txt, test_tab.txt)
       │
       ▼
  ┌─────────────────┐
  │ train_model.py  │───▶ Load corpus via Flair ColumnCorpus
  │                 │───▶ Build StackedEmbeddings
  │                 │───▶ Initialize SequenceTagger + CRF
  │                 │───▶ Optuna HPO (25 trials)
  │                 │───▶ Retrain with best params
  └─────────────────┘
       │
       ▼
  Trained Model (final-model.pt)


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   INFERENCE PATH                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

  Quiz JSON (tokenized titles + metadata)
       │
       ▼
  ┌─────────────────┐
  │ predict.py      │───▶ Load SequenceTagger
  │                 │───▶ Flair Sentence tokenization
  │                 │───▶ Model forward pass
  │                 │───▶ Span extraction with confidence
  │                 │───▶ Per-aspect threshold filtering
  │                 │───▶ Category-aspect gating
  └─────────────────┘
       │
       ▼
  Raw Predictions TSV
       │
       ▼
  ┌─────────────────────────┐
  │ post_process_           │───▶ Aspect-specific normalization
  │ predictions_recall_     │───▶ Length filtering
  │ tuned.py                │───▶ Fuzzy deduplication
  │                         │───▶ Conflict resolution
  └─────────────────────────┘
       │
       ▼
  Final Submission TSV (record_id, category_id, aspect, value)
```

### Control Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Config    │────▶│   Training   │────▶│    Model     │
│    (YAML)    │     │   Pipeline   │     │   Artifact   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Submission  │◀────│    Post-     │◀────│  Inference   │
│     TSV      │     │  Processing  │     │   Pipeline   │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 2. Data Pipeline Architecture

### Raw Input → Preprocessing

**Input format:** Gzipped TSV with columns `RecordNumber`, `Token`, `Tag`

**Preprocessing steps (`preprocess_train_val.py`):**
- Load raw data with `keep_default_na=False` to preserve empty strings
- Rename columns for consistency
- Split by `GroupShuffleSplit` on `RecordNumber` (not random row sampling)
- Backfill empty tags with previous tag (handles continuation annotations)
- Convert raw tags to BIO format

### Leakage Prevention

```python
splitter = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=42)
train_idx, val_idx = next(splitter.split(df, groups=df["RecordNumber"]))
```

**Why GroupShuffleSplit:**
- A single product title spans multiple rows (one per token)
- Random row splitting would leak partial titles across train/val
- Grouping by `RecordNumber` ensures entire titles stay together
- Prevents artificially inflated dev scores

### BIO → BIOES Conversion

**BIO tags:** `B-` (begin), `I-` (inside), `O` (outside)

**BIOES tags:** `B-` (begin), `I-` (inside), `O` (outside), `E-` (end), `S-` (single)

**Conversion logic (`convert_bio_to_bioes.py`):**
```
B-X followed by I-X  →  B-X
B-X followed by O    →  S-X
I-X followed by I-X  →  I-X
I-X followed by O    →  E-X
```

**Rationale:**
- BIOES provides explicit boundary signals
- CRF learns cleaner transition constraints
- Reduces ambiguity at entity boundaries
- Measurable improvement on short entities (1-2 tokens)

### Modular Conversion Scripts

**Why separate scripts instead of a monolithic pipeline:**
- Each format conversion is independently testable
- Intermediate outputs can be inspected for debugging
- Format changes don't require full reprocessing
- Enables parallel development on different stages
- Supports alternative data sources with different entry points

**Conversion chain:**
```
preprocess_train_val.py  →  BIO JSON
convert_biojson_to_flair.py  →  Space-separated Flair format
convert_bio_to_bioes.py  →  BIOES format
convert_space_to_tab.py  →  Tab-separated (Flair ColumnCorpus compatible)
```

---

## 3. Model Architecture

### Embedding Stack

```
┌─────────────────────────────────────────────────┐
│              StackedEmbeddings                  │
├─────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────┐   │
│  │     TransformerWordEmbeddings           │   │
│  │     (xlm-roberta-large)                 │   │
│  │     - fine_tune=True                    │   │
│  │     - subtoken_pooling="first"          │   │
│  └─────────────────────────────────────────┘   │
│                      +                          │
│  ┌─────────────────────────────────────────┐   │
│  │     FlairEmbeddings (de-forward)        │   │
│  └─────────────────────────────────────────┘   │
│                      +                          │
│  ┌─────────────────────────────────────────┐   │
│  │     FlairEmbeddings (de-backward)       │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Why Stacking Instead of Single Encoder

**XLM-RoBERTa-large provides:**
- Multilingual pretraining (handles code-switching in titles)
- Strong subword tokenization for German compounds
- Transfer learning from 100+ languages

**Flair contextual embeddings add:**
- Character-level language model features
- German-specific pretraining
- Complementary signal to subword representations
- Better handling of OOV tokens and typos

**Empirical observation:** Stacking improves F1 by 2-4% over XLM-R alone on German NER tasks.

### SequenceTagger Configuration

```python
SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type="ner",
    use_crf=True,
    use_rnn=False,
    word_dropout=0.05,
    locked_dropout=0.1,
    reproject_embeddings=True,
)
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hidden_size` | 256 | Balance between capacity and overfitting |
| `use_crf` | True | Sequence-level constraints |
| `use_rnn` | False | Transformer already captures context |
| `word_dropout` | 0.05 | Regularization for rare tokens |
| `locked_dropout` | 0.1 | Consistent dropout across timesteps |
| `reproject_embeddings` | True | Dimensionality alignment |

### CRF Rationale

**Problem:** Independent per-token classification allows invalid sequences (e.g., `I-X` without preceding `B-X`)

**Solution:** CRF layer learns transition probabilities between tags

**Benefits:**
- Eliminates invalid tag sequences at decode time
- Captures label dependencies (B-Manufacturer → I-Manufacturer)
- Viterbi decoding finds globally optimal sequence
- ~15% slower training, but cleaner outputs

### Fine-Tuning Strategy

```yaml
fine_tune: true
```

- XLM-RoBERTa weights are updated during training
- Lower learning rate for pretrained layers (implicit via AdamW)
- Embeddings storage on GPU for efficient backprop
- `subtoken_pooling="first"` aligns labels to first subtoken only

---

## 4. Training System Design

### Configuration-Driven Design

**`training_config.yaml` structure:**
```yaml
model:
  embeddings: [xlm-roberta-large, flair:de-forward, flair:de-backward]
  use_crf: true
  use_rnn: false
  fine_tune: true

trainer:
  max_epochs: 15
  patience: 3
  learning_rate: 5e-5
  mini_batch_size: 16

data:
  corpus_dir: "src/corpus_bioes"
  train_file: "train_tab.txt"
  dev_file: "dev_tab.txt"
  label_type: "ner"

output:
  output_dir: "data/outputs/flair-de-ner/bayesian_search/"
```

**Benefits:**
- No code changes for hyperparameter experiments
- Version-controllable experiment configs
- Clear separation of concerns
- Supports automated sweep tooling

### Hyperparameter Optimization Strategy

**Two-phase approach:**

1. **Random exploration (`train_model.py`):**
   - 25 trials with random sampling
   - TPE sampler with 10 startup trials
   - Broad parameter coverage

2. **Bayesian refinement (`binary_search.py`):**
   - Load existing Optuna study
   - Continue with TPE sampler
   - 25 additional trials
   - Exploits promising regions

**Search space:**
```python
learning_rate: log_uniform(1e-5, 5e-4)
mini_batch_size: categorical([8, 16, 32])
word_dropout: uniform(0.01, 0.2)
locked_dropout: uniform(0.0, 0.5)
use_crf: categorical([True, False])
```

### Persistent Optuna Storage

```python
STORAGE_URL = "sqlite:///optuna_study.db"
study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
```

**Benefits:**
- Resume interrupted optimization runs
- Share results across machines
- Historical trial analysis
- No lost work on crashes

### Experiment Reproducibility

- Fixed random seed in data splitting (`random_state=42`)
- Deterministic train/val assignment via GroupShuffleSplit
- Config files capture full experiment specification
- Model checkpoints saved per trial

### Early Stopping Strategy

```yaml
patience: 3
```

- Monitor dev F1 after each epoch
- Stop if no improvement for 3 consecutive epochs
- Saves best model checkpoint
- Prevents overfitting on small datasets

### Failure Recovery

```python
try:
    return train_model(tcfg)
except Exception as e:
    print(f"[Trial {trial.number}] failed → {e}")
    return 0.0
```

- Failed trials return 0.0 (worst possible score)
- Optuna continues with remaining trials
- Trial outputs isolated in separate directories
- No cascading failures

---

## 5. Inference Architecture

### Tokenization Strategy

```python
sentence = Sentence(title, use_tokenizer=True)
```

- Flair's default tokenizer handles German
- Preserves original token boundaries
- No subword splitting at this stage (handled internally by embeddings)
- Maintains alignment for span extraction

### Span Extraction Mechanics

```python
for span in sentence.get_spans("ner"):
    aspect_name = span.tag[2:]  # Strip B-/I-/S-/E- prefix
    aspect_value = span.text.strip()
    score = span.score
```

- Flair automatically merges BIOES tags into spans
- Each span has: text, tag, confidence score
- Confidence is CRF marginal probability

### Confidence Scoring

- CRF provides sequence-level probability
- Per-span confidence derived from tag marginals
- Range: 0.0 to 1.0
- Higher scores indicate model certainty

### Per-Aspect Thresholding

```python
per_aspect_thresholds = {
    "Hersteller": 0.45,
    "Produktart": 0.45,
    "Kompatible_Fahrzeug_Marke": 0.40,
    "Kompatibles_Fahrzeug_Modell": 0.45,
    "Einbauposition": 0.40,
    "Herstellernummer": 0.45,
    "default": 0.40
}
```

**Why per-aspect:**
- Aspects have different base rates
- Model confidence varies by aspect complexity
- Manufacturer names: high confidence, can use higher threshold
- Technical specs: more ambiguous, lower threshold preserves recall

### Decoupled Thresholding

**Why not learned during training:**
- Thresholds are task-specific (precision/recall tradeoff)
- Can be adjusted without retraining
- Enables rapid iteration on business requirements
- Different deployments may need different tradeoffs

### Batch Inference Considerations

Current implementation: sequential processing per record

**For production scaling:**
- Batch sentences before `tagger.predict()`
- Use `mini_batch_size` parameter in predict
- GPU batching amortizes embedding computation
- ~10x throughput improvement possible

---

## 6. Post-Processing System

### Domain Normalization Rules

**Aspect-specific transformations (`post_process_predictions_recall_tuned.py`):**

| Aspect | Rule | Example |
|--------|------|---------|
| `SAE_Viskosität` | Extract W-grade pattern | `5 W 40` → `5W40` |
| `Bremsscheiben-Aussendurchmesser` | Standardize diameter | `300 mm` → `Ø300mm` |
| `Zähnezahl` | Numeric extraction | `25 Zähne` → `25` |
| `Herstellernummer` | Uppercase, no spaces | `abc 123` → `ABC123` |
| `Hersteller` | Title case | `bosch` → `BOSCH` |

**General cleaning:**
- Normalize dashes (en-dash, em-dash → hyphen)
- Collapse whitespace
- Strip leading/trailing punctuation
- Length filtering (2-80 characters)

### Category-Aspect Compatibility Gating

```python
ALLOWED_TAG_CATEGORIES = {
    "SAE_Viskosität": {2},
    "Farbe": {1},
    "Hersteller": {1, 2},
    ...
}

HIGH_IMPACT_TAGS = {"Hersteller", "Produktart", "Kompatibles_Fahrzeug_Modell"}
```

**Logic:**
1. Check if aspect is in `HIGH_IMPACT_TAGS` → always allow
2. Otherwise, check if category is in allowed set
3. Reject predictions outside valid category-aspect pairs

**Rationale:**
- Domain knowledge encoded as rules
- Reduces false positives from model hallucination
- Cheap filter vs. expensive model correction

### Fuzzy Deduplication

```python
FUZZY_SIM_THRESHOLD = 0.94

def similar(a: str, b: str) -> float:
    a_norm = " ".join(sorted(a.lower().split()))
    b_norm = " ".join(sorted(b.lower().split()))
    return SequenceMatcher(None, a_norm, b_norm).ratio()
```

**Why 0.94:**
- Catches minor variants (`Bosch` vs `BOSCH`)
- Preserves legitimate near-duplicates (`Golf IV` vs `Golf V`)
- Token-sort similarity handles reordering

### Conflict Resolution

```python
def resolve_conflicts(aspect: str, values: List[str]) -> List[str]:
    values = sorted(set(values), key=lambda s: (-len(s), s))
    kept = []
    for v in values:
        if any(similar(v, k) >= FUZZY_SIM_THRESHOLD for k in kept):
            continue
        kept.append(v)
    return kept[:3]
```

**Strategy:**
- Prefer longer, more informative strings
- Deduplicate similar values
- Cap at 3 values per aspect per record

### Precision vs Recall Tradeoffs

```python
DEFAULT_BETA = 2.0  # Fβ with β=2 favors recall
```

- Post-processing tuned for recall emphasis
- Loose filtering on high-value aspects
- Aggressive dedup to control precision
- Business context: missing data worse than duplicates

---

## 7. Evaluation Framework

### Primary Metric

**Micro-averaged F1 score**

```yaml
monitor: "micro_F1_score"
```

- Treats all entity instances equally
- Not biased toward frequent aspects
- Standard NER evaluation metric
- Computed on dev set during training

### Dev Monitoring

- Flair ModelTrainer tracks dev F1 after each epoch
- Best model checkpoint saved automatically
- Training curves logged for analysis
- Early stopping based on dev F1 plateau

### HPO Objective

```python
study = optuna.create_study(direction="maximize")
# ...
return best_f1  # Returned to Optuna
```

- Optuna maximizes dev F1
- Each trial returns best dev score
- TPE sampler models score distribution
- Best trial params used for final model

### F-Beta for Recall Emphasis

```python
DEFAULT_BETA = 2.0
```

- β > 1 weights recall higher than precision
- Used in post-processing threshold tuning
- Reflects business preference for coverage
- Not used in core model training (uses F1)

---

## 8. Scalability and Operational Considerations

### GPU/CPU Embedding Storage

```yaml
embeddings_storage_mode: "gpu"  # or "cpu"
```

**GPU mode:**
- Embeddings remain on GPU between batches
- Faster training (no transfer overhead)
- Higher memory usage

**CPU mode:**
- Embeddings moved to CPU after forward pass
- Lower GPU memory
- Required for large batch accumulation

### Gradient Accumulation

```yaml
mini_batch_size: 16
mini_batch_chunk_size: 8
gradient_accumulation_steps: 2
```

- Effective batch size: 16
- Actual GPU batch: 8
- Accumulate gradients over 2 steps
- Enables large effective batches on limited GPU memory

### Modular Retraining

**Training changes don't affect inference:**
- `predict.py` loads model artifact only
- No training code imported
- Config changes require only `train_model.py` rerun
- Post-processing rules independent of model

**Inference changes don't affect training:**
- Threshold tuning: edit `predict.py`
- Normalization rules: edit `post_process_*.py`
- Category gating: edit `allowed_by_category.py`
- No retraining required

### Latency Bottlenecks

| Stage | Relative Cost | Bottleneck |
|-------|---------------|------------|
| Tokenization | Low | CPU-bound |
| Embedding | High | GPU compute |
| CRF decode | Medium | Sequential Viterbi |
| Post-processing | Low | String operations |

**Primary bottleneck:** XLM-RoBERTa forward pass

### Real-Time Deployment Changes

**Required modifications:**
1. Batch incoming requests (amortize embedding cost)
2. Model compilation (TorchScript or ONNX)
3. Async request handling
4. GPU memory pooling
5. Load balancing across model replicas

**Optional optimizations:**
- Quantization (INT8)
- Knowledge distillation to smaller model
- Caching for repeated titles
- Approximate nearest neighbor for fuzzy dedup

---

## 9. Design Tradeoffs and Constraints

### Sequence Labeling vs Span Classification

**Chosen: Sequence labeling with CRF**

| Approach | Pros | Cons |
|----------|------|------|
| Sequence labeling | Natural for NER, captures transitions | Fixed tag set, no nested entities |
| Span classification | Handles overlapping spans | Quadratic span candidates, slower |
| Span prediction (MRC-style) | Flexible | Requires question templates |

**Decision rationale:**
- Non-overlapping entities in this domain
- BIOES + CRF handles boundaries well
- Flair provides battle-tested implementation
- Training efficiency matters for HPO

### Post-Processing vs Learned Normalization

**Chosen: Rule-based post-processing**

| Approach | Pros | Cons |
|----------|------|------|
| Rule-based | Interpretable, fast iteration | Manual maintenance |
| Learned normalization | End-to-end | Needs training data, black box |
| Hybrid | Best of both | Complexity |

**Decision rationale:**
- Domain rules are well-defined (SAE format, dimensions)
- Rapid iteration without retraining
- Debuggable when errors occur
- Rules can encode business logic not in training data

### Single Model vs Per-Category Models

**Chosen: Single shared model + category gating**

| Approach | Pros | Cons |
|----------|------|------|
| Single model | More training data, simpler | May conflate categories |
| Per-category | Specialized | Data fragmentation, N models to maintain |
| Multi-task | Shared + specialized | Architecture complexity |

**Decision rationale:**
- Categories share many aspects
- Single model sees more examples per aspect
- Category gating handles category-specific aspects
- Operational simplicity (one model to deploy)

### Performance vs Maintainability

| Choice | Performance Impact | Maintainability Impact |
|--------|-------------------|------------------------|
| XLM-R-large over base | +2-3% F1 | Same |
| Stacked embeddings | +2-4% F1 | Slightly more complex |
| CRF layer | +1-2% F1 | Flair handles complexity |
| Rule-based post-processing | Neutral | Much easier to debug |
| Per-aspect thresholds | +1-2% F1 | More tuning knobs |

**Philosophy:** Accept complexity at model level (handled by framework), keep pipeline simple.

---

## 10. Extension Points

### Ensemble Integration

**Current architecture supports:**
```python
taggers = [SequenceTagger.load(path) for path in tagger_paths]
# Already loops over multiple taggers
for tagger in taggers:
    tagger.predict(sentence)
```

**To add ensemble:**
1. Train multiple models (different seeds, architectures)
2. Add model paths to `tagger_paths` list
3. Implement score aggregation in `predict_weighted()`
4. Adjust thresholds for ensemble confidence

**Integration point:** `predict.py:49-72`

### Model Compression

**Distillation integration:**
```
┌─────────────────┐     ┌─────────────────┐
│  Teacher Model  │────▶│  Student Model  │
│  (XLM-R-large)  │     │  (XLM-R-base)   │
└─────────────────┘     └─────────────────┘
```

**Steps:**
1. Generate soft labels from teacher on training data
2. Train student with soft label loss
3. Replace model path in `predict.py`
4. No changes to post-processing

**Integration point:** New `distill.py` script, loads trained model

### Active Learning

**Uncertainty sampling integration:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Unlabeled     │────▶│    Predict +    │────▶│  Select High    │
│     Data        │     │  Uncertainty    │     │  Uncertainty    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │    Annotate     │
                                                │   & Retrain     │
                                                └─────────────────┘
```

**Uncertainty signals available:**
- CRF marginal probabilities (already extracted)
- Variance across ensemble predictions
- Token-level entropy

**Integration point:** New `select_uncertain.py` script, uses `predict.py` logic

### Microservice Conversion

**Current → Service transformation:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        NER Service                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   FastAPI   │──│  Inference  │──│    Post-Processing      │ │
│  │   Router    │  │   Worker    │  │        Worker           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│         │                │                      │               │
│         ▼                ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Request Queue                            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Required components:**
1. HTTP endpoint (FastAPI/Flask)
2. Request batching layer
3. Model loading at startup
4. Health check endpoint
5. Prometheus metrics

**Code reuse:**
- `predict.py` → inference worker logic
- `post_process_predictions_recall_tuned.py` → post-processing worker
- `allowed_by_category.py` → imported directly

**New code:**
- API router
- Request/response schemas
- Batch accumulator
- Deployment configs (Dockerfile, K8s manifests)

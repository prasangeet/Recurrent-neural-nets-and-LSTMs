# Problem 2: Character-Level Name Generation Using RNN Variants

## Objective

Design and compare sequence models for character-level name generation using recurrent neural architectures — specifically a Vanilla RNN, a Bidirectional LSTM, and an Attention-based RNN — trained on a dataset of Indian names.

---

## Project Structure

```
.
├── main.py                  # Entry point
├── constants.py             # Configurations and hyperparameters
├── TrainingNames.txt        # Dataset (1000 Indian names)
├── vocab.json               # Generated vocabulary (char2idx / idx2char)
│
├── src/
│   ├── preprocess.py        # Data preprocessing
│   ├── dataset.py           # PyTorch Dataset class
│   ├── data_module.py       # DataLoader setup
│   ├── generator.py         # Name generation logic
│   └── logger.py            # Logging utilities
│
├── model_classes/
│   ├── vanilla_rnn.py       # Vanilla RNN implementation
│   ├── blstm.py             # Bidirectional LSTM implementation
│   ├── attention_rnn.py     # Attention RNN implementation
│   └── model_factory.py     # Model instantiation factory
│
├── train/
│   └── train.py             # Training loop
│
├── test/
│   └── evaluator.py         # Novelty and Diversity evaluation
│
├── logs/                    # Training logs (auto-generated)
└── models/                  # Saved model checkpoints (auto-generated)
```

---

## How to Run

### Step 1 — Create a virtual environment

```bash
python -m venv .venv
```

### Step 2 — Activate the virtual environment

On **Linux / macOS:**
```bash
source .venv/bin/activate
```

On **Windows:**
```bash
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the pipeline

```bash
python -m main
```

This will train all three models sequentially, evaluate them, and save logs and checkpoints to `logs/` and `models/` respectively.

---

## Dataset (Task 0)

1000 Indian names were generated using an LLM and stored in `TrainingNames.txt`. The dataset is split into 800 training samples and 200 validation samples. The vocabulary size is 25 unique characters.

---

## Models (Task 1)

### 1. Vanilla RNN

A simple recurrent model that processes sequences one character at a time, maintaining a hidden state across timesteps.

- **Architecture:** Single-layer RNN with a linear output head
- **Limitation:** Prone to vanishing gradients; struggles to capture long-range character dependencies

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | CrossEntropyLoss (padding ignored) |
| Epochs | 30 |

---

### 2. Bidirectional LSTM (BiLSTM)

Processes sequences in both forward and backward directions, concatenating the hidden states from each pass.

- **Architecture:** Bidirectional LSTM with a linear output head
- **Strength:** Captures context from both sides of each character position
- **Limitation:** With a small dataset (~1000 names), the model memorizes training examples rather than learning generalisable patterns

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | CrossEntropyLoss (padding ignored) |
| Epochs | 30 |

---

### 3. Attention RNN

Extends the Vanilla RNN by adding an attention mechanism over previous hidden states, allowing the model to selectively weight past context when predicting the next character.

- **Architecture:** RNN encoder + attention layer + linear output head
- **Strength:** Better at capturing non-local dependencies compared to Vanilla RNN
- **Limitation:** Attention benefits are limited by the small dataset size

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | CrossEntropyLoss (padding ignored) |
| Epochs | 30 |

---

## Evaluation (Task 2)

For each model, 1000 names were generated and scored on:

- **Novelty Rate** — percentage of generated names not present in the training set
- **Diversity** — number of unique generated names divided by total generated

| Model | Final Train Loss | Final Val Loss | Novelty Rate | Diversity |
|---|---|---|---|---|
| Vanilla RNN | 0.8111 | 1.0500 | 1.0000 | 0.9850 |
| BiLSTM | 0.0002 | 0.0027 | 0.9980 | 0.9470 |
| Attention RNN | 0.8446 | 1.0788 | 1.0000 | 0.9480 |

---

## Qualitative Analysis (Task 3)

### Vanilla RNN

**Sample outputs:**
```
arungankanitetetek
deteprangangandete
viteteshangankanes
anitetenanititetit
pranetiteteteekanu
```

**Observations:**
- High novelty and diversity, but outputs are mostly non-words
- Strong repetitive patterns (`tetete`, `gangang`) indicate the model is stuck in local loops
- Fails to produce phonetically plausible Indian names

---

### BiLSTM

**Sample outputs:**
```
annushaloolallomoo
ananultttteshnantu
nikkkrakkkkaralkkk
nittttttilmmita
ralal
```

**Observations:**
- Extremely low training and validation loss — the model has memorised the training set
- Generated outputs show character run collapse (`kkkk`, `tttt`, `llll`)
- Classic overfitting: near-zero loss does not translate to quality generation
- Variable output lengths suggest the model has not learned a consistent stopping condition

---

### Attention RNN

**Sample outputs:**
```
andhiteteteeetete
ranititeteshitetes
sankshitetu
saniteghitetu
praniteteteghitete
```

**Observations:**
- Marginally better phonetic structure than Vanilla RNN (prefixes like `san-`, `pran-`, `ran-` are common in Indian names)
- Still dominated by the `tete` repetition pattern
- Attention mechanism provides a marginal improvement but cannot overcome the dataset size constraint

---

## Key Insights

**Low loss ≠ good generation.** The BiLSTM achieves near-zero loss through memorisation, yet produces the worst outputs of the three models.

**High diversity ≠ meaningful names.** All models generate unique strings, but uniqueness alone does not imply phonetic or linguistic validity.

**Dataset size is the primary bottleneck.** With only ~1000 names, all three models lack sufficient signal to learn generalisable character-level structure.

**Character-level repetition is a known failure mode.** Without regularisation or constrained decoding, RNNs collapse into short repetitive loops, especially on small corpora.

---

## Limitations

- Small training dataset (~1000 names)
- No regularisation (dropout, weight decay)
- Greedy decoding only — no beam search or temperature sampling
- Character-level only — no phonetic, syllabic, or morphological priors

---

## Possible Improvements

- Expand the dataset significantly (5000+ names)
- Add dropout and weight decay for regularisation
- Use temperature scaling or nucleus sampling during generation
- Move to syllable-level or subword tokenisation to reduce repetition
- Experiment with Transformer-based architectures (e.g., GPT-style character LM)
- Introduce an n-gram language model prior to penalise implausible character sequences

---

## Deliverables

- Source code for all three model implementations
- Generated name samples (logged to `logs/`)
- Evaluation scripts (`test/evaluator.py`)
- This report / README

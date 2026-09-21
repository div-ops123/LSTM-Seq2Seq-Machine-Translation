
## Destination

**Understand the encoder-decoder architecture as the precursor to attention, implement a vanilla LSTM seq2seq model, and empirically expose the fixed-context bottleneck.**

That's it.

---

# Precise sequence

### MILESTONE 1 — Architecture  ---- DONE

**Study:**

1. Why sequence-to-sequence problems need encoder-decoder
2. Encoder
3. Context/final hidden state
4. Decoder
5. Autoregressive generation
6. `<SOS>` / `<EOS>`
7. Teacher forcing
8. Training vs inference

**Do not study:**

* attention
* Transformer encoder
* Transformer decoder
* cross-attention
* masking in the Transformer sense
* beam search

**Deliverable:**

One encoder-decoder diagram + your own written explanation.

**Time: 45 min**

---

### MILESTONE 2 — Real dataset + preprocessing

Use **machine translation**.

Your dataset should contain:

```text
source sentence → target sentence
```

You perform only the preprocessing necessary for your experiment:

```text
clean sentence pairs
→ tokenize
→ vocabulary
→ numerical IDs
→ <SOS>/<EOS>
→ padding
→ train/validation/test
```

Do not use somebody else's engineered features.

For NLP, tokenization and sequence length are themselves part of the learning objective. The updated note explicitly treats tokenization, embeddings and sequence length as foundational. 

**Deliverable:**

Dataset specification:

```text
Dataset:
Task:
Source:
Target:
# examples:
source length distribution:
target length distribution:
tokenization:
vocabulary:
```

**Time: 45 min**

---

### MILESTONE 3 — Implement vanilla seq2seq

Build:

```text
Source
  ↓
Embedding
  ↓
LSTM Encoder
  ↓
final hidden/cell state
  ↓
LSTM Decoder
  ↓
Linear
  ↓
Target vocabulary
```

With:

* teacher forcing during training
* autoregressive decoding during inference

**No attention.**

**Deliverable:**

Working PyTorch encoder-decoder.

**Time: 1.5 hr**

---

### MILESTONE 4 — Baseline

Train your model once with a fixed configuration.

Record:

* validation loss
* translation metric
* training time
* a few translations
* parameter count

This is your **reference point**, not an experiment.

**Deliverable:**

Baseline table.

**Time: 30–45 min**

---

# MILESTONE 5 — Main controlled experiment

## Experiment: Sequence length

This is the **one experiment I definitely want you to keep.**

Take your real translation dataset and evaluate performance by source-length bucket.

For example, after inspecting the actual distribution:

```text
short
medium
long
very long
```

or numerical bins such as:

```text
1–10
11–20
21–30
31–40
...
```

Don't decide the bins until you inspect the data.

### Control everything else.

Same:

* model
* hidden size
* optimizer
* learning rate
* training procedure

Variable:

> **source sequence length**

Measure:

* translation quality
* inference behavior
* possibly latency

### Question:

> **Does the fixed-size encoder representation become a bottleneck as the source sequence becomes longer?**

**Time: 1.5 hr**

---

# MILESTONE 6 — One supporting experiment

Now change:

> **encoder hidden/context size**

For example:

```text
64 → 128 → 256
```

while keeping the sequence-length setup fixed.

Question:

> **Can increasing the capacity of the fixed representation compensate for the degradation seen on longer sequences?**

Measure:

* translation quality
* parameter count
* training time

This gives you the engineering tradeoff:

```text
larger representation
        ↓
potentially more information capacity
        ↓
but
        ↓
more parameters / compute
```

**Time: 1 hr**

---

# MILESTONE 7 — Diagnosis

**No coding.**

Look at the two experiments.

Answer:

1. What happened as source length increased?
2. Did increasing hidden size help?
3. What information does the decoder receive?
4. What information does it *not* directly receive?
5. What architectural requirement does your evidence reveal?

Your final answer should establish the problem:

> The decoder is forced to rely on a fixed representation of the entire source rather than selectively accessing different source positions.

That leads directly to **attention**.

**Time: 30 min**

---

# MILESTONE 8 — Final evidence

Create one concise experiment document.

### Structure

```text
# Vanilla Encoder-Decoder: Information Bottleneck

## 1. Problem
Machine translation

## 2. Architecture
LSTM encoder-decoder without attention

## 3. Hypothesis
...

## 4. Experiment 1
Source length

## 5. Experiment 2
Encoder representation size

## 6. Results
Tables/plots

## 7. Failure observed
...

## 8. Engineering interpretation
...

## 9. Requirement for the next architecture
...
```

**Time: 30–45 min**

---
# conBERT: Multi-Output Essay Scoring with BERT

This project implements a multi-output regression model using BERT to automatically score essays on grammar, coherence, and content relevance.

## 🔍 Overview

- Uses `bert-base-uncased` for contextual encoding
- Predicts 3 scores per essay: grammar, coherence, relevance
- Includes semantic similarity scoring using `all-MiniLM-L6-v2`
- Trained using Hugging Face `Trainer` API

## 📁 Dataset

- Input: `essay_text`, `grammar_score`, `coherence_score`, `content_relevance_score`
- Preprocessing includes:
  - Dropping missing/short/duplicate entries
  - Assigning topics via keyword matching
  - Injecting semantic similarity features

## 🧠 Model Architecture

```python
class BertMultiRegressor(nn.Module):
    def __init__(self, model_ckpt, n_outputs=3):
        ...

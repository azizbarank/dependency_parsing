# Dependency Parsing with RoBERTa - Report

## Overview

This project implements a neural dependency parser using RoBERTa and biaffine attention, predicting syntactic head-dependent relationships for English sentences from the Universal Dependencies corpus.

## Implementation

### Data

The English Universal Dependencies corpus (en_ewt) was used:
- Training: 12,543 sentences
- Validation: 2,002 sentences

Tokenization alignment handles RoBERTa's subword tokenization by assigning head labels only to first subword tokens, with subsequent subwords receiving -100 (ignored during training).

### Components

**1. Data Loading & Tokenization**

The dataset is loaded from Hugging Face and tokenized using RoBERTa's tokenizer with `add_prefix_space=True`. Key challenges addressed:
- CoNLL-U uses 1-based indexing (0 = root)
- RoBERTa prepends `<s>` token, requiring position remapping
- Head positions mapped from word indices to token indices

**2. Model Architecture**

The `DependencyParser` consists of:
- RoBERTa-base encoder (768-dimensional hidden states)
- Head MLP: Linear(768, 500) + ReLU
- Dependent MLP: Linear(768, 500) + ReLU
- Biaffine attention parameters: U1 (500×500 matrix), u2 (500-dim vector)

Score computation: `Score(i,j) = H_head[i]^T · U1 · H_dep[j] + H_head[i]^T · u2`

Implemented using `torch.einsum` for efficient vectorized computation without loops.

**3. Training**

- Loss: CrossEntropyLoss with `ignore_index=-100`
- Optimizer: AdamW (learning rate 2e-5)
- Epochs: 10
- Batch size: 32

**4. Evaluation**

Two evaluation metrics implemented:
- **Head Tagging Accuracy**: Proportion of tokens assigned the correct head (ignoring -100 tokens)
- **UAS (Unlabeled Attachment Score)**: Accuracy after applying Chu-Liu-Edmonds MST algorithm to ensure valid tree structure

## Results

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.0533 |
| Head Tagging Accuracy | 93.44% |
| UAS (MST) | 93.48% |

### Training Progression

| Epoch | Loss | Head Accuracy |
|-------|------|---------------|
| 1 | 0.9782 | 89.77% |
| 2 | 0.3347 | 92.08% |
| 3 | 0.2393 | 92.52% |
| 4 | 0.1852 | 92.94% |
| 5 | 0.1441 | 93.18% |
| 6 | 0.1174 | 93.12% |
| 7 | 0.0944 | 93.43% |
| 8 | 0.0785 | 93.44% |
| 9 | 0.0649 | 93.20% |
| 10 | 0.0533 | 93.43% |

### Learning Curves

![Learning Curves](learning_curves.png)

## Discussion

### Why Biaffine Attention?

Words can act as either heads or dependents. Using two separate MLPs lets the model learn different representations for each role, which makes sense since being a head is different from being a dependent.

### Head Accuracy vs UAS

Both metrics are nearly identical (93.44% vs 93.48%). This means the model already predicts valid trees most of the time. The MST algorithm just fixes the rare cases where the predictions would create invalid structures like cycles.

### Why This Works Well

The model benefits from RoBERTa already knowing a lot about language from its pretraining. Instead of starting from scratch, we're building on top of that knowledge, which is why we get good results.

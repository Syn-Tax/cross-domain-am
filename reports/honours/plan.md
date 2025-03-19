# Introduction

# Background

## Argumentation Theory

- Inference Anchoring Theory & argument theory
  - what is an argument? *
  - utterances/discourse *
  - speech act theory / rhetorical structure theory (possibly) *
  - ADUs *
  - Propositions - including reconstruction*
  - Propositional relations *
    - Inference *
    - Conflict *
    - Rephrase *
  - Locutions *
  - Transitions *
  - Illocutionary connections *
  - argument schemes *
- Argument Interchange Format
  - what? why? *
  - argument-markup language *
  - types of node *
    - I-nodes, L-nodes *
    - S-nodes *

## Machine Learning

- NLP
  - text pre-trained transformers *
  - LLMs *
  - audio PTs *
- multimodal techniques *
  - usage in medicine *
  - usage in LLMs *
  - usage in argumentation *

## Argument Mining

- Argument Mining (& MAM)
  - what? *
  - manual stages *
    - segmentation *
    - argument/non-arument classification *
    - simple structure *
    - refined structure *
  - automatic stages *
    - segmentation *
    - argument/non-argument classification *
    - causal properties *
    - relational properties *
  - computational tasks *
  - 3 and 4 class problem *
  - sampling unrelated nodes *
  - extending AM cross-domain *
  - recent results in MAM (inc. MAMKit) *

# Datasets

## Preprocessing

- preprocessing pipeline:
  - processing AIF data
  - forced alignment (both attempts)
    - issues surrounding forced alignment on incorrect transcripts
    - how CTC works
- How relations are sampled

## QT30

- discussion of data origins
- dataset statistics
- Alignment characteristics

## Moral Maze

- discussion of data origins
- dataset statistics for each episode
- Alignment characteristics

# Models

- Early sequence fusion
  - how sequences are fused for both modalities
- Late sequence fusion
- Multimodal fusion
  - only late fusion techniques
  - concat, product, cross-attention

# Results

## Experimental Setup

- training parameters:
  - loss (cross-entropy)
  - optimizer (adamw)
- evaluation
  - metrics (macro-F1, precision, recall)
  - data splits
  - evaluating different NO sampling strategies

## In-Dataset

- traditionally sampled results
  - comparison between MM fusion and sequence fusion
- comparison between traditional sampling and oversampling
- comparison between NO sampling methods

## Cross-Dataset

- traditionally sampled results (likely only using better performing MM and sequence fusion techniques)
- comparison between traditional sampling and oversampling
- comparison between NO sampling methods in training

# Limitations

# Conclusions

# Future Work

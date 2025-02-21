# Introduction

# Previous Work

# Datasets

- What is AIF

## Preprocessing

- preprocessing pipeline:
  - processing AIF data
  - forced alignment (both attempts)
    - issues surrounding forced alignment on incorrect transcripts
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

## In-Dataset

- traditionally sampled results
  - comparison between MM fusion and sequence fusion
- comparison between traditional sampling and oversampling
- comparison between NO sampling methods

## Cross-Dataset

- traditionally sampled results (likely only using better performing MM and sequence fusion techniques)
- comparison between traditional sampling and oversampling
- comparison between NO sampling methods in training

# Conclusions

# Future Work
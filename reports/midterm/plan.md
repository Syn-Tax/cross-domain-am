# Introduction

- What is AM
- What is ARI (how it differs from ARC)
- Why multimodal models could be useful & intro to early and late fusion
- Intro on transformers (both text and audio)

# Datasets

- What is AIF

## Preprocessing

- preprocessing pipeline:
  - processing AIF data
  - forced alignment (both attempts)
    - issues surrounding forced alignment on incorrect transcripts
- How relations are sampled

## QT30

- dataset statistics
- Alignment characteristics

## Moral Maze

- dataset statistics for each episode
- Alignment characteristics

# Models

- early concatenation text-only

# Preliminary Results

- for each model type implemented (no CD analysis just yet probably)

# Conclusion

- just conclude m8



# Introduction

Argument Mining (AM) is the automatic identification and extraction of argumentative structures from natural language discourse [@lawrenceArgumentMiningSurvey2020]. In order to achieve this, the discourse can first be segmented into Elementary Discourse Units (EDUs), atomic units of the discourse. The EDUs can then be classified into argument and non-argument, the argumentative EDUs can be termed Argumentative Discourse Units (ADUs). Following this, clausal properties can be determined for each ADU (e.g for an ADU X, is X evidence? or is X a premise?). Finally, the relations between ADUs can be determined (e.g. does one ADU support or attack another?). It is with this final step that this research is concerned.

The task of identifying the relational properties between already identified ADUs is known as Argument Relation Identification (ARI). ARI concerns itself both with identifying whether a relation exists between to ADUs, and also classifying the type of relation, (typically either support or attack). This typically becomes a 3-class classification problem, concerning no relation, support and attack [@gemechuARIESGeneralBenchmark2024].

There are, however, varying opinions on whether having only two relational classes is enough, Inference Anchoring Theory (IAT) presents three different classes [@budzynskaArgumentMiningDialogue2014;@budzynskaModelProcessingIllocutionary2014]. In IAT, the 'support' class is split into 'inference' or RA and 'rephrase' or MA. The 'attack' relation is also termed 'conflict' or CA. Using the IAT relation classes produces a 4-class classification problem.

Transformer models [@vaswaniAttentionAllYou2017] have significantly improved performance in many tasks across natural language processing, including AM [@ruiz-dolzTransformerBasedModelsAutomatic2021;@wuKnowCompDialAM2024Finetuning2024]. Subsequent task-agnostic pretraining approaches have allowed these models to be fine-tuned on a wide variety of tasks. As the RoBERTa model [@liuRoBERTaRobustlyOptimized2019a] has been shown to perform well in the ARI task [@ruiz-dolzTransformerBasedModelsAutomatic2021] it is used for this research to allow comparison with previous work.

Multimodal models have also proven useful in audio-based natural language tasks, where transcripts of the spoken word are aso available. Generally it has appeared that combining both audio and textual features has seen improved performance over unimodal techniques [@manciniMultimodalArgumentMining2022;@manciniMAMKitComprehensiveMultimodal2024].

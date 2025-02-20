---
title: A Cross-Domain Evaluation of Multimodal Argument Relation Identification
author: Oscar Morris

bibliography: [../Cross-Domain AM.bib]
documentclass: article
numbersections: true
codeBlockCaptions: true
classoption: twocolumn
fontsize: 12pt
cref: false
header-includes: |
    \usepackage[margin=2cm]{geometry}
    \usepackage{graphicx}
    \graphicspath{ {./assets/} }

    \usepackage{fvextra}
    \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}

    \usepackage{longtable}

    \setlength{\columnsep}{1.2cm}
---

# Introduction

# Previous Work

# Datasets

All datasets used in this project are available as corpora on AIFdb^[https://corpora.aifdb.org/]. Using consistently annotated Argument Interchange Format (AIF) data allows many different datasets to be used and tested. The AIF Format [@chesnevarArgumentInterchangeFormat2006] allows the annotation of argument data across all AM tasks, providing a platform for many different kinds of research.

Throughout the project two primary corpora have been considered: QT30 [@hautli-janiszQT30CorpusArgument2022], a corpus consisting of 30 AIF annotated Question Time episodes, and a corpus of 9 AIF annotated Moral Maze episodes available on AIFdb.

## Preprocessing

### Argument Data

In order to use AIF data efficiently for ARI, it is useful to perform some preprocessing. This process produces a graph, where each node contains a locution, its related proposition, and the proposition's AIF identifier. This identifier corresponds to the audio data, allowing it to be easily loaded when required. Each edge in this graph corresponds to a relation between the propositions, one of RA (inference), MA (rephrase) or CA (conflict).

\begin{figure}[h!]
\centering
\includegraphics[width=8cm]{argument-map}
\caption{Example sub-graph. \label{fig:arg-map}}
\end{figure}

Figure \ref{fig:arg-map} shows an example sub-graph from the larger argument graph. Each node is truncated for brevity and only shows the node's ID, and the proposition. This sub-graph is taken from the Moral Maze episode on the 75th Anniversary of D-Day.

An example of the JSON structure used to store the argument data is shown in Listing \ref{lst:arg-map}.

```json {#lst:arg-map .numberLines caption="Example JSON object corresponding to a Node."}
{
    "id": 433407,
    "locution": "Matthew Taylor : she answered questions about norms and structures by talking about beliefs and campaigns and I think beliefs are different to norms and I think campaigns are different to social structures",
    "proposition": "Nancy Sherman answered questions about norms and structures by talking about beliefs and campaigns and beliefs are different to norms and campaigns are different to social structures",
    "relations": [
        {
            "type": "CA",
            "to_node_id": 433393
        },
        {
            "type": "RA",
            "to_node_id": 433416
        }
    ],
}
```

An array of these JSON objects can then be used to create the node pairs required for the training and evaluation of the model.

### Audio Data

The audio data first had to be downsampled from 44.1kHz to the 16kHz which is best accepted by the Wav2Vec2 transformer [@baevskiWav2vec20Framework2020] among many others. This can easily be achieved using FFmpeg^[https://ffmpeg.org/]. In the case of QT30, first audio had to be extracted from the video, and collapsed into a mono track before it could be downsampled, this was also easily achieved with FFmpeg.

Next, start and end times for each locution in the argument graph need to be found, to allow the audio to be split per-locution (and therefore per node). This can be achieved using PyTorch's forced alignment api^[https://pytorch.org/audio/].

Initially this was achieved by aligning each word in the transcript of the episode, producing start and end times for each word. A search can then be performed through this data to find the required locution. While this technique initially produced promising results, it was not robust enough to allow for errors in the transcripts or the crosstalk common in debates.

To solve this problem, the PyTorch forced alignment api is able to take wildcard tokens as input, therefore, each locution can be searched for individually. To achieve this, the partial transcript used as input to the forced aligner took the following form: `* {locution} *`.

Using this system allows the forced aligner to work well through crosstalk (since each locution's alignments are searched for independently of all others), and qualitatively seems to be more resilient to errors. Error resilience is helped since errors are less common in the locution texts as opposed to the transcripts. Using this system also allowed for confidence scores to be collected for analysis. In this section general analysis across all corpora is performed, with corpus specific analysis in the relevant section.

\begin{figure}[h!]
\centering
\includegraphics[width=8cm]{complete-confidence}
\caption{Confidence distribution across all corpora. \label{fig:complete-confidence}}
\end{figure}

Figure \ref{fig:complete-confidence} shows the distribution of confidence scores across both the QT30 corpus and all Moral Maze corpora. This distribution shows that the system can relatively confidently align the majority of locutions, with only approx. 8% of locutions with a confidence score less than $0.50$.

In order to further analyse the performance of this system, locutions were selected at random and qualitatively analysed. Throughout this process, all locutions appeared correct, however, it was very challenging to accurately determine the accuracy of the system on locutions with confidence scores $<0.2$. This shows that this method of aligning locutions with their corresponding audio is accurate.

### Pair Creation

Finally, a set of node pairs and their relations can be generated in order to train a neural network. For related nodes this can be done trivially in that for each relation, the corresponding pair of nodes can be added to the set. When sampling unrelated nodes, however, things are more complex.

For this project, Short Context Sampling (SCS) is used as presented in [@ruiz-dolzLookingUnseenEffective2025]. Given the episodic structure of both QT30 and the Moral Maze corpora, a short context can be defined as the episode. This also allows the model to learn in a more realistic environment. Since the vast majority of node pairs have no relation, a number equal to that of Inference relations are generated.

## QT30

The QT30 argument corpus [@hautli-janiszQT30CorpusArgument2022] contains transcripts and argument annotations for 30 episodes of the BBC's Question Time, a series of televised topical debates across the United Kingdom. All episodes aired in 2020 and 2021. The corpus is split into 30 subcorpora, each spanning a single episode. This allows analysis of each episode individually, or combined as a single corpus.

\begin{table}[h!]
\centering
\caption{Disribution of propositional relations in QT30. \label{tbl:qt-rel}}
\begin{tabular}{|l|ll|}
\hline
Relation Type & Count & Proportion (\%) \\ \hline
Inference     & 5,761       & 51.4\%       \\
Conflict      & 947         & 8.5\%      \\
Rephrase      & 4,496       & 40.1\%     \\ \hline
Total         & 11,204      & 100\%      \\ \hline
\end{tabular}
\end{table}

Table \ref{tbl:qt-rel} shows the distribution of each type of relation across QT30. Inference and Rephrase relations make up a total of $91.5\%$ of the dataset, with Conflict relations being significantly less common, only making up $8.5\%$ of the dataset. It is obvious that this is an unbalanced dataset, which will have to be considered during training.

\begin{table}[h!]
\centering
\caption{Mean confidence scores ($\mu$) and standard deviation of confidence scores ($\sigma$) across each QT30 subcorpus. \label{tbl:qt-confidence}}
\begin{tabular}{|l|ll|}
\hline
Corpus Name     & $\mu$ & $\sigma$ \\ \hline
28May2020       & 0.76            & 0.16               \\
4June2020       & 0.72            & 0.17               \\
18June2020      & 0.76            & 0.16               \\
30July2020      & 0.75            & 0.16               \\
2September2020  & 0.78            & 0.15               \\
22October2020   & 0.76            & 0.16               \\
5November2020   & 0.77            & 0.17               \\
19November2020  & 0.74            & 0.19               \\
10December2020  & 0.77            & 0.16               \\
14January2021   & 0.74            & 0.17               \\
28January2021   & 0.70            & 0.17               \\
18February2021  & 0.75            & 0.16               \\
4March2021      & 0.76            & 0.16               \\
18March2021     & 0.75            & 0.17               \\
15April2021     & 0.75            & 0.15               \\
29April2021     & 0.70            & 0.19               \\
20May2021       & 0.76            & 0.17               \\
27May2021       & 0.79            & 0.15               \\
10June2021      & 0.75            & 0.17               \\
24June2021      & 0.74            & 0.17               \\
8July2021       & 0.72            & 0.17               \\
\textbf{22July2021}      & \textbf{0.44}   & \textbf{0.28}      \\
5August2021     & 0.76            & 0.17               \\
19August2021    & 0.77            & 0.17               \\
2September2021  & 0.78            & 0.16               \\
16September2021 & 0.77            & 0.16               \\
\textbf{30September2021} & \textbf{0.24}   & \textbf{0.08}      \\
14October2021   & 0.75            & 0.19               \\
28October2021   & 0.75            & 0.17               \\
11November2021  & 0.78            & 0.16               \\ \hline
\textbf{QT30}   & 0.75            & 0.17               \\ \hline
\end{tabular}
\end{table}

Table \ref{tbl:qt-confidence} shows the mean and standard deviation of the confidence scores across each of the QT30 subcorpora. The lowest two mean scores are shown in bold. Manually analysing samples in these episodes indicates a high error rate in the alignment of locutions. Because of this high error rate, it was decided to exclude these episodes from the corpus used for training. The excluded episodes are: 22July2021 and 30September2021. The rest of the episodes from QT30 will form its multimodal subcorpus (QT30-MM).

\begin{table}[t]
\centering
\caption{Disribution of propositional relations in QT30-MM. \label{tbl:qt-mm-rel}}
\begin{tabular}{|l|ll|}
\hline
Relation Type & Count & Proportion (\%) \\ \hline
Inference     & 5,740       & 51\%       \\
Conflict      & 937         & 8.4\%      \\
Rephrase      & 4,479       & 40.6\%     \\ \hline
Total         & 11,156      & 100\%      \\ \hline
\end{tabular}
\end{table}

Similarly to the complete QT30 corpus, Inference and Rephrase make up the vast majority of the QT30-MM dataset, with the proportion of Conflict relations decreasing to 8.4%.

\begin{figure}[h!]
\centering
\includegraphics[width=8cm]{qt30-mm-confidence}
\caption{Confidence distribution across QT30-MM. \label{fig:qt30-mm-confidence}}
\end{figure}

As can be seen in Figure \ref{fig:qt30-mm-confidence} the distribution of confidence scores closely matches that shown in Figure \ref{fig:complete-confidence}. This still indicates that the audio alignments are calculated with high accuracy.

\begin{table}[h!]
\centering
\caption{Distribution of propositional relations after sampling non-related nodes. \label{tbl:qt-mm-rel-no}}
\begin{tabular}{|l|ll|}
\hline
Relation Type & Count  & Proportion (\%) \\ \hline
None          & 5,470  & 34\%            \\
Inference     & 5,470  & 34\%            \\
Conflict      & 937    & 6\%             \\
Rephrase      & 4,479  & 27\%            \\ \hline
Total         & 16,896 & 100\%           \\ \hline
\end{tabular}
\end{table}

\begin{table*}[t]
\centering
\caption{Distribution of propositional relations across the Moral Maze corpus. \label{tbl:moral-rel-no}}
\begin{tabular}{|l|llll|l|}
\hline
Relation Type & None         & Inference    & Conflict  & Rephrase  & Total \\ \hline
B             & 132 (45\%)   & 132 (45\%)   & 24 (8\%)  & 3 (1\%)   & 291   \\
E             & 151 (41\%)   & 151 (41\%)   & 39 (11\%) & 25 (7\%)  & 366   \\
M             & 255 (43\%)   & 255 (43\%)   & 29 (5\%)  & 58 (10\%) & 597   \\
P             & 236 (42\%)   & 236 (42\%)   & 40 (7\%)  & 45 (8\%)  & 557   \\
S             & 181 (42\%)   & 181 (42\%)   & 63 (14\%) & 10 (2\%)  & 435   \\
G             & 301 (41\%)   & 301 (41\%)   & 46 (6\%)  & 93 (13\%) & 741   \\
D             & 72 (40\%)    & 72 (40\%)    & 7 (4\%)   & 28 (16\%) & 179   \\
H             & 207 (43\%)   & 207 (43\%)   & 23 (5\%)  & 43 (9\%)  & 480   \\ \hline
Total         & 1,535 (42\%) & 1,535 (42\%) & 271 (7\%) & 305 (8\%) & 3,646 \\ \hline
\end{tabular}
\end{table*}

\pagebreak

## Moral Maze

Similar to Question Time, the BBC's Moral Maze is a series of radio broadcast debates, with each episode focusing on a certain topic. Seven different Moral Maze episodes have been AIF annotated and made available on AIFdb. It is therefore these seven episodes, released from 2012 to 2019, which this project considers. Each episode focuses on a very different domain which allows for a robust, cross-domain analysis of any models trained on another corpus (e.g. QT30). The Moral Maze corpus contains data from nine different episodes: Banking (B), Empire (E), Money (M), Problem (P), Syria (S), Green Belt (G), D-Day (D) and Hypocrisy (H). Each episode consists of a debate focusing on a different topic, and hence has a different distribution of classes.

Table \ref{tbl:moral-rel-no} shows the distribution of propositional relations across the Moral Maze corpus, after non-related pairs have been sampled. Comparing the corpus to QT30, a significantly lower proportion of the corpus is made up of Rephrase relations. It is possible that the differing formats of the debates has an impact here.

\begin{table}[h]
\centering
\caption{Mean confidence scores ($\mu$) and Standard Deviation of confidence scores ($\sigma$) across Moral Maze subcorpora. \label{tbl:moral-confidence}}
\begin{tabular}{|l|ll|}
\hline
Subcorpus & $\mu$ & $\sigma$ \\ \hline
B         & 0.79 & 0.14   \\
E         & 0.76 & 0.15   \\
M         & 0.78 & 0.14   \\
P         & 0.80 & 0.15   \\
S         & 0.74 & 0.16   \\
G         & 0.80 & 0.14   \\
D         & \textbf{0.50} & \textbf{0.34}   \\
H         & 0.74 & 0.16   \\ \hline
\end{tabular}
\end{table}

In Table \ref{tbl:moral-confidence} the mean and standard deviation of audio alignemnt confidence scores are compared across subcorpora. Generally the results match what is expected and are similar to those in QT30, the system does achieve unusually low scores when considering the D-Day subcorpus, the reason for this is unclear, however, manually analysing both random and low-confidence samples indicates they are generally correct and so the subcorpus can be used for the cross-domain evaluation.

\begin{figure}[h!]
\centering
\includegraphics[width=8cm]{moral-confidence}
\caption{Confidence distribution across all Moral Maze subcorpora. \label{fig:moral-confidence}}
\end{figure}

Figure \ref{fig:moral-confidence} shows the distribution of audio alignment confidence scores across all Moral Maze episodes. This follows the expected pattern as shown in Figures \ref{fig:complete-confidence} and \ref{fig:qt30-mm-confidence}. The secondary peak around a confidence of $0.10$ is caused by the D-Day subcorpus.

# Models {#sec:models}

There are many different approaches when considering multimodal models. Generally they can be split into two categories: early fusion and late fusion. Early fusion models initially combine the features from each modality together, before being used as input to a single transformer model. Late fusion models use multiple transformers, one specialised in each modality. After being fed through each transformer, the hidden vectors are combined before being fed into the model's head. Initially, only late fusion models have been considered in this research.

This late fusion model uses RoBERTa-base [@liuRoBERTaRobustlyOptimized2019a] as its text transformer. To process the audio data, the model uses the Wav2Vec2-base audio transformer [@baevskiWav2vec20Framework2020], having been pre-trained on 960 hours of dialogue^[https://huggingface.co/facebook/wav2vec2-base-960h]. The base models are used currently in order to increase the speed at which experiments can be conducted, however, they are used with a view to transitioning to their large variants towards the end of the project.

\begin{figure}[h!]
\centering
\includegraphics[width=8cm]{model-diag}
\caption{Concatenation model with late fusion. \label{fig:model-diag}}
\end{figure}

Figure \ref{fig:model-diag} povides a visualisation of the primary model used at this point in the project. Similar models have been shown to achieve good multimodal performance while also being relatively simple to implement [@manciniMultimodalArgumentMining2022;@manciniMAMKitComprehensiveMultimodal2024]. It is for these reasons that this model has been implemented using the Huggingface^[https://huggingface.co/] and PyTorch^[https://pytorch.org/] python modules.

Some preliminary experiments have also been conducted using text-only models in an attempt to replicate results produced in [@gemechuARIESGeneralBenchmark2024]. For this the RoBERTa-base model is used and each sentence is concatenated before tokenisation, delimited by RoBERTa's special purpose token `</s>`.

# Results

# Conclusions

# Future Work

# References

---
bibliography: [../../Cross-Domain AM.bib]
---

# Datasets {#sec:datasets}

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

Figure \ref{fig:arg-map} shows an example sub-graph from the larger argument graph. Each node is truncated for brevity and only shows the node's ID, and the proposition. This sub-graph is taken from the Moral Maze episode on the 75th Anniversary of D-Day. The major downside of processing the data in this way is simply that much of the nuance encoded within AIF is lost. This is primarily the inference and conflict structures (e.g. linked arguments, undercutting conflict etc.) but also the transitions and illocutionary connections. In the context of this project this is not an issue but is worth remembering when examining the data.

An example of the JSON structure used to store the argument data is shown in Listing \ref{lst:arg-map}. It is also worth understanding the link between the locution and the proposition, of which those in Listing \ref{lst:arg-map} are good examples. The locution is exactly what is said, and the speaker is given (in this case Matthew Taylor), whereas the proposition can be thought of as adding a bit more context from the surrounding dialogue. This primarily includes pronoun resolution as seen in the first word "she" in the locution vs. "Nancy Sherman" in the proposition.

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

Next, start and end times for each locution in the argument graph need to be found, to allow the audio to be split per-locution (and therefore per node). This can be achieved using Connectionist Temporal Classification (CTC) [@gravesConnectionistTemporalClassification2006] as exposed by PyTorch's forced alignment api^[https://pytorch.org/audio/].

In order to understand the abilities of a CTC-based forced alignment system it is of course useful to understand how the algorithms work. Simply, CTC provides the probability distribution across a set of tokens, for each timestep (known as a frame). For a forced alignment task, these tokens are typically each letter of the alphabet and a blank token. The blank token is used for frames which cannot be classified as any other token (e.g. silence).

\begin{figure}[h!]
\centering
\includegraphics[width=8cm]{framewise-probs}
\caption{Example framewise probabilities. \label{fig:framewise-probs}}
\end{figure}

Figure \ref{fig:framewise-probs} (taken from the PyTorch tutorial on the subject^[https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html]) shows an example of the framewise probability distribution across each token, token 0 here is the blank token. This distribution provides the probability (or confidence) of any particular token appearing in any given frame. Taking simply the most probable tokens provides something that looks like the following: `- i - - h h a - - - d -` where `-` is the blank token. That sequence describes the words 'I had', so it can be seen that the duplicates need to be removed, along with the blank tokens. This is the process that would be undertaken for Automated Speech Recognition.

When looking at forced alignment however, the process is a bit different since we already have a transcript. For forced alignment the goal is to find the most probably route through the framewise probability matrix matching the transcript. To do this a so-called trellis matrix can be generated. This represents the probabilities of remaining at the same token in the transcript, or moving on to the next one in each frame.

We are then looking for the path across the most likely transitions, $k_{(t+1, j+1)}$, where $j$ is the current location in the transcript, and $t$ is the current timeframe. The trellis can then be defined as in Equation @eq:trellis.

$$ k_{t+1, j+1} = \max\left(k_{(t,j)p(t+1, c_{j+1})}, k_{(t,j+1)}p(t+1, repeat)\right) $$ {#eq:trellis}

Where $k$ is the trellis matrix and $p(t, c_j)$ is the probability of any token $c_j$ appearing in frame $t$, effectively referencing the framewise probability matrix, and $repeat$ represents the blank token.

Once the trellis matrix is generated, an example of which is shown in Figure \ref{fig:trellis-matrix}^[https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html] where the yellow high-probability path is visually obvious, it can be traversed using a backtracking algorithm, starting from the last token in the transcript and following either $(c_j \rightarrow c_j)$ or $(c_j \rightarrow c_j+1)$ transitions, based on their probability, until reaching the beginning of the transcript.

\begin{figure}[h!]
\centering
\includegraphics[width=8cm]{trellis-matrix}
\caption{Example trellis matrix where yellow shows a high probability. \label{fig:trellis-matrix}}
\end{figure}

At this point, we have start and end frame-numbers for each token, and based on the probabilities the model has traversed, a 'confidence' score can be calculated based on the mean of the probabilities traversed, this group of three values is known as a span (in this case a token span). The token spans can be generated by using the PyTorch forced alignment API, allowing the algorithm to be easily implemented and used. Finally, the token spans can be combined into word spans based on word boundaries in the transcript. This provides start and end frame numbers for each word, along with a confidence score. The same process can be conducted later to find a locution-level span with a confidence score. The frame numbers can then be easily converted back into times in the waveform and then split into the required segment.

Initially the forced alignment of the argumentative discourse was achieved by aligning each word in the complete transcript of the episode, producing start and end times for each word. A search can then be performed through this data to find the required locution. While this technique initially produced promising results, it was not robust enough to allow for errors in the transcripts or the crosstalk common in debates. This problem can be seen in the fact that the trellis matrix only allows for a single token to appear in each timeframe, which is trivially not applicable to the real world in an argumentative context where a lot of crosstalk (multiple speakers talking over each other) exists.

To solve this problem, the PyTorch forced alignment API is able to take wildcard tokens as input, therefore, each locution can be searched for individually. To achieve this, the partial transcript used as input to the forced aligner took the following form: `* {locution} *`.

Using this system allows the forced aligner to work well through crosstalk (since each locution's alignments are searched for independently of all others), and qualitatively seems to be more resilient to errors. Error resilience is helped since errors are less common in the locution texts as opposed to the transcripts. Using this system also allowed for confidence scores to be collected for analysis. In this section general analysis across all corpora is performed, with corpus specific analysis in the relevant section.

\begin{figure}[h!]
\centering
\includegraphics[width=8cm]{complete-confidence}
\caption{Confidence distribution across all corpora. \label{fig:complete-confidence}}
\end{figure}

Figure \ref{fig:complete-confidence} shows the distribution of confidence scores across both the QT30 corpus and all Moral Maze corpora. This distribution shows that the system can relatively confidently align the majority of locutions, with only approx. 8% of locutions with a confidence score less than $0.50$.

In order to further analyse the performance of this system, locutions were selected at random and qualitatively analysed. Throughout this process, all locutions appeared correct, however, it was very challenging to accurately determine the accuracy of the system on locutions with confidence scores $<0.2$. This shows that this method of aligning locutions with their corresponding audio is accurate for the purposes of this project, as long as the confidence scores are taken into account.

The distribution of lengths for each audio clip was also analysed in order to ensure the models are being provided with enough data. The primary statistics are shown in Table \ref{tbl:audio-complete}. This data shows that the majority of locutions are shorter than 8 seconds (approximately 120,000 samples at the sampling rate of 16kHz). In total, across both corpora, there is over 24 hours of argumentative audio, out of over 36 hours of total audio processed. The relevant data for the specific corpora are detailed in the relevant section.

\begin{table}[h]
\centering
\caption{Audio data for locutions across all corpora.\label{tbl:audio-complete}}
\begin{tabular}{|l|ll|}
\hline
Quantity        & Length (s) & No. of Samples \\ \hline
Mean            & 3.9        & 62,000         \\
75th Percentile  & 5.1        & 81,000         \\
90th Percentile & 7.7        & 120,000        \\
Maximum         & 31         & 490,000        \\ \hline
\end{tabular}
\end{table}

### Pair Creation {#sec:pair-creation}

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

Table \ref{tbl:audio-data-qt} shows the statistics for the audio part of QT30-MM. Generally the values are very similar to those shown in Table \ref{tbl:audio-complete}. The QT30-MM dataset contains almost 20 hours of argumentative audio taken from approximately 29.5 hours of total audio.

\begin{table}[h]
\centering
\caption{Audio data for locutions across QT30-MM.\label{tbl:audio-data-qt}}
\begin{tabular}{|l|ll|}
\hline
Quantity        & Length (s) & No. of Samples \\ \hline
Mean            & 3.8        & 60,000         \\
75th Percentile  & 5.0        & 79,000         \\
90th Percentile & 7.5        & 120,000        \\
Maximum         & 30         & 470,000        \\ \hline
\end{tabular}
\end{table}

## Moral Maze

Similar to Question Time, the BBC's Moral Maze is a series of radio broadcast debates, with each episode focusing on a certain topic. Nine different Moral Maze episodes have been AIF annotated and made available on AIFdb. It is therefore these nine episodes, released from 2012 to 2019, which this project considers. Each episode focuses on a very different domain which allows for a robust, cross-domain analysis of any models trained on another corpus (e.g. QT30). The Moral Maze corpus contains data from nine different episodes: Banking (B), Empire (E), Money (M), Problem (P), Syria (S), Green Belt (G), D-Day (D), Hypocrisy (H) and Welfare (W). Each episode consists of a debate focusing on a different topic, and hence has a different distribution of classes.

Table \ref{tbl:moral-rel-no} shows the distribution of propositional relations across the Moral Maze corpus, after non-related pairs have been sampled. Comparing the corpus to QT30, a significantly lower proportion of the corpus is made up of Rephrase relations. It is possible that the differing formats of the debates has an impact here.

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
H             & 207 (43\%)   & 207 (43\%)   & 23 (5\%)  & 43 (9\%)  & 480   \\
W             & 211 (40\%)   & 211 (40\%)   & 59 (11\%)  & 43 (8\%)  & 524   \\ \hline
Total         & 1,746 (42\%) & 1,746 (42\%) & 330 (8\%) & 348 (8\%) & 4,170 \\ \hline
\end{tabular}
\end{table*}

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
H         & 0.74 & 0.16   \\
W         & 0.75 & 0.15   \\ \hline
\end{tabular}
\end{table}

In Table \ref{tbl:moral-confidence} the mean and standard deviation of audio alignemnt confidence scores are compared across subcorpora. Generally the results match what is expected and are similar to those in QT30, the system does achieve unusually low scores when considering the D-Day subcorpus, the reason for this is unclear, however, manually analysing both random and low-confidence samples indicates they are generally correct and so the subcorpus can be used for the cross-domain evaluation.

\begin{figure}[h!]
\centering
\includegraphics[width=8cm]{moral-confidence}
\caption{Confidence distribution across all Moral Maze subcorpora. \label{fig:moral-confidence}}
\end{figure}

Figure \ref{fig:moral-confidence} shows the distribution of audio alignment confidence scores across all Moral Maze episodes. This follows the expected pattern as shown in Figures \ref{fig:complete-confidence} and \ref{fig:qt30-mm-confidence}. The secondary peak around a confidence of $0.10$ is caused by the D-Day subcorpus.

Table \ref{tbl:audio-data-mm} shows the statistics for the audio part of the combined Moral Maze corpus. Generally the locutions seem to be a bit longer in the Moral Maze when compared to QT30. The Moral Maze combined corpus contains almost 5 hours of argumentative audio taken from approximately 6.5 hours of total audio.

\begin{table}[h]
\centering
\caption{Audio data for locutions across Moral Maze.\label{tbl:audio-data-mm}}
\begin{tabular}{|l|ll|}
\hline
Quantity        & Length (s) & No. of Samples \\ \hline
Mean            & 4.4        & 70,000         \\
75th Percentile & 5.8        & 94,000         \\
90th Percentile & 9.0        & 140,000        \\
Maximum         & 31         & 490,000        \\ \hline
\end{tabular}
\end{table}
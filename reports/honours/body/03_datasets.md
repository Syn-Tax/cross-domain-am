---
bibliography: [../../Cross-Domain AM.bib]
---

# Datasets {#sec:datasets}

All argument data used in this project are available as corpora on AIFdb^[https://corpora.aifdb.org/]. Using consistently annotated Argument Interchange Format (AIF) data allows many different datasets to be used and tested. The AIF Format [@chesnevarArgumentInterchangeFormat2006] allows the annotation of argument data across all AM tasks, providing a platform for many different kinds of research.

Throughout the project two primary corpora have been considered: QT30 [@hautli-janiszQT30CorpusArgument2022], a corpus consisting of 30 AIF annotated Question Time episodes, and a corpus of 9 AIF annotated Moral Maze episodes available on AIFdb [@lawrenceAIFdbCorpora2014].

## Preprocessing

### Argument Data {#sec:arg-data}

In order to use AIF data efficiently for ARI, it is useful to perform some preprocessing. This process produces a graph, where each node contains a locution, its related proposition, and the proposition's AIF identifier. This identifier corresponds to the audio data, allowing it to be easily loaded when required. Each edge in this graph corresponds to a relation between the propositions, one of RA (inference), MA (rephrase) or CA (conflict).

Figure \ref{fig:arg-map} shows an example sub-graph from the larger argument graph. Each node is truncated for brevity and only shows the node's ID, and the proposition. The major downside of processing the data in this way is simply that much of the nuance encoded within AIF is lost. This is primarily the advanced structures (e.g. linked arguments, undercutting conflict etc.) but also the transitions and illocutionary connections. In the context of this project this is not an issue but is worth remembering when examining the data.

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{argument-map}
\caption{Example sub-graph. \label{fig:arg-map}}
\end{figure}

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

### Forced Alignment

#### Design

The audio data first had to be downsampled from 44.1kHz to the 16kHz which is best accepted by the Wav2Vec2 transformer [@baevskiWav2vec20Framework2020] among many others. This can easily be achieved using FFmpeg^[https://ffmpeg.org/]. In the case of QT30, first audio had to be extracted from the video, and collapsed into a mono track before it could be downsampled, this was also easily achieved with FFmpeg.

Next, start and end times for each locution in the argument graph need to be found, to allow the audio to be split per-locution (and therefore per node). This can be achieved using Connectionist Temporal Classification (CTC) [@gravesConnectionistTemporalClassification2006] as exposed by PyTorch's forced alignment api^[https://pytorch.org/audio/]. CTC allows a model to classify the data in a certain timestep into one of several categories (in this case each letter) considering the data in surrounding timesteps. This then provides the probability distribution across a set of tokens, for each timestep (known as a frame). For a forced alignment task, these tokens are typically each letter of the alphabet and a blank token. The blank token is used for frames which cannot be classified as any other token (e.g. silence).

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{framewise-probs}
\caption{Example framewise probabilities. \label{fig:framewise-probs}}
\end{figure}

Figure \ref{fig:framewise-probs} (taken from the PyTorch tutorial on the subject^[https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html]) shows an example of the framewise probability distribution across each token, token 0 here is the blank token. This distribution provides the probability (or confidence) of any particular token appearing in any given frame. Taking simply the most probable tokens provides something that looks like the following: `- i - - h h a - - - d -`, where `-` represents the blank token. That sequence describes the words 'I had', so it can be seen that the duplicates need to be removed, along with the blank tokens. This is the process that would be undertaken for Automated Speech Recognition.

When looking at forced alignment, however, the process is a bit different since we already have a transcript. For forced alignment the goal is to find the most probable route through the framewise probability matrix which matches the transcript. To do this a so-called trellis matrix can be generated. Here it is useful to envision two 'pointers', one of which represents the current frame in the audio and the other represents the current position in the transcript. Then for every transition between frames, consider the probability that the position in the transcript remains the same vs. the probability that it moves forward one character.

We are then looking for the path across the most likely transitions, $k_{(t+1, j+1)}$, where $j$ is the current location in the transcript, and $t$ is the current timeframe. The trellis can then be defined as in Equation \ref{eq:trellis}.

\begin{align}
\label{eq:trellis}
k_{t+1, j+1} = \max \Bigl( & k_{(t,j)}p(t+1, c_{j+1}), \\
& k_{(t,j+1)}p(t+1, repeat) \Bigr) \nonumber
\end{align}

Where $k$ is the trellis matrix and $p(t, c_j)$ is the probability of any token $c_j$ appearing in frame $t$, effectively referencing the framewise probability matrix, and $repeat$ represents the blank token.

Once the trellis matrix is generated, an example of which is shown in Figure \ref{fig:trellis-matrix}^[https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html] where the yellow high-probability path is visually obvious, it can be traversed using a backtracking algorithm, starting from the last token in the transcript and following either $(c_j \rightarrow c_j)$ or $(c_j \rightarrow c_{j-1})$ transitions, based on their probability, until reaching the beginning of the transcript.

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{trellis-matrix}
\caption{Example trellis matrix where yellow shows a high probability. \label{fig:trellis-matrix}}
\end{figure}

At this point, we have start and end frame-numbers for each token, and based on the probabilities the model has traversed, a 'confidence' score can be calculated based on the mean of the probabilities traversed, this group of three values is known as a span (in this case a token span). The token spans can be generated by using the PyTorch forced alignment API, allowing the algorithm to be easily implemented and used. Finally, the token spans can be combined into word spans based on word boundaries in the transcript. This provides start and end frame numbers for each word, along with a confidence score. The same process can be conducted later to find a locution-level span with a confidence score. The frame numbers can then be easily converted back into times in the waveform and then split into the required segment.

#### Implementation

Initially the forced alignment of the argumentative discourse was achieved by aligning each word in the complete transcript of the episode, producing start and end times for each word. A search can then be performed through these data to find the required locution. While this technique initially produced promising results, it was not robust enough to allow for errors in the transcripts or the crosstalk common in debates. This problem can be seen in the fact that the trellis matrix only allows for a single token to appear in each timeframe, which is trivially not applicable to the real world in an argumentative context where a lot of crosstalk (multiple speakers talking over each other) exists.

To solve this problem, the PyTorch forced alignment API is able to take wildcard tokens as input, therefore, each locution can be searched for individually. To achieve this, the partial transcript used as input to the forced aligner took the following form: `* {locution} *`.

Using this system allows the forced aligner to work well through crosstalk (since each locution's alignments are searched for independently of all others), and qualitatively seems to be more resilient to errors. Error resilience is helped since errors are less common in the locution texts as opposed to the transcripts. Using this system also allowed for confidence scores to be collected for analysis. In Section @sec:datasets-audio general analysis across all corpora is performed, with corpus specific analysis in the relevant section.

### Audio Data {#sec:datasets-audio}

Figure \ref{fig:complete-confidence} shows the distribution of confidence scores across both the QT30 corpus and all Moral Maze corpora. This distribution shows that the system can relatively confidently align the majority of locutions, with only approximately 8% of locutions with a confidence score less than $0.50$.

\begin{figure}[h!]
\centering
\includegraphics[width=8cm]{complete-confidence}
\caption{Confidence distribution across all corpora. \label{fig:complete-confidence}}
\end{figure}

In order to further analyse the performance of this system, locutions were selected at random and qualitatively analysed. Throughout this process, all locutions appeared correct, however, it was very challenging to manually determine the accuracy of the system on locutions with confidence scores $<0.2$ due to high amounts of crosstalk or other acoustic artefacts. This shows that this method of aligning locutions with their corresponding audio is accurate for the purposes of this project, as long as the confidence scores are taken into account. However, these confidence scores should not be mistaken for a 'probability of being correct'.

The distribution of lengths for each audio clip was also analysed in order to ensure the models are being provided with enough data. The primary statistics are shown in Table \ref{tbl:audio-complete}. These data shows that the majority of locutions are shorter than 8 seconds (approximately 120,000 samples at the sampling rate of 16kHz). In total, across both corpora, there is over 24 hours of argumentative audio, out of over 36 hours of total audio processed. The relevant data for the specific corpora are detailed in the relevant section.

\begin{table}[h]
\centering
\caption{Audio data for locutions across all corpora.\label{tbl:audio-complete}}
\begin{tabular}{|l|ll|}
\hline
Quantity        & Length (s) & No. of Samples \\ \hline
Mean            & 3.9        & 62,000         \\
75th Percentile & 5.1        & 81,000         \\
90th Percentile & 7.7        & 120,000        \\
Maximum         & 31         & 490,000        \\ \hline
\end{tabular}
\end{table}

Next, a comparison can be made between the sizes of the datasets presented here (QT30-MM and Moral Maze) when compared to previous work in ARI and other AM subtasks. Table \ref{tbl:mm-comparison} compares the M-Arg [@mestreMArgMultimodalArgument2021], VivesDebate-Speech [@ruiz-dolzVivesDebateSpeechCorpusSpoken2023], UKDebates [@lippiArgumentMiningSpeech2016], MM-USED [@manciniMultimodalArgumentMining2022] and MM-USED-fallacy [@manciniMultimodalFallacyClassification2024a]. As far as could be ascertained, QT30-MM is by far the biggest multimodal argument mining dataset created to date with 20 hours of argumentative audio and although the Moral Maze corpus only contains 5 hours of argumentative audio, it is the only cross-domain corpus.

\begin{table}[h]
\centering
\caption{Comparison between different multimodal argument mining datasets. Lengths are in hours. *dataset lengths were not reported by initial authors so are derived from downloaded audio data. \label{tbl:mm-comparison}}
\begin{tabular}{|l|l|l|}
\hline
Dataset            & Task              & Length \\ \hline
M-Arg              & ARI               & 7              \\
VivesDebate-Speech & ARI, ASD          & 12             \\
UKDebates          & Claim Detection   & 2              \\
MM-USED            & ASD, ACC          & 3.6*           \\
MM-USED-fallacy    & Fallacy Detection & 4.2*           \\ \hline
QT30-MM            & ARI               & 20             \\
Moral Maze         & ARI               & 5              \\ \hline
\end{tabular}
\end{table}

### ADU Pair Creation {#sec:pair-creation}

Finally, a set of node pairs and their relations can be generated in order to train a neural network. For related nodes this can be done trivially in that for each relation, the corresponding pair of nodes can be added to the set. When sampling unrelated nodes, however, things are more complex.

It has also been shown that how unrelated node pairs are sampled is very relevant to the model's performance [@ruiz-dolzLookingUnseenEffective2025]. For this reason, it is also useful to provide a comparison between the different methods in a multimodal context. Since a short context is defined as being within an episode, the sampling strategies are only relevant for QT30, all Moral Maze episodes are simply undersampled. The following methods are compared:

- **Undersampling (US)** is the simplest method. The set of all possible pairs is created and then randomly undersampled to the number of inference/support relations.
- **Long Context Sampling (LCS)** samples unrelated nodes such that each node comes from a different episode with the result that they are 'far apart' in the discourse, this often takes the form of a different topic and such the task is slightly easier than the other methods. This list can then be randomly undersampled to the number of inference/support relations.
- **Short Context Sampling (SCS)** samples unrelated such that each node comes from the same episode so they are 'close together' in the discourse meaning that they often involve the same topic with the result that the task is slightly harder than other methods. This set is then randomly undersampled to the number of inference/support relations.

## QT30

The QT30 argument corpus [@hautli-janiszQT30CorpusArgument2022] contains transcripts and argument annotations for 30 episodes of the BBC's Question Time, a series of televised topical debates across the United Kingdom. All episodes aired in 2020 and 2021. The corpus is split into 30 subcorpora, each spanning a single episode. This creates a large corpus with almost 20k locutions. What follows is an analysis of the corpus and how audio data were added.

Table \ref{tbl:qt-rel} shows the distribution of each type of relation across QT30. Inference and Rephrase relations make up a total of $91.5\%$ of the dataset, with Conflict relations being significantly less common, only making up $8.5\%$ of the dataset. It is obvious that this is an unbalanced dataset, which will have to be considered during training.

\begin{table}[h]
\centering
\caption{Distribution of propositional relations in QT30. \label{tbl:qt-rel}}
\begin{tabular}{|l|ll|}
\hline
Relation Type & Count & Proportion (\%) \\ \hline
Inference     & 5,761       & 51.4\%    \\
Conflict      & 947         & 8.5\%     \\
Rephrase      & 4,496       & 40.1\%    \\ \hline
Total         & 11,204      & 100\%     \\ \hline
\end{tabular}
\end{table}

Figure \ref{fig:qt30-confidence-box} shows a box plot of the mean confidence values across each episode. This plot shows two outliers corresponding to the 22July2021 episode (with mean confidence 0.44) and the 30September2021 episode (with mean confidence 0.24). Manually analysing samples in these episodes indicates a high error rate in the alignment of locutions. Because of this high error rate, it was decided to exclude these episodes from the corpus used for training. The rest of the episodes from QT30 will form its multimodal subcorpus (QT30-MM) where QT30-MM has a mean confidence score of 0.75 with a standard deviation of 0.17.

\begin{figure}[h]
\centering
\includegraphics[height=8cm]{confidence-box}
\caption{Box plot of episodic confidence values across QT30. \label{fig:qt30-confidence-box}}
\end{figure}

As can be seen in Figure \ref{fig:qt30-mm-confidence} the distribution of confidence scores closely matches that shown in Figure \ref{fig:complete-confidence}. This still indicates that the audio alignments are calculated with high accuracy.

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{qt30-mm-confidence}
\caption{Confidence distribution across QT30-MM. \label{fig:qt30-mm-confidence}}
\end{figure}

Table \ref{tbl:qt-mm-rel-no} shows the distribution of relations after sampling unrelated nodes, this process increases the dataset size to a total of almost 17k samples. When comparing QT30 to QT30-MM the total number of relations only drops from 11,204 in QT30 to 11,156 in QT30-MM (only losing 48 relations), however, the quality of audio alignment does increase.

\begin{table}[h]
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

The audio data contained within the dataset can also be analysed as shown in Table \ref{tbl:audio-data-qt}. Generally the values are very similar to those shown in Table \ref{tbl:audio-complete}. The QT30-MM dataset contains almost 20 hours of argumentative audio taken from approximately 29.5 hours of total audio. This makes the QT30-MM corpus the largest multimodal ARI dataset currently available.

\begin{table}[h]
\centering
\caption{Audio data for locutions across QT30-MM.\label{tbl:audio-data-qt}}
\begin{tabular}{|l|ll|}
\hline
Quantity        & Length (s) & No. of Samples \\ \hline
Mean            & 3.8        & 60,000         \\
75th Percentile & 5.0        & 79,000         \\
90th Percentile & 7.5        & 120,000        \\
Maximum         & 30         & 470,000        \\ \hline
\end{tabular}
\end{table}

## Moral Maze

Similar to Question Time, the BBC's Moral Maze is a series of radio broadcast debates, with each episode focusing on a certain topic. Several episodes of the Moral Maze have been annotated with IAT and AIF and also made available on AIFdb. Of these episodes, eight were chosen from different fields. It is therefore these eight episodes, released from 2012 to 2019, which this project considers. Each episode focuses on a very different domain which allows for a robust, cross-domain analysis of any models trained on another corpus (e.g. QT30). The Moral Maze corpus contains data from eight different episodes: Banking (B), Empire (E), Money (M), Problem (P), Syria (S), Green Belt (G), Hypocrisy (H) and Welfare (W). Each episode consists of a debate focusing on a different topic, and hence has a different distribution of classes.

Table \ref{tbl:moral-rel-no} shows the distribution of propositional relations across the Moral Maze corpus, after unrelated pairs have been sampled. Comparing the corpus to QT30, a significantly lower proportion of the corpus is made up of Rephrase relations. It is possible that the differing formats of the debates has an impact here.

\begin{table*}[t]
\centering
\caption{Distribution of propositional relations across the Moral Maze corpus. \label{tbl:moral-rel-no}}
\begin{tabular}{|l|llll|l|}
\hline
Subcorpus \textbackslash  Relation Type & None         & Inference    & Conflict  & Rephrase  & Total \\ \hline
B             & 132 (45\%)   & 132 (45\%)   & 24 (8\%)  & 3 (1\%)   & 291   \\
E             & 151 (41\%)   & 151 (41\%)   & 39 (11\%) & 25 (7\%)  & 366   \\
M             & 255 (43\%)   & 255 (43\%)   & 29 (5\%)  & 58 (10\%) & 597   \\
P             & 236 (42\%)   & 236 (42\%)   & 40 (7\%)  & 45 (8\%)  & 557   \\
S             & 181 (42\%)   & 181 (42\%)   & 63 (14\%) & 10 (2\%)  & 435   \\
G             & 301 (41\%)   & 301 (41\%)   & 46 (6\%)  & 93 (13\%) & 741   \\
H             & 207 (43\%)   & 207 (43\%)   & 23 (5\%)  & 43 (9\%)  & 480   \\
W             & 211 (40\%)   & 211 (40\%)   & 59 (11\%)  & 43 (8\%)  & 524   \\ \hline
Total         & 1,674 (42\%) & 1,674 (42\%) & 323 (8\%) & 320 (8\%) & 3,991 \\ \hline
\end{tabular}
\end{table*}

Figure \ref{fig:mm-confidence-box} shows a box plot of the mean confidence scores for each subcorpus of Moral Maze. Generally the results match what is expected and are similar to those in QT30 although there are no outliers and therefore no episodes were omitted.

\begin{figure}[h]
\centering
\includegraphics[height=8cm]{mm-confidence-box}
\caption{Per-subcorpus mean of confidence scores for each Moral Maze episode. \label{fig:mm-confidence-box}}
\end{figure}

In Figure \ref{fig:moral-confidence} the distribution of audio alignment confidence scores across all Moral Maze episodes can be seen. This follows the expected pattern as shown in Figures \ref{fig:complete-confidence} and \ref{fig:qt30-mm-confidence}.

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{moral-confidence}
\caption{Confidence distribution across all Moral Maze subcorpora. \label{fig:moral-confidence}}
\end{figure}

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
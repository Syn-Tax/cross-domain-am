---
bibliography: [../../Cross-Domain AM.bib]
---

# Results

## Experimental Setup {#sec:exp-setup}

To provide a comparable set of results, all experiments were run using the same hyperparameters. Each model was trained on a single Nvidia RTX 4070 Super for 15 epochs with a batch size of 32 using a weighted cross-entropy loss and the AdamW optimiser [@loshchilovDecoupledWeightDecay2019a] initialised with a learning rate of $10^{-5}$, a linear learning rate scheduler and 10\% of training used as warm-up steps. The cross-entropy weights were calculated as in Equation @eq:weight, where $\mathbf{c}$ is a vector containing the number of samples in each class, and $\mathbf{w}$ is a vector containing the relevant cross-entropy weight.

$$ w_i = \frac{\max(\mathbf{c})}{c_i} $$ {#eq:weight}

Using these hyperparameters was found to provide a good balance between model performance and training time on the hardware used with the weighted cross-entropy loss.

In order to evaluate the models, the following metrics are reported: macro-averaged F1 score, precision and recall. These are all described for each class in Equations @eq:f1, @eq:precision and @eq:recall where $TP$ is the number of true positives, $FP$ is the number of false positives and $FN$ is the number of false negatives. The arithmetic mean can then be taken for each class to provide a holistic overview of the model's performance.

$$ F1 = \frac{2TP}{2TP + FP + FN} $$ {#eq:f1}

$$ \text{Precision} = \frac{TP}{TP + FP} $$ {#eq:precision}

$$ \text{Recall} = \frac{TP}{TP + FN} $$ {#eq:recall}

Generally the macro-averaged F1 score is the standard to evaluate a multi-class classification problem, including ARI systems [@ruiz-dolzTransformerBasedModelsAutomatic2021;@ruiz-dolzLookingUnseenEffective2025;@manciniMAMKitComprehensiveMultimodal2024], where only a single metric is reported, it will be a macro-averaged F1 score for this reason.

To provide a useful evaluation and simulate a real-world environment, the QT30-MM dataset is split into three splits: train, validation and test. 70% of the data is allocated for training, 10% for validation and the remaining 20% for testing. The model is evaluated on the validation split after every training epoch, the best performing model, based on macro-F1, is then chosen to be tested on the testing split, the metrics are then calculated and reported in the following sections.

After training, each model is then evaluated on the complete dataset for each Moral Maze episode (Banking, Empire, Money, Problem, Syria, Green Belt, Hypocrisy and Welfare) and each metric calculated to provide an overview of the cross-domain performance of the model.

In order to evaluate the different methods to sample unrelated arguments as described in Section @sec:pair-creation. Models are trained on SCS, US and LCS, the validation dataset is sampled identically to the training set, and tested, both in-domain and cross-domain on SCS. This is used because of its description as a more realistic problem [@ruiz-dolzLookingUnseenEffective2025]. All code used for the experiments can be found on GitHub^[https://github.com/Syn-Tax/cross-domain-am].

## In-Domain {#sec:res-id}

Results are reported for both the 3-class problem (considering support, attack and no relation) and the 4-class problem (considering RA, CA, MA and NO). First results are considered when evaluating in-domain, i.e. on the test set of the QT30-MM dataset. Only macro-f1 scores are reported here, precision and recall scores are also reported in Appendix @app:results.

### The 4-Class Problem {#sec:id-4class}

Table \ref{tbl:results-early-4class} shows the macro-f1 scores of each model using RoBERTa-base as the text encoder and Wav2Vec2-base as the audio encoder. All that is shown are the results for models performing early sequence fusion across different NO-sampling strategies. In the 4-class problem it does not seem that the addition of acoustic features makes much if any difference to the performance of the model on the ARI task. This result has also been found by others [@mestreMArgMultimodalArgument2021].

In some runs, the audio only models do not learn above the performance of the random baseline, it is unclear why this occurs but it appears to be random. Comparing the NO-sampling strategies there does not seem to make an appreciable difference when tested on SCS. Because SCS is considered a much more challenging task [@ruiz-dolzLookingUnseenEffective2025] it would not be expected that models trained on LCS or US would perform nearly as well as those trained on SCS, however, this does not seem to be the case. The result of this experiment implies that regardless of the sampling strategy the model is able to acquire the same knowledge and would likely perform similarly in a real-world setting, it is simply that SCS evaluations are harder than US evaluations which in turn are harder than LCS evaluations.

\begin{table}[h]
\centering
\caption{Macro-F1 scores for early sequence fusion models on the 4-class problem. Highest results in each column are shown in bold.\label{tbl:results-early-4class}}
\begin{tabular}{|l|lll|}
\hline
Model         & SCS          & LCS          & US          \\ \hline
Text Only     & \textbf{.58} & \textbf{.59} & \textbf{.59} \\
Audio Only    & .43          & .41          & .20          \\ \hline
Concatenation & \textbf{.58} & .57          & .58          \\
Product       & .56          & .57          & .58          \\
CA Text       & .57          & .46          & .57          \\
CA Audio      & \textbf{.58} & .57          & .57          \\ \hline
Random        & .22          & .23          & .24          \\
Majority      & .14          & .14          & .14          \\ \hline
\end{tabular}
\end{table}

The different sequence fusion techniques are also compared in Table \ref{tbl:results-seq-4class}. Here it can be seen that early sequence fusion techniques outperform late fusion techniques by approximately 50% for both text only and audio only with concatenation showing greater improvement at approximately 60%. This increase in performance is likely attributed to the fact that when the sequences are fused before the attention mechanisms are applied the model is able to make the long-range dependencies across sequences. This comes in contrast to the fact that the model is unable to make any dependencies cross-sequence when they are fused after the attention mechanisms are applied.

\begin{table}[h]
\centering
\caption{Macro-F1 scores across sequence fusion types when trained on SCS on the 4-class problem. \label{tbl:results-seq-4class}}
\begin{tabular}{|l|ll|}
\hline
Model         & Early       & Late      \\ \hline
Text Only     & .58         & .36       \\
Audio Only    & .43         & .28       \\
Concatenation & .58         & .38       \\ \hline
Random        & \multicolumn{2}{c|}{.22} \\
Majority      & \multicolumn{2}{c|}{.14} \\ \hline
\end{tabular}
\end{table}

What follows is a more in-depth discussion of the results with the hope that it will yield some understanding of the models' limitations and how they could be improved. In order to do this the text only model trained on SCS is explored in detail, however, the conclusions were found to hold on other models. A good place to begin here is by analysing the class F1 distribution, here F1 scores are reported for each of the four classes (NO, RA, CA and MA). As can be seen in Table \ref{tbl:class-f1-4class} the model performs significantly worse when shown a conflict relation as opposed to the other possible classes. It is possible that this is due to the significant class imbalance present in almost all ARI datasets as discussed in Section @sec:datasets.

\begin{table}[h]
\centering
\caption{Class F1 distribution for text only SCS model.\label{tbl:class-f1-4class}}
\begin{tabular}{|llll|}
\hline
NO & RA & CA & MA              \\ \hline
.76         & .60         & .30         & .66 \\ \hline
\end{tabular}
\end{table}

A further analysis can be conducted by looking at the confusion matrix generated as shown in Figure \ref{fig:text-only-conf-mat-4class}. The ideal confusion matrix shows a diagonal line, in this case from the top left down to the bottom right of the matrix, and can be used to determine which classes the model struggles to distinguish. For ARI, it is generally expected that the model is able to distinguish CA from other classes, while RA and MA are often confused with each other and sometimes with NO. This follows from the difficulties that human annotators have when determining the different relations [@lawrenceArgumentMiningSurvey2020]. However, this is not what Figure \ref{fig:text-only-conf-mat-4class} shows, instead the model is generally confusing most classes, most notable is the underprediction of the CA class. This implies the model is simply predicting the majority classes which is a well known and well studied problem in all classification problems involving unbalanced data [@junsomboonCombiningOverSamplingUnderSampling2017]. Typically such problems are relatively simple to solve, often using either weighted loss functions (as is explained in Section @sec:exp-setup) or some form of data augmentation or manipulation technique. During this project, primarily random resampling was experimented with. Random resampling generally involves a combination (or only one) of oversampling minority classes (e.g. randomly duplicating samples labelled as CA) or undersampling majority classes (e.g. randomly discarding samples labelled as RA or NO). Often this has been shown to be the best technique for solving class-imbalance problems, despite the rise of other resampling techniques (such as SMOTE) [@mohammedMachineLearningOversampling2020]. Unfortunately in this case no resampling distribution could be found to meaningfully improve the model performance.

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{text-only-conf-mat-4class}
\caption{Confusion matrix showing true and predicted labels for the text only SCS model.\label{fig:text-only-conf-mat-4class}}
\end{figure}

### The 3-Class Problem {#sec:id-3class}

Table \ref{tbl:results-early-3class} shows the macro-f1 scores across each NO-sampling strategy for the 3-class problem using RoBERTa-base as the text encoder and Wav2Vec2-base as the audio encoder. Similarly to the 4-class problem, the addition of acoustic features to not seem to make an appreciable difference to the performance of the model. However, it is still useful to discuss the results in more detail, again using the text only model trained on SCS.

\begin{table}[h]
\centering
\caption{Macro-F1 scores for early sequence fusion models on the 3-class problem. Highest results in each column are shown in bold.\label{tbl:results-early-3class}}
\begin{tabular}{|l|lll|}
\hline
Model         & SCS          & LCS          & US           \\ \hline
Text Only     & \textbf{.62} & .59          & .61          \\
Audio Only    & .21          & .54          & .22          \\ \hline
Concatenation & .61          & .61          & .60          \\
Product       & .58          & .61          & .62          \\
CA Text       & .53          & .53          & .52          \\
CA Audio      & .61          & \textbf{.62} & \textbf{.63} \\ \hline
Random        & .28          & .30          & .29          \\
Majority      & .22          & .22          & .22          \\ \hline
\end{tabular}
\end{table}

Table \ref{tbl:results-seq-3class} shows the results when considering the different sequence fusion techniques. It can be seen that early fusion techniques significantly outperform late sequence fusion. Similarly to the 4-class US run, the early audio only model was not able to learn during this run. For both the text only and multimodal models, early fusion improves upon late fusion by approximately 44%.

\begin{table}[h]
\centering
\caption{Macro-F1 scores across sequence fusion types when trained on SCS on the 3-class problem. \label{tbl:results-seq-3class}}
\begin{tabular}{|l|ll|}
\hline
Model         & Early       & Late      \\ \hline
Text Only     & .62         & .43    \\
Audio Only    & .21         & .33    \\
Concatenation & .61         & .43    \\ \hline
Random        & \multicolumn{2}{c|}{.23} \\
Majority      & \multicolumn{2}{c|}{.15} \\ \hline
\end{tabular}
\end{table}

What follows is a more in-depth analysis of the 3-class in-domain results. The class F1 distribution for the 3-class problem is shown in Table \ref{tbl:class-f1-3class}. Similarly to the 4-class F1 distribution, the Attack class appears to be significantly harder to prodict than the other classes. Similarly to the 4-class problem it can be hypothesised that this is due to the class imbalance in the dataset. What is interesting, is the fact that the class F1 scores for the Attack/CA relations have dropped when compared with the 4-class results and the scores for non-attack/CA relations have risen. It is possible that this is simply due to the change in the number of classes, however, it can also be considered that the 3-class dataset is more heavily unbalanced against the Attack relation which could also have this effect.

\begin{table}[h]
\centering
\caption{Class F1 distribution for text only SCS model on test split.\label{tbl:class-f1-3class}}
\begin{tabular}{|lll|}
\hline
None & Support & Attack              \\ \hline
.82         & .78         & .26  \\ \hline
\end{tabular}
\end{table}

The confusion matrix for the 3-class problem, as shown in Figure \ref{fig:text-only-conf-mat-3class} can also be discussed. Here the underprediction of the attack class and the overprediction of the majority class. When compared to the 4-class confusion matrix (Figure \ref{fig:text-only-conf-mat-4class}) it appears that the model is effectively combining incorrect predictions between samples labelled inference and rephrase. This result could imply that the model's learning is approximately equivalent between the 3- and 4-class approaches, although it should be noted that more investigation is required to prove this.

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{text-only-conf-mat-3class}
\caption{Confusion matrix showing true and predicted labels for the text only SCS model.\label{fig:text-only-conf-mat-3class}}
\end{figure}

## Cross-Domain {#sec:res-cd}

In this section the results presented in Section @sec:res-id are extended across the Moral Maze subcorpora. Results here are presented across each subcorpus and the arithmetic mean calculated and also reported. The goal of this section is to analyse how well the models and techniques are able to generalise into different topics and domains. Similar to the In-Domain results, first the 4-class problem is discussed and then its 3-class equivalent. Only the Macro-F1 scores are reported here with precision and recall scores added in Appendix @app:results.

### The 4-Class Problem {#sec:cd-4class}

Table \ref{tbl:cross-4-SCS} provides a cross-domain evaluation of the different model architectures across the nine Moral Maze subcorpora when trained on SCS. Taking a broad overview, there is a distinct lack of significant improvement when the addition of acoustic features is considered. Since the evaluation is significantly harder than evaluating in-domain, the models show a significant decrease in the macro-F1 scores when comparing back to the in-domain results. This drop shows how challenging it is for the models to generalise effectively across the different domains.

\begin{table}[h]
\centering
\caption{Mean cross-domain macro-F1 scores for early sequence fusion models on the 4-class problem across different sampling strategies. Highest results in each column are shown in bold.\label{tbl:results-cd-4class-sampling}}
\begin{tabular}{|l|lll|}
\hline
Model         & SCS          & LCS          & US           \\ \hline
Text Only     & .46          & \textbf{.46} & \textbf{.46}          \\
Audio Only    & .37          & .36          & .22          \\ \hline
Concatenation & .45          & .43          & .45          \\
Product       & .45          & .44          & \textbf{.46}          \\
CA Text       & .43          & .32          & .41          \\
CA Audio      & \textbf{.47} & \textbf{.46} & .43 \\ \hline
Random        & .23          & .23          & .23          \\
Majority      & .15          & .15          & .15          \\ \hline
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Macro-F1 scores across sequence fusion types when trained on SCS on the 4-class problem. The mean is taken across all Moral Maze subcorpora. \label{tbl:results-seq-4class-cd}}
\begin{tabular}{|l|ll|}
\hline
Model         & Early       & Late      \\ \hline
Text Only     & .46         & .30       \\
Audio Only    & .37         & .28       \\
Concatenation & .45         & .30       \\ \hline
Random        & \multicolumn{2}{c|}{.23} \\
Majority      & \multicolumn{2}{c|}{.15} \\ \hline
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Class F1 distribution for text only and CA audio SCS models on banking subcorpus.\label{tbl:class-f1-4class-banking}}
\begin{tabular}{|l|llll|}
\hline
Model     & NO & RA & CA & MA              \\ \hline
Text Only & .67         & .60         & .41         & .065 \\
CA Audio  & .65         & .64         & .41         & .095 \\ \hline
\end{tabular}
\end{table}

\begin{table*}[t!]
\centering
\caption{Cross-Domain macro-averaged F1 scores on 4-class SCS trained models. Best scores in each column are shown in bold. \label{tbl:cross-4-SCS}}
\begin{tabular}{|l|llllllll|l|}
\hline
Model         & B            & E            & M            & P            & S            & G            & H            & W            & Mean \\ \hline
Text Only     & .44          & .46          & \textbf{.48} & .42          & .43          & .50          & \textbf{.51} & .41          & .46  \\
Audio Only    & .34          & .37          & .39          & .38          & .33          & .38          & .40          & .37          & .37 \\ \hline
Concatenation & .43          & .43          & .45          & \textbf{.45} & .45          & .49          & .45          & .43          & .45  \\
Product       & .41          & .44          & .42          & .42          & \textbf{.46} & \textbf{.52} & .46          & .43          & .45 \\
CA Text       & .40          & .44          & .44          & .40          & .42          & .49          & .43          & .44          & .43  \\
CA Audio      & \textbf{.45} & \textbf{.49} & .44          & .42          & .45          & .53          & .48          & \textbf{.46} & \textbf{.47}  \\ \hline
Random        & .19          & .23          & .20          & .19          & .22          & .25          & .21          & .24          & .23 \\
Majority      & .16          & .15          & .15          & .15          & .15          & .14          & .15          & .14          & .15  \\ \hline
\end{tabular}
\end{table*}

Table \ref{tbl:results-seq-4class-cd} compares early and late sequence fusion in a cross-domain setting. The results reported are the arithmetic mean across the Moral Maze subcorpora. Generally the increase is similar to that found in the in-domain evaluation with approximately a 50% increase for the text only and multimodal models and a 32% increase in performance for the audio only model.

Table \ref{tbl:results-cd-4class-sampling} can be used to compare the different NO-sampling strategies which again, seem to show little difference between the various methods, both for unimodal and multimodal approaches. Similarly to the in-domain results, what follows is a close look into the results of both the text only and CA audio models when trained on SCS and evaluated on the Banking subcorpus.

The class F1 distribution can be found in Table \ref{tbl:class-f1-4class}. Similarly to the in-domain results, the model is very able to predict unrelated pairs and pairs connected by an inference, and the score for pairs connected by a conflict are similar. The major difference in the class distribution is the inability of the model to accurately predict rephrases, it is possible that this is the result of an increase in domain specific knowledge and terminology necessary to predict these rephrases but these data are in no way conclusive in that respect.

Figures \ref{fig:res-cd-text-banking} and \ref{fig:res-cd-ca-banking} show the confusion matrices for both the text only model and the CA audio model when trained on SCS and evaluated on the Banking subcorpus. While the matrices are generally very similar, there are some notable differences that show as a trend across all subcorpora. Firstly the crossmodal attention model is better able to distinguish the rephrases. The text only model seems to be more likely to predict MA for true CA or RA samples. The other notable distinction between the two models' performance is that the text only model is more likely to confuse conflict-labelled pairs as being unrelated, as opposed to the crossmodal attention model which is more likely to predict the label as an inference in all cases. Similarly to the in-domain results, there was no resampling of the training data that could be found to change the F1 distribution across classes.

\begin{figure}[H]
\centering
\centering
\includegraphics[width=5cm]{text-only-conf-mat-4class-banking}
\caption{Text only model confusion matrix on the banking subcorpus.\label{fig:res-cd-text-banking}}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=5cm]{ca-audio-conf-mat-4class-banking}
\caption{CA Audio model confusion matrix on the banking subcorpus.\label{fig:res-cd-ca-banking}}
\end{figure}

\begin{table*}[t]
\centering
\caption{Cross-Domain macro-averaged F1 scores on 3-class SCS trained models. Best scores in each column are shown in bold. \label{tbl:cross-3-SCS}}
\begin{tabular}{|l|llllllll|l|}
\hline
Model         & B            & E            & M            & P            & S            & G            & H            & W            & Mean  \\ \hline
Text Only     & .54          & .47          & .50          & \textbf{.50} & .58          & .55          & .63          & .51          & \textbf{.54} \\
Audio Only    & .21          & .20          & .22          & .21          & .20          & .21          & .22          & .21          & .21           \\ \hline
Concatenation & .58          & \textbf{.49} & .47          & \textbf{.50} & .57          & \textbf{.57} & .59          & .49          & \textbf{.54} \\
Product       & .51          & .45          & .44          & .47          & .53          & .54          & \textbf{.64} & .47          & .51          \\
CA Text       & .43          & .40          & .43          & .41          & .44          & .46          & .47          & .44          & .44          \\
CA Audio      & \textbf{.59} & .44          & \textbf{.52} & .48          & \textbf{.59} & .54          & .60          & \textbf{.52} & \textbf{.54} \\ \hline
Random        & .27          & .32          & .30          & .27          & .30          & .29          & .29          & .33          & .30          \\
Majority      & .21          & .21          & .22          & .21          & .20          & .21          & .22          & .20          & .21           \\ \hline
\end{tabular}
\end{table*}

### The 3-Class Problem {#sec:cd-3class}

Next, the data from the 3-class problem is extended across different domains. The data can be seen in Table \ref{tbl:cross-3-SCS}. Generally the results are similar to the 4-class problem with a similar drop in macro-F1 scores from the in-domain results. Although the 3-class problem is slightly easier than the 4-class problem, the drop in performance is still roughly similar.

Table \ref{tbl:results-seq-3class-cd} allows a comparison between early and late sequence fusion methods on the 3-class problem. Here for both the text only and multimodal approaches early fusion improves performance by around 40% whereas the early audio only model did not learn effectively. It should be noted that although generally achieving lower performance, the late fusion models were always able to learn when only trained on audio data.

\begin{table}[H]
\centering
\caption{Macro-F1 scores across sequence fusion types when trained on SCS on the 3-class problem. The mean is taken across all Moral Maze subcorpora. \label{tbl:results-seq-3class-cd}}
\begin{tabular}{|l|ll|}
\hline
Model         & Early       & Late      \\ \hline
Text Only     & .54         & .38    \\
Audio Only    & .21         & .33    \\
Concatenation & .54         & .40    \\ \hline
Random        & \multicolumn{2}{c|}{.23} \\
Majority      & \multicolumn{2}{c|}{.15} \\ \hline
\end{tabular}
\end{table}

What follows is another discussion regarding the detailed results of the CA Audio model and the text only model. First, the class F1 distribution can be found in Table \ref{tbl:class-f1-3class-banking}. Here, as could reasonably be expected, the results differ significantly from the 4-class problem. The model is able to effectively classify the support relations and no significant drop is observed when compared to the in-domain results. However, the drop in macro-F1 seems to originate from the drop in the model's ability to classify unrelated nodes.

\begin{table}[H]
\centering
\caption{Class F1 distribution for text only and CA audio SCS models on banking subcorpus.\label{tbl:class-f1-3class-banking}}
\begin{tabular}{|l|llll|}
\hline
Model     & None & Support & \multicolumn{1}{l|}{Attack} \\ \hline
Text Only & .66           & .72              & \multicolumn{1}{l|}{.26}         \\
CA Audio  & .58           & .69              & \multicolumn{1}{l|}{.29}         \\ \hline
\end{tabular}
\end{table}

Figure \ref{fig:res-cd-ca-banking-3} shows the confusion matrix for the CA audio model when evaluated on the banking subcorpus. Much like what was seen in Section @sec:id-3class the matrix is characterised by the overprediction of Support relations, with the main difference being that this overprediction is worse than when evaluated in-domain.

\begin{figure}[H]
\centering
\includegraphics[width=8cm]{ca-audio-conf-mat-3class-banking}
\caption{CA Audio model confusion matrix on the banking subcorpus.\label{fig:res-cd-ca-banking-3}}
\end{figure}

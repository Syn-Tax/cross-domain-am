---
bibliography: [../../Cross-Domain AM.bib]
---

# Results

## Experimental Setup {#sec:exp-setup}

To provide a comparable set of results, all experiments were run using the same hyperparameters. Each model was trained on a single Nvidia RTX 4070 Super for 15 epochs with a batch size of 32 using a weighted cross-entropy loss and the AdamW optimiser [@loshchilovDecoupledWeightDecay2019a] initialised with a learning rate of $10^{-5}$, a linear learning rate scheduler and 10% of training used as warm-up steps. The cross-entropy weights were calculated as in Equation @eq:weight, where $\mathbf{c}$ is a vector containing the number of samples in each class, and $\mathbf{w}$ is a vector containing the relevant cross-entropy weight.

$$ w_i = \frac{\max(\mathbf{c})}{c_i} $$ {#eq:weight}

In order to evaluate the models, the following metrics are reported: macro-averaged F1 score, precision and recall. These are all described for each class in Equations @eq:f1, @eq:precision and @eq:recall where $TP$ is the number of true positives, $FP$ is the number of false positives and $FN$ is the number of false negatives. The arithmetic mean can then be taken for each class to provide a holistic overview of the model's performance.

$$ F1 = \frac{2TP}{2TP + FP + FN} $$ {#eq:f1}

$$ \text{Precision} = \frac{TP}{TP + FP} $$ {#eq:precision}

$$ \text{Recall} = \frac{TP}{TP + FN} $$ {#eq:recall}

Generally the macro-averaged F1 score is the standard to evaluate a multi-class classification problem, including ARI systems [@ruiz-dolzTransformerBasedModelsAutomatic2021;@ruiz-dolzLookingUnseenEffective2025;@manciniMAMKitComprehensiveMultimodal2024], where only a single metric is reported, it will be a macro-averaged F1 score for this reason.

The QT30-MM dataset is split into three splits: train, validation and test. 70% of the data is allocated for training, 10% for validation and the remaining 20% for testing. The model is evaluated on the validation split after every training epoch, the best performing model, based on macro-F1, is then chosen to be tested on the testing split, the metrics are then calculated and reported in the following sections.

Each model is then evaluated on the complete dataset for each Moral Maze episode (Banking, Empire, Money, Problem, Syria, Green Belt and D-Day) and each metric calculated to provide an overview of the cross-domain performance of the model.

In order to evaluate the different methods to sample unrelated arguments as described in Section @sec:pair-creation. Models are trained on SCS, US and LCS, the validation dataset is sampled identically to the training set, and tested, both in-domain and cross-domain on SCS. This is used because of its description as a more realistic problem [@ruiz-dolzLookingUnseenEffective2025]. All code used for the experiments can be found on GitHub^[https://github.com/Syn-Tax/cross-domain-am].

## In-Domain

Results are reported for both the 3-class problem (considering support, attack and no relation) and the 4-class problem (considering RA, CA, MA and NO). First results are considered when evaluating in-domain, i.e. on the test set of the QT30-MM dataset. Only macro-f1 scores are reported here, precision and recall scores are also reported in Appendix @app:results.

### The 4-Class Problem

Table \ref{tbl:results-early-4class} shows the macro-f1 scores of each model using RoBERTa-base as the text encoder and Wav2Vec2-base as the audio encoder. All that is shown are the results for models performing early sequence fusion across different NO-sampling strategies. In the 4-class problem it does not seem that the addition of acoustic features makes much if any difference to the performance of the model on the ARI task. This result has also been found by others [@mestreMArgMultimodalArgument2021]. However, a more in-depth discussion of the results may still yield some understanding in their limitations and how they could be improved. In order to do this the Text-Only model trained on SCS is explained in detail, however, the conclusions were found to hold on other models.

\begin{table}[h]
\centering
\caption{Macro-F1 scores for early sequence fusion models on the 4-class problem. Highest results are shown in bold.\label{tbl:results-early-4class}}
\begin{tabular}{|llll|}
\hline
\multicolumn{1}{|l|}{Model}         & \multicolumn{1}{c|}{\textbf{SCS}} & \multicolumn{1}{c|}{\textbf{LCS}} & \multicolumn{1}{c|}{\textbf{US}} \\ \hline
\multicolumn{1}{|l|}{Text-Only}     & .58                               & \textbf{.59}                      & \textbf{.59}                     \\
\multicolumn{1}{|l|}{Audio-Only}    & .43                               & .41                               & .20                              \\ \hline
\multicolumn{1}{|l|}{Concatenation} & .58                               & .57                               & .58                              \\
\multicolumn{1}{|l|}{Product}       & .56                               & .57                               & .58                              \\
\multicolumn{1}{|l|}{CA Text}       & .57                               & .46                               & .57                              \\
\multicolumn{1}{|l|}{CA Audio}      & .58                               & .57                               & .57                              \\ \hline
\multicolumn{1}{|l|}{Random}        & .22                               & .23                               & .24                              \\
\multicolumn{1}{|l|}{Majority}      & .14                               & .14                               & .14                              \\ \hline
\end{tabular}
\end{table}

A good place to begin here is by analysing the class F1 distribution, here F1 scores are reported for each of the four classes (NO, RA, CA and MA). As can be seen in Table \ref{tbl:class-f1-4class} the model performs significantly worse when shown a conflict relation as opposed to the other possible classes. It is possible that this is due to the significant class imbalance present in almost all ARI datasets as discussed in Section @sec:datasets.

\begin{table}[h]
\centering
\caption{Class F1 distribution for text-only SCS model on test split.\label{tbl:class-f1-4class}}
\begin{tabular}{|llll|}
\hline
\textbf{NO} & \textbf{RA} & \textbf{CA} & \textbf{MA}              \\ \hline
.76         & .60         & .30         & .66 \\ \hline
\end{tabular}
\end{table}

A further analysis can be conducted by looking at the confusion matrix generated as shown in Figure \ref{fig:text-only-conf-mat-4class}. The ideal confusion matrix shows a diagonal line, in this case from the top left down to the bottom right of the matrix, and can be used to determine which classes the model struggles to distinguish. For ARI, it is generally expected that the model is able to distinguish CA from other classes, while RA and MA are often confused with each other and sometimes with NO. This follows from the difficulties that human annotators have when determining the different relations [@lawrenceArgumentMiningSurvey2020]. However, this is not what Figure \ref{fig:text-only-conf-mat-4class} shows, instead the model is generally confusing most classes, most notable is the overprediction of the RA class. This implies the model is simply predicting the majority class which is a well known and well studied problem in all classification problems involving unbalanced data [@junsomboonCombiningOverSamplingUnderSampling2017]. Typically such problems are relatively simple to solve, often using either weighted loss functions (as is explained in Section @sec:exp-setup) or some form of data augmentation or manipulation technique. During this project, primarily random resampling was experimented with. Random resampling generally involves a combination (or only one) of oversampling minority classes (e.g. randomly duplicating samples labelled as CA) or undersampling majority classes (e.g. randomly discarding samples labelled as RA or NO). Often this has been shown to be the best technique for solving class-imbalance problems, despite the rise of other resampling techniques (such as SMOTE) [@mohammedMachineLearningOversampling2020]. Unfortunately in this case no resampling method or distribution could be found to meaningfully improve the model performance.

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{text-only-conf-mat-4class}
\caption{Confusion matrix showing true and predicted labels for the text-only SCS model.\label{fig:text-only-conf-mat-4class}}
\end{figure}

### The 3-Class Problem

Table \ref{tbl:results-early-3class} shows the macro-f1 scores across each NO-sampling strategy for the 3-class problem using RoBERTa-base as the text encoder and Wav2Vec2-base as the audio encoder. Similarly to the 4-class problem, the addition of acoustic features to not seem to make an appreciable difference to the performance of the model. However, it is still useful to discuss the results in more detail, again using the text-only model trained on SCS.

\begin{table}[h]
\centering
\caption{Macro-F1 scores for early sequence fusion models on the 3-class problem. Highest results are shown in bold.\label{tbl:results-early-3class}}
\begin{tabular}{|llll|}
\hline
\multicolumn{1}{|l|}{Model}         & \multicolumn{1}{c|}{\textbf{SCS}} & \multicolumn{1}{c|}{\textbf{LCS}} & \multicolumn{1}{c|}{\textbf{US}} \\ \hline
\multicolumn{1}{|l|}{Text-Only}     & .62                               & .59                               & .61                              \\
\multicolumn{1}{|l|}{Audio-Only}    & .21                               & .54                               & .22                              \\ \hline
\multicolumn{1}{|l|}{Concatenation} & .61                               & .61                               & .60                              \\
\multicolumn{1}{|l|}{Product}       & .58                               & .61                               & .62                              \\
\multicolumn{1}{|l|}{CA Text}       & .53                               & .53                               & .52                              \\
\multicolumn{1}{|l|}{CA Audio}      & .61                               & .62                               & \textbf{.63}                              \\ \hline
\multicolumn{1}{|l|}{Random}        & .28                               & .30                               & .29                              \\
\multicolumn{1}{|l|}{Majority}      & .22                               & .22                               & .22                              \\ \hline
\end{tabular}
\end{table}

The class F1 distribution for the 3-class problem is shown in Table \ref{tbl:class-f1-3class}. Similarly to the 4-class F1 distribution, the Attack class appears to be significantly harder to prodict than the other classes. Similarly to the 4-class problem it can be hypothesised that this is due to the class imbalance in the dataset. What is interesting, is the fact that the class F1 scores for the Attack/CA relations have dropped when compared with the 4-class results and the scores for non-attack/CA relations have risen. It is possible that this is simply due to the change in the number of classes, however, it can also be considered that the 3-class dataset is more heavily unbalanced against the Attack relation which could also have this effect.

\begin{table}[h]
\centering
\caption{Class F1 distribution for text-only SCS model on test split.\label{tbl:class-f1-3class}}
\begin{tabular}{|lll|}
\hline
\textbf{None} & \textbf{Support} & \textbf{Attack}              \\ \hline
.82         & .78         & .26  \\ \hline
\end{tabular}
\end{table}

The confusion matrix for the 3-class problem, as shown in Figure \ref{fig:text-only-conf-mat-3class} can also be discussed. Here the overprediction of the Support class is a bit more obvious

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{text-only-conf-mat-3class}
\caption{Confusion matrix showing true and predicted labels for the text-only SCS model.\label{fig:text-only-conf-mat-3class}}
\end{figure}

## Cross-Domain
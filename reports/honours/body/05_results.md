# Results

## Experimental Setup

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

\begin{table}[h]
\centering
\caption{Macro-F1 scores for late sequence fusion models on the 4-class problem. Highest results are shown in bold.\label{tbl:results-late}}
\begin{tabular}{|llll|}
\hline
\multicolumn{4}{|c|}{\textbf{4-class}}                                                                                                         \\ \hline
\multicolumn{1}{|l|}{Model}         & \multicolumn{1}{c|}{\textbf{SCS}} & \multicolumn{1}{c|}{\textbf{LCS}} & \multicolumn{1}{c|}{\textbf{US}} \\ \hline
\multicolumn{4}{|c|}{\textbf{Unimodal}}                                                                                                        \\ \hline
\multicolumn{1}{|l|}{Text-Only}     & .58                               & \textbf{.59}                      & \textbf{.59}                     \\
\multicolumn{1}{|l|}{Audio-Only}    & .43                               & .41                               & .20                              \\ \hline
\multicolumn{4}{|c|}{\textbf{Multimodal}}                                                                                                      \\ \hline
\multicolumn{1}{|l|}{Concatenation} & .58                               & .57                               & .58                              \\
\multicolumn{1}{|l|}{Product}       & .56                               & .57                               & .58                              \\
\multicolumn{1}{|l|}{CA Text}       & .57                               & .46                               & .57                              \\
\multicolumn{1}{|l|}{CA Audio}      & .58                               & .57                               & .57                              \\ \hline
\multicolumn{4}{|c|}{\textbf{Baselines}}                                                                                                       \\ \hline
\multicolumn{1}{|l|}{Random}        & .22                               & .23                               & .24                              \\
\multicolumn{1}{|l|}{Majority}      & .14                               & .14                               & .14                              \\ \hline
\end{tabular}
\end{table}

Table \ref{tbl:results-late} shows the macro-f1 scores of each model using RoBERTa-base as the text encoder and Wav2Vec2 as the audio encoder. All that is shown are the results for models performing late sequence fusion across different NO-sampling strategies. In the 4-class problem it does not seem that the addition of acoustic features makes much if any difference to the performance of the model on the ARI task. This result has also been found to  However, a more in-depth discussion of the results may still yield some understanding in their limitations and how they could be improved. In order to do this the Text-Only model trained on SCS is analysed.

\begin{table}[h]
\centering
\caption{Class F1 distribution for text-only SCS model.\label{tbl:class-f1-4class}}
\begin{tabular}{|llll|}
\hline
\textbf{NO} & \textbf{RA} & \textbf{CA} & \textbf{MA}              \\ \hline
.76         & .60         & .30         & .66 \\ \hline
\end{tabular}
\end{table}

A good place to begin here is by analysing the class F1 distribution, here F1 scores are reported for each of the four classes (NO, RA, CA and MA). As can be seen in Table \ref{tbl:class-f1-4class} the model performs significantly worse when shown a conflict relation as opposed to the other possible classes. It is possible that this is due to the significant class imbalance present in almost all ARI datasets as discussed in Section @sec:datasets.

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{text-only-conf-mat-4class}
\caption{Confusion matrix showing true and predicted labels for the text-only SCS model.\label{fig:text-only-conf-mat-4class}}
\end{figure}

A further analysis can be conducted by looking at the confusion matrix generated as shown in Figure \ref{fig:text-only-conf-mat-4class}. The ideal confusion matrix shows a diagonal line, in this case from the top left down to the bottom right of the matrix.

## Cross-Domain
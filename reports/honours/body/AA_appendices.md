---
bibliography: [../../Cross-Domain AM.bib]
---

\appendix
# Results {#app:results}

## In-Domain {#app:res-id}

\begin{table*}[h]
\centering
\caption{In-domain results when testing different model architectures on the 4-class problem. F1: Macro-averaged F1, P: precision, R: recall. Highest scores in each column are shown in bold.}
\begin{tabular}{|lllllllllll|}
\hline
\multicolumn{2}{|c|}{Fusion Methods}                                              & \multicolumn{3}{c|}{\textbf{SCS}}                               & \multicolumn{3}{c|}{\textbf{LCS}}                               & \multicolumn{3}{c|}{\textbf{US}}           \\ \hline
\multicolumn{1}{|l|}{Sequence}               & \multicolumn{1}{l|}{Multimodal}    & F1           & P            & \multicolumn{1}{l|}{R}            & F1           & P            & \multicolumn{1}{l|}{R}            & F1           & P            & R            \\ \hline
\multicolumn{11}{|c|}{\textbf{Text Only}}                                                                                                                                                                                                                          \\ \hline
\multicolumn{1}{|l|}{Early}                  & \multicolumn{1}{l|}{-}             & \textbf{.58} & \textbf{.58} & \multicolumn{1}{l|}{\textbf{.58}} & \textbf{.59} & \textbf{.59} & \multicolumn{1}{l|}{\textbf{.59}} & \textbf{.59} & \textbf{.59} & \textbf{.59} \\
\multicolumn{1}{|l|}{Late}                   & \multicolumn{1}{l|}{-}             & .36          & .36          & \multicolumn{1}{l|}{.35}             & .34          & .36          & \multicolumn{1}{l|}{.34}             & .35          & .35          & .35          \\ \hline
\multicolumn{11}{|c|}{\textbf{Audio Only}}                                                                                                                                                                                                                         \\ \hline
\multicolumn{1}{|l|}{Early}                  & \multicolumn{1}{l|}{-}             & .43          & .48          & \multicolumn{1}{l|}{.44}          & .41          & .41          & \multicolumn{1}{l|}{.42}          & .20          & .31          & .26          \\
\multicolumn{1}{|l|}{Late}                   & \multicolumn{1}{l|}{-}             & .28          & .29          & \multicolumn{1}{l|}{.29}             & .29          & .31          & \multicolumn{1}{l|}{.29}             & .29          & .29          & .29          \\ \hline
\multicolumn{11}{|c|}{\textbf{Multimodal}}                                                                                                                                                                                                                         \\ \hline
\multicolumn{1}{|l|}{\multirow{4}{*}{Early}} & \multicolumn{1}{l|}{Concatenation} & \textbf{.58} & \textbf{.58} & \multicolumn{1}{l|}{\textbf{.58}} & .57          & .57          & \multicolumn{1}{l|}{.57}          & .58          & .58          & .57          \\
\multicolumn{1}{|l|}{}                       & \multicolumn{1}{l|}{Product}       & .56          & .56          & \multicolumn{1}{l|}{.57}          & .57          & .57          & \multicolumn{1}{l|}{.57}          & .58          & .58          & \textbf{.59} \\
\multicolumn{1}{|l|}{}                       & \multicolumn{1}{l|}{CA Text}       & .57          & .56          & \multicolumn{1}{l|}{\textbf{.58}} & .46          & .46          & \multicolumn{1}{l|}{.48}          & .57          & .56          & .57          \\
\multicolumn{1}{|l|}{}                       & \multicolumn{1}{l|}{CA Audio}      & \textbf{.58} & \textbf{.58} & \multicolumn{1}{l|}{\textbf{.58}} & .57          & \textbf{.59} & \multicolumn{1}{l|}{.56}          & .57          & .58          & .57          \\ \hline
\multicolumn{1}{|l|}{Late}                   & \multicolumn{1}{l|}{Concatenation} & .36          & .36          & \multicolumn{1}{l|}{.35}          & .37          & .36          & \multicolumn{1}{l|}{.36}          & .35          & .36          & .35          \\ \hline
\multicolumn{11}{|c|}{\textbf{Baselines}}                                                                                                                                                                                                                          \\ \hline
\multicolumn{2}{|c|}{Random}                                                      & .22          & .24          & \multicolumn{1}{l|}{.23}          & .23          & .25          & \multicolumn{1}{l|}{.24}          & .24          & .25          & .26          \\
\multicolumn{2}{|c|}{Majority}                                                    & .14          & .09          & \multicolumn{1}{l|}{.25}          & .14          & .09          & \multicolumn{1}{l|}{.25}          & .14          & .09          & .25          \\ \hline
\end{tabular}
\end{table*}

\begin{table*}[h]
\centering
\caption{In-domain results when testing different model architectures on the 3-class problem. F1: Macro-averaged F1, P: precision, R: recall. Highest scoring models (based on Macro-F1) are shown in bold.}
\begin{tabular}{|lllllllllll|}
\hline
\multicolumn{2}{|c|}{Fusion Methods}                                                       & \multicolumn{3}{c|}{\textbf{SCS}}                               & \multicolumn{3}{c|}{\textbf{LCS}}                               & \multicolumn{3}{c|}{\textbf{US}}           \\ \hline
\multicolumn{1}{|l|}{Sequence}               & \multicolumn{1}{l|}{Multimodal}    & F1           & P            & \multicolumn{1}{l|}{R}            & F1           & P            & \multicolumn{1}{l|}{R}            & F1           & P            & R            \\ \hline
\multicolumn{11}{|c|}{\textbf{Text Only}}                                                                                                                                                                                                                          \\ \hline
\multicolumn{1}{|l|}{Early}                  & \multicolumn{1}{l|}{-}             & \textbf{.62} & \textbf{.63} & \multicolumn{1}{l|}{\textbf{.62}} & .59          & \textbf{.63} & \multicolumn{1}{l|}{.58}          & .61          & .63          & .60          \\
\multicolumn{1}{|l|}{Late}                   & \multicolumn{1}{l|}{-}             & .43          & .45          & \multicolumn{1}{l|}{.43}             & .44          & .45          & \multicolumn{1}{l|}{.44}             & .43          & .44          & .44          \\ \hline
\multicolumn{11}{|c|}{\textbf{Audio Only}}                                                                                                                                                                                                                         \\ \hline
\multicolumn{1}{|l|}{Early}                  & \multicolumn{1}{l|}{-}             & .21          & .16          & \multicolumn{1}{l|}{.33}          & .54          & .54          & \multicolumn{1}{l|}{.55}          & .22          & .16          & .33          \\
\multicolumn{1}{|l|}{Late}                   & \multicolumn{1}{l|}{-}             & .33          & .33          & \multicolumn{1}{l|}{.34}             & .34          & .33           & \multicolumn{1}{l|}{.34}             & .34          & .33          & .35          \\ \hline
\multicolumn{11}{|c|}{\textbf{Multimodal}}                                                                                                                                                                                                                         \\ \hline
\multicolumn{1}{|l|}{\multirow{4}{*}{Early}} & \multicolumn{1}{l|}{Concatenation} & .61          & .62          & \multicolumn{1}{l|}{.60}          & .61          & .61          & \multicolumn{1}{l|}{.61}          & .60          & .61          & .59          \\
\multicolumn{1}{|l|}{}                       & \multicolumn{1}{l|}{Product}       & .58          & .60          & \multicolumn{1}{l|}{.57}          & .61          & .62          & \multicolumn{1}{l|}{.61}          & .62          & \textbf{.64} & .61          \\
\multicolumn{1}{|l|}{}                       & \multicolumn{1}{l|}{CA Text}       & .53          & .52          & \multicolumn{1}{l|}{.54}          & .53          & .52          & \multicolumn{1}{l|}{.55}          & .52          & .51          & .53          \\
\multicolumn{1}{|l|}{}                       & \multicolumn{1}{l|}{CA Audio}      & .61          & .62          & \multicolumn{1}{l|}{.60}          & \textbf{.62} & \textbf{.63} & \multicolumn{1}{l|}{\textbf{.62}} & \textbf{.63} & .62          & \textbf{.64} \\ \hline
\multicolumn{1}{|l|}{Late}                   & \multicolumn{1}{l|}{Concatenation} & .43          & .44          & \multicolumn{1}{l|}{.43}             & .45          & .47           & \multicolumn{1}{l|}{.45}             & .44          & .45          & .44          \\ \hline
\multicolumn{11}{|c|}{\textbf{Baselines}}                                                                                                                                                                                                                          \\ \hline
\multicolumn{2}{|c|}{Random}                                                      & .28          & .33          & \multicolumn{1}{l|}{.33}          & .30          & .34          & \multicolumn{1}{l|}{.36}          & .24          & .25          & .26          \\
\multicolumn{2}{|c|}{Majority}                                                    & .22          & .16          & \multicolumn{1}{l|}{.33}          & .22          & .16          & \multicolumn{1}{l|}{.33}          & .22          & .16          & .33          \\ \hline
\end{tabular}
\end{table*}

## Cross-Domain {#app:res-cd}
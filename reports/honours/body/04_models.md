---
bibliography: [../../Cross-Domain AM.bib]
---

# Models {#sec:models}

This section descries the model architectures that are evaluated in this project. Since the models will be aimed at a sequence pair classification task, there is a distinction between when data from each sequence is combined (which can be termed sequence fusion) and when data from each modality is combined (which can be termed multimodal fusion). The following subsections define the different approaches evaluated for each of these stages.

In previous work it has been shown that the RoBERTa models perform better than other pretrained transformers on the task of ARI [@ruiz-dolzTransformerBasedModelsAutomatic2021]. To encode the text data, the RoBERTa-base model is used, with 12 encoder layers, each with 12 attention heads and a hidden size of 768, resulting in 110M total parameters^[https://huggingface.co/FacebookAI/roberta-base]. Wav2Vec 2.0 is also used in many audio processing tasks [@manciniMAMKitComprehensiveMultimodal2024;@manciniMultimodalFallacyClassification2024] and therefore is used to encode the audio data. To ensure both models output the same hidden size, the wav2vec2-base model is used with 95M total parameters^[https://huggingface.co/facebook/wav2vec2-base-960h]. The wav2vec2 model used is fine-tuned on 960 hours of librispeech for automatic speech recognition. Once the data has been encoded, it can be fed into the classification head to output a final prediction.

The classification head used for most models is a simple linear projection from the hidden vector down to the required number of classes. The best performing model on ARI as proposed in MAMKit [@manciniMAMKitComprehensiveMultimodal2024] (MM-RoBERTa) is also evaluated. Their model uses the fusion architecture described in Figure \ref{fig:model-diag-late} but with a 3 layer Multilayer Perceptron (MLP) model as the classification head. They also only train the classification head, without training the text or audio encoders.

## Sequence Fusion {#sec:seq-fusion}

In most text-processing approaches data from each sequence is combined at the text level using special tokens defined in the encoder's tokeniser [@gemechuARIESGeneralBenchmark2024;@wuKnowCompDialAM2024Finetuning2024;@zhengKNOWCOMPPOKEMONTeam2024]. For this project, this approach is termed early sequence fusion. To achieve this, the input sequences can be delimited by a separator token, and the entire sequence wrapped in the start of sequence (SOS) and end of sequence (EOS) tokens. An example using the RoBERTa tokeniser takes the following form: `<s> [sequence 1] </s> [sequence 2] </s>`. Here the `<s>` token corresponds to the SOS, and `</s>` does the job of both the separator token and the EOS token.

Next the audio data can be considered. As far as was found, there is no existing literature on how early sequence fusion could function for audio models. Therefore, it was decided to deliniate each audio sequence by a certain amount of silence. The exact amount of silence could be adjusted as a training hyperparameter and was eventually set to 5 seconds.

Figure \ref{fig:model-diag-early} shows an example of a model architecture using this late fusion technique, text related steps are shown in purple and audio related steps are shown in green. RoBERTa and Wav2Vec2 are simply used as examples and could be substituted for other models.

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{model-diag-early}
\caption{Model diagram with early sequence and late multimodal fusion.\label{fig:model-diag-early}}
\end{figure}

Mestre *et al.* [@mestreMArgMultimodalArgument2021] and Mancini *et al.* [@manciniMAMKitComprehensiveMultimodal2024] approach the problem differently. They first put each sequence through the text encoder independently, before fusing the outputs and feeding the combined encodings into the classification head. While concatenation is the only fusion method examined for this sequence fusion technique, others (such as an element-wise product or cross-attention) could be used. This approach extends much more easily to the audio modality, since the audio encodings can be combined in the same way as the text encodings. This approach to fusing the data in each sequence can be termed late sequence fusion. Figure \ref{fig:model-diag-late} shows an example of a model architecture using late sequence fusion, text processing steps and data are shown in purple and audio-related steps are shown in green.

\begin{figure}[h]
\centering
\includegraphics[width=8cm]{model-diag-late}
\caption{Model diagram with late concatenation sequence and multimodal fusion.\label{fig:model-diag-late}}
\end{figure}

## Multimodal Fusion {#sec:mm-fusion}

Multimodal fusion decribes the method by which the text and audio data is combined. As detailed in Section @sec:background-ml, fusion techniques can be split into two major categories: early and late fusion. This project only evaluates late fusion techniques due to their ease of development and the applicability of pre-trained models. The following techniques are evaluated:

- **Concatenation** where the pooled encodings for each modality are simply concatenated before being fed into the classification head.
- **Elementwise-product** (otherwise known as a Hadamard product) takes the product of each element in the pooled encodings for each modality.
- **Crossmodal Attention** (CA) is similar to the self-attention mechanism found in transformers, however, the queries are taken from a different modality when compared to the keys and values. To compute crossmodal attention features, the query and key matrices are multiplied and then put through a softmax. This is then multiplied with the value matrix and the result can then be pooled using an arithmetic mean. For the purposes of this project, the CA module is labelled based on the modality from which the query matrix is derived (e.g. a `CA_Text` module derives the query matrix from the text encodings and the key and value matrices from the audio encodings). This is shown in Figure \ref{fig:crossmodal-attention}.

\begin{figure*}[t!]
\centering
\includegraphics[width=16cm]{crossmodal-attention}
\caption{Crossmodal attention system with both text and audio queries. $\otimes$ is used to denote matrix multiplication.\label{fig:crossmodal-attention}}
\end{figure*}

Listing \ref{lst:crossattention} shows how a crossmodal attention mechanism can be implemented into a PyTorch module. The code is contained within the forward method of a PyTorch module, where `self.query_proj`, `self.key_proj` and `self.value_proj` are the linear projections for the queries, keys and values respectively. `torch.bmm` refers to a matrix multiplication and `F.softmax` is a mathematical function to ensure all values (across the requested dimension) lie in the range $[0,1]$ and sum to 1, the softmax function for a vector $\mathbf{x}$ is given in Equation @eq:softmax.

```py {#lst:crossattention .numberLines caption="PyTorch forward method for a crossmodal attention mechanism."}
# project query features
queries = self.query_proj(query_modality)

# project kv features
keys = self.key_proj(kv_modality)
values = self.value_proj(kv_modality)

# compute attention scores
attn_scores = torch.bmm(queries, keys.transpose(1, 2))
attn_probs = F.softmax(attn_scores, dim=-1)

# compute and return cross modal features
return torch.bmm(attn_probs, values)
```

$$ \text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} $$ {#eq:softmax}
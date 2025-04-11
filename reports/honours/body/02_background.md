---
bibliography: [../../Cross-Domain AM.bib]
---

# Background

## Argumentation Theory

Argument and debate has been studied since the time of the ancient Greek philosophers and rhetoricians where argument theorists have sought to formalise discourse and discover some standard of proof for determining the 'correctness' of an argument. Over time, theories of arguments and discussions have evolved, notably when Hamblin [@hamblinFallacies1970] refashioned an argumentative discourse as a game, where one party makes moves offering premises that may be acceptable to the another party in the discourse who doubts the conclusion of the argument. When viewing a discourse as a game, it becomes possible to model discourse in such a way that it can be viewed through the lens of formal logic, and therefore computationally too.

In order to describe various dialogue, argument and illocutionary structures different models (annotation schemes) can be used, some annotation schemes focus on types of the text itself (such as speech act theory [@searleSpeechActsEssay1969]) or on the types of relations between components (such as Rhetorical Structure Theory [@mannRhetoricalStructureTheory1988]). Inference Anchoring Theory (IAT) [@reedHowDialoguesCreate2011] is an annotation scheme constructed to benefit from insights across both types, whilst focusing specifically on argumentative discourse. This makes IAT a very useful tool to analyse arguments and their relations.

In IAT, the discourse is first segmented into Argumentative Discourse Units (ADUs). An ADU is any span of the discourse which has both propositional content and discrete argumentative function [@reedQuickStartGuide2017]. An IAT argument graph is typically composed of two main parts: the left-hand side and the right-hand side. The right-hand side is concerned with locutions and transitions between them. A locution is simply the text of the ADU as uttered, without reconstructing ellipses or resolving pronouns. Locutions also include the speaker and may even include a timestamp. Transitions connect locutions capturing a functional relationship between predecessor and successor locutions (i.e. a response or reply). The left-hand side of an argument graph is more concerned with the content of the ADU, rather than directly reflecting what was uttered. This consists of the propositions made, and the relations between those propositions. To create a proposition from an ADU, the content is reconstructed to be a coherent, lone-standing sentence. This means that any missing or implicit material has to be reconstructed, including anaphoric references (e.g. pronouns). Performing this reconstruction allows both a human annotator and a computational system to view some of the context surrounding the locution and therefore make a better judgement as to a proposition's relation to others.

IAT defines three different types of propositional relation: *inference*, *conflict* and *rephrase*. An inference relation (also termed RA) holds between two propositions when one (the premise) is used to provide a reason to accept the other (the conclusion). This may include annotation of the kind of support e.g. Modus Ponens or Argument from Expert Opinion. These subtypes of relation are often called *argument schemes* [@waltonArgumentationSchemes2008;@waltonArgumentationTheoryVery2009]. There are also several different inference structures (images from [@lawrenceArgumentMiningSurvey2020]):

- **Serial arguments** occur when one proposition supports another, which in turn supports a third.

\begin{figure}[H]
    \centering
    \includegraphics[height=2cm]{serial}
\end{figure}

- **Convergent arguments** occur when multiple premises act independently to support the same conclusion.

\begin{figure}[H]
    \centering
    \includegraphics[height=2cm]{convergent}
\end{figure}

- **Linked arguments** occur when multiple premises work together to support a conclusion.

\begin{figure}[H]
    \centering
    \includegraphics[height=2cm]{linked}
\end{figure}

- **Divergent arguments** occur when a single premise is used to support multiple conclusions.

\begin{figure}[H]
    \centering
    \includegraphics[height=2cm]{divergent}
\end{figure}

A conflict relation (also termed CA) holds between two propositions when one is used to provide an incompatible alternative to another and can also be of a given kind (e.g. Conflict from Bias, Conflict from Propositional Negation). The following conflict structures are identified by IAT:

- **Rebutting conflict** occurs if one proposition is directly targeting another by indicating that the latter is not acceptable.
- **Undermining conflict** occurs if a conflict is targeting the premise of an argument, then it is undermining its conclusion.
- **Undercutting conflict** occurs if the conflict is targeting the inference relation between two propositions.

A rephrase relation (also termed MA) holds when one proposition rephrases, restates or reformulates another but with different propositional content (i.e. one proposition cannot simply repeat the other). There are many different kinds of rephrase, such as Specialisation, Generalisation, Instantiation etc. Generally, question answering will often involve a rephrase because the propositional content of the question is typically instantiated, resolved or refined by its answer. In contrast to inference, conflict and rephrase structures only have a single incoming an one outgoing edge.

The left and right-hand sides are connected by *illocutionary connections*. These illocutionary connections are based on illocutionary force as introduced by speech act theory [@searleSpeechActsEssay1969]. The speech act $F(p)$ is the act which relates the locution and and the propositional content $p$ through the illocutionary force $F$ e.g. asserting $p$, requesting $p$, promising $p$ etc. There are many diferent types of illocutionary connection, including: assertions, questions, challenges, concessions and (dis-)affirmations [@budzynskaModelProcessingIllocutionary2014]. By modelling these relations it is possible to gain a better understanding of how something is said and its purpose within the discourse. In doing this, we create a graph (known as an *argument map*) which can then be stored computationally.

Several ways to store argumentative data have been created, for example Argument Markup Language (AML) [@reedAraucariaSoftwareArgument2004], an XML-based language used to describe arguments in the Araucaria software. More recently, the Argument Interchange Format (AIF) [@chesnevarArgumentInterchangeFormat2006] has been created to standardise the storage of IAT graphs.

AIF treats all relevant parts of the argument as nodes within a graph. These nodes can be put into two categories: *information nodes* (I-nodes) and *scheme nodes* (S-nodes). I-nodes represent the claims made in the discourse whereas S-nodes indicate the application of an argument scheme. Initially I-nodes only included the propositions made [@chesnevarArgumentInterchangeFormat2006], but when Reed *et al.* [@reedAIFDialogueArgument] extended AIF to cater to dialogues, they added L-nodes as a subclass of I-nodes to represent locutions. For the purposes of this research, I-nodes and L-nodes are considered separate classes where I-nodes contain propositions and L-nodes contain locutions.

Since AIF data can be easily shared, it became the basis for a Worldwide Argument Web (WWAW) [@rahwanLayingFoundationsWorld2007]. Since then, many corpora have been annotated using IAT and published on the AIFdb^[https://www.aifdb.org/] [@lawrenceAIFdbInfrastructureArgument2012] providing a very useful resource for argumentation research of many kinds.

## Natural Language Processing {#sec:background-ml}

In recent years there have been several major advances in the field of natural language processing (NLP), most notably the introduction of the transformer architecture [@vaswaniAttentionAllYou2017]. The transformer architecture, based on self-attention, allows the model to determine much longer range dependencies than previous approaches. In doing so, the model is able to learn from the context surrounding a word, even gaining insight from context 'far away' in the text.

Even before Vaswani *et al.* introduced the transformer architecture supervised and semi-supervised pre-training approaches were already being explored, and proven to be a very useful tool for improving the performance of language models [@petersDeepContextualizedWord2018;@daiSemisupervisedSequenceLearning2015]. When the transformer was introduced these pre-training techniques were adapted for use in transformers creating models which are able to be fine-tuned with relatively minimal effort and compute to allow high performance on a wide variety of tasks [@devlinBERTPretrainingDeep2019b;@liuRoBERTaRobustlyOptimized2019a]. The pre-training approaches introduced by BERT and RoBERTa use a combination of masked language modelling (where the model is trained to predict the token hidden under a `[mask]` token) and next sentence prediction. The models are then trained using this approach on large datasets (the dataset used to pre-train RoBERTa totals over 160GB of uncompressed text).

Transformer models have recently become much more well-known due to the introduction of Large Language Models (LLMs) such as GPT-4 [@openaiGPT4TechnicalReport2024] and LLaMA [@touvronLLaMAOpenEfficient2023]. LLMs have proven very useful across NLP due to their ability to achieve high performance on many tasks without the need for fine-tuning, this can, however, include few-shot techniques to allow them to 'learn' at inference time [@brownLanguageModelsAre2020;@sharmaArgumentativeStancePrediction2023].

A similar progression can be seen in the development of audio models. Pre-training was notably introduced into speech recognition with wav2vec [@schneiderWav2vecUnsupervisedPretraining2019], where the model is trained to predict future samples from a given signal. The wav2vec model has two main stages, first raw audio samples are fed into a convolutional network which performs a similar role to the tokenisation seen in text-based language models by using a sliding window approach to downsample the audio data. These encodings are then fed into a second convolutional network to create a final encoding for the sequence.

Transformer models were introduced into the architecture of audio models with wav2vec2 [@baevskiWav2vec20Framework2020] and HuBERT [@hsuHuBERTSelfSupervisedSpeech2021], where the second convolutional model is replaced with a transformer in order to better learn dependencies across the entire sequence. These models are then pre-trained on significant amounts of audio data (960 hours in the case of wav2vec2) in order to then be fine-tuned on a downstream task. It is also possible to combine text and audio in order to gain insights from both modalities.

Combining modalities (such as text and audio) has also proven to be a useful tool across several tasks, including medical imaging [@delbrouckViLMedicFrameworkResearch2022;@sunCMAFNetCrossmodalAttention2024] and natural language processing [@totoAudiBERTDeepTransfer2021;@tsaiMultimodalTransformerUnaligned2019], including argument mining [@mestreMArgMultimodalArgument2021;@manciniMAMKitComprehensiveMultimodal2024]. Generally fusion techniques can be split into two categories: early and late. Early fusion techniques combine representations of each modality before being used as input to an encoder, with the primary benefit that only a single encoder is used. Late fusion techniques use a separate encoder for each modality, and the encodings are then fused to provide a crossmodal representation of the input.

In early fusion the input representations are transformed into a common information space, often using vectorisation techniques dependent on the modality. Late fusion techiques allow for the encodings of each modality to be combined in several different ways, often either simple operations (such as concatenation or an element-wise product) but a cross-modal attention module can also be used to combine the modalities [@rajanCrossAttentionPreferableSelfAttention2022;@yeCrossModalSelfAttentionNetwork2019]. The fusion techniques used in this project are explained in detail in Section @sec:models.

## Argument Mining

Various NLP techniques have been beneficial to AM, from statistical methods to the more recent neural networks, in particular the transformer architecture [@ruiz-dolzTransformerBasedModelsAutomatic2021]. Before discussing the automation of AM, it is useful to understand how argument analysis is conducted manually. Manual argument analysis considers the following steps:

- **Text Segmentation** involves the splitting of the original text/discourse into the pieces that will form the resulting argument structure. These pieces are often termed Elementary Discourse Units (EDUs).
- **Argument / Non-Argument Classification** is the task of determining which of the segments found in the text segmentation step are relevant to the argument. For most manual analysis, this step is performed in conjunction with text segmentation i.e. the analyst doesn't segment parts of the text which are not relevant to the argument.
- **Simple Structure** is the identification of relations between the arguments (e.g. inference, conflict and rephrase) and their structures (e.g. convergent, serial etc.).
- **Refined Structure** refers to the identification of argumentation schemes (e.g. Argument from Expert Opinion, Conflict From Bias etc.).

When the argument analysis process is automated, the stages are very similar to those in the manual process. Lawrence and Reed [@lawrenceArgumentMiningSurvey2020] define the steps as follows, increasing in computational complexity:

- **Identifying Argument Components** combines the stages of text segmentation and argument / non-argument classification in the manual process.
- **Identifying Clausal Properties** involves the identification of both intrinsic clausal properties (e.g. is X evidence?, is X reported speech?) of the ADU and the contextual properties (e.g. is X a premise?, is X a conclusion?).
- **Identifying Relational Properties** relates to the identification of *general relations* between ADUs (e.g. is X a premise for Y?, is X in conflict with Y?) and the identification of argument schemes.

Generally these stages of AM are not directly used in the literature, but instead a set of AM sub-tasks which map onto each of these stages, these tasks are defined by Mancini et *al.* [@manciniMAMKitComprehensiveMultimodal2024] as follows:

- **Argumentative Sentence Detection (ASD)** is the task of classifying a sequence as containing an argument, or not. ASD can be extended to include the task of claim detection, where a sequence is classified as containing a claim or not containing a claim.
- **Argument Component Classification (ACC)** is the task of determining whether an argumentative sentence $x$ contains one or more argumentative components e.g. a claim or premise. This is loosely analogous to the identification of clausal properties as defined by Lawrence and Reed.
- **Argumentative Relation Identification (ARI)** is the task of identifying the relation between a pair of sentences where given a pair $(x_i, x_j)$ the task is to identify the argumentative relation $x_i \rightarrow x_j$ across some relation model.

There are varying relation models for ARI, the most commonly used is simply classifying the pair as one of support (a combination of inference and rephrase), attack (conflict) or unrelated. For the purposes of this project this is termed 3-class ARI. ARI can also be conducted using all relations described in IAT (inference, rephrase and conflict), as well as unrelated nodes. For the purpose of this project this is termed 4-class ARI. Some literature makes a distinction between Argument Relation Identification and Argument Relation Classification, where the latter does not involve unrelated pairs (i.e. given that the pair $(x_i, x_j)$ is related, what is the type of relation?), however this distinction is by no means universal among AM literature [@gemechuARIESGeneralBenchmark2024].

Much of the AM literature only evaluates their systems in the same domain (dataset) as it was trained on [@lawrenceArgumentMiningSurvey2020;@gorurCanLargeLanguage2024;@wuKnowCompDialAM2024Finetuning2024;@zhengKNOWCOMPPOKEMONTeam2024;@egerNeuralEndtoEndLearning2017;@haddadanYesWeCan2019]. Recently, however, more research has been conducted into how these models perform across different domains [@ruiz-dolzTransformerBasedModelsAutomatic2021;@stabCrosstopicArgumentMining2018;@al-khatibCrossDomainMiningArgumentative2016], this generally involves training the model on one domain and then evaluating its performance across several others. A good example of this is the ARIES benchmark [@gemechuARIESGeneralBenchmark2024], which provides results for various different approaches to the ARI task across popular ARI datasets. Another notable contribution is Ruiz-Dolz *et al.* (2021) [@ruiz-dolzTransformerBasedModelsAutomatic2021] which compares the cross-domain performance of the most popular pre-trained transformer models (e.g. BERT [@devlinBERTPretrainingDeep2019b] and RoBERTa [@liuRoBERTaRobustlyOptimized2019a]) showing that the RoBERTa models tend to perform better both in-domain and cross-domain. In order to perform a cross-domain evaluation, it is useful to understand how the data can be constructed and what influence that has.

Ruiz-Dolz *et al.* (2025) [@ruiz-dolzLookingUnseenEffective2025] proposes techniques to answer the question: How do we sample unrelated arguments? If all possible examples of unrelated samples are used it constitutes an overwhelming proportion of the dataset (98-100%) which would be detrimental to model performance in the real world. To achieve this, they propose the following methods:

- **Undersampling** creates a more balanced class distribution by randomly choosing unrelated propositions from the set of all possible combinations.
- **Long Context Sampling** where unrelated propositions are chosen such that they are 'far apart' in the discourse. Ruiz-Dolz *et al.* define this as being from different argument maps.
- **Short Context Sampling** where unrelated propositons are chosen such that they are 'close together' in the discourse. Ruiz-Dolz *et al.* define this as being from the same argumet map.
- **Semantic Similarity Sampling** where unrelated propositions are chosen such that they are semantically similar.

They show that Short Context Sampling is the most challenging method when looking in-domain, however, the model is better able to generalise across different domains than the other methods and is a more realistic task.

Next, the applicability of these techniques is extended to multiple modalities. AM has been performed using both Vision-Language systems [@liuImageArgMultimodalTweet2022;@zongTILFAUnifiedFramework2023;@liuOverviewImageArg2023First2023] and perhaps the more obvious Audio-Language systems [@manciniMultimodalArgumentMining2022;@manciniMAMKitComprehensiveMultimodal2024;@ruiz-dolzVivesDebateSpeechCorpusSpoken2023]. Making use of acoustic features has been shown to improve performance across both ASD and ARI tasks [@ruiz-dolzVivesDebateSpeechCorpusSpoken2023;@mestreMArgMultimodalArgument2021] but there has not been any research into the applicability of Audio-Language systems in cross-domain contexts.

Mancini *et al.* [@manciniMAMKitComprehensiveMultimodal2024] created a comprehensive toolkit for argument mining research. They include both datasets and models that can be used for the creation and evaluation of audio-language argument mining systems, across different tasks, including ASD, ACC and ARI. Therefore, MAMKit provides a very useful benchmark for the development of audio-language AM techniques.
---
bibliography: [../../Cross-Domain AM.bib]
---

# Limitations {#sec:limitations}

**Models** Throughout the project there is a distinct lack of comparison between different encoder models both from a text and an audio perspective. While it is likely that the general overarching conclusions would still hold for other encoders it is very possible that the more specific conclusions could be encoder-specific. This leads into the fact that only the base versions of RoBERTa and Wav2Vec2 are used, it has been shown that using the large models has increased performance both in-domain and cross-domain for text only tasks [@ruiz-dolzTransformerBasedModelsAutomatic2021].

**Forced Alignment** Through the use of forced alignment technologies it is very possible for an amount of accuracy to be lost, especially when considering the poor environments in which such a system will have to perform in the real world (i.e. with lots of crosstalk, a wide range of accents and regional dialects etc.). While the data presented here has been proven to be correct to a certain degree, it is not possible to completely guarantee correctness.

**Statistical Significance** In this project only a single run was performed for each reported result. Therefore it is impossible to tell whether any comparison is statistically significant, however, given the size of the comparisons made it is very likely that they still hold.

# Conclusions

When considering goal (i) as stated in Section @sec:introduction, as a result of this project, two major argumentative corpora have been extended with acoustic features into multimodal datasets for use in many areas of argumentative research. This includes the large QT30 corpus, of which a multimodal subset (QT30-MM) is presented, and the much smaller, cross-domain Moral Maze corpus. QT30-MM is the largest dataset for any AM task currently available.

Goal (ii) of this project was to use the created datasets to conduct an evaluation of different multimodal techniques in a cross-domain setting, comparing early and late sequence fusion and various different late multimodal fusion techniques. Through this evaluation it was found that the addition of acoustic features does not improve the performance of argument relation identification systems in-domain and does not improve the models' ability to generalise across multiple domains.

Although acoustic features were not useful for ARI, it was discovered that the sequence fusion technique chosen during model creation is vitally important for the performance of the model. By combining sequence data early in the process (before encoding) the model is able to make cross-sequence dependencies and therefore significantly improves performance on ARI.

Counterintuitively, all models struggle to distinguish the attack/conflict relations, it is unclear why this occurs however, since data augmentation techniques had no effect it appears unlikely that it is due to the dataset imbalance present. Generally the areas with which the models struggle do not seem to match human intuition, nor how human annotators struggle with the task. When considering the cross-domain results on the 4-class problem, while the models still struggle distinguishing conflict relations, they struggle even more with rephrases, often incorrectly predicting conflicts as rephrases, or predicting rephrases as inferences.

Through the discussion of techniques used to sample unrelated node pairs they have little difference in-domain indicating that the models are able to gain the same knowledge and understanding regardless of the sampling method used, it is simply the difficulty of the evaluation which changes. This supports previous conclusions that the sampling method is an important consideration when creating and evaluating ARI systems [@ruiz-dolzLookingUnseenEffective2025].

The results presented also compare different multimodal fusion strageties showing that simple concatenation, an elementwise product and a crossmodal attention mechanism all perform well. The performance of the crossmodal attention mechanism does vary depending on which modality is used for queries (and by extension keys and values), it was found that the mechanism performs best when text is used to generate keys and values while audio is used to generate the queries.

# Future Work

To provide a more holistic view of this topic in the future, it would be useful to further understand the importance of different encoders and with that whether the varying pre-training strategies have an influence on the ability of the model to generalise across domains. Another possible extension would be to consider data processing and classification methods which preserve inference and conflict structures as discussed in Section @sec:arg-data.

With the recent boom in LLMs and especially multimodal LLMs they provide an opportunity in terms of their potential ability to generalise across domains with minimal amounts of learning (e.g. using in-context learning). LLMs have already shown promise in argument mining tasks so it seems an appropriate next step to discover how they fair in a cross-domain environment [@gorurCanLargeLanguage2024].

# Discussion

With the recent increases in the availability of AI technologies the considerations around the uses of such systems has come under scrutiny. Much of this scrutiny relates to how these intelligent systems can be created and used ethically. Due to the power of AI systems they are an incredibly useful tool across many fields, heavily reducing the workload and increasing the productivity of many people, on the other hand, however, because of this power even relatively incompetent, malevolent actors have the ability to cause damage. Before the introduction and increase in the availability of LLMs, malevalent actors had to be quite competent in order to cause damage, that has however changed because of the ease of use of such systems.

It is often possible to cause damage simply through laziness. This has been seen in the author's previous work in education, where overrelience on AI systems and assistants can cause a significant decrease in accuracy, in systems where the use of 'human-in-the-loop' type systems are used with the goal to increase efficiency while still maintaining accuracy and minimising mistakes. Such a system only works if a. the tools are used correctly and b. the human does not become complacent and therefore accuracy falls in relatively critical situations. This is also a discussion worth having in AM specifically with the release of interactive assistants for the task [@lenzArgueMapperAssistantInteractive2025]. With the introduction of openly-useable LLMs such as Open-AI's Chat-GPT models, many people are using these systems as 'life-assistants' and thus this overreliance does not have bounds in any specific field but rather life as a whole.

Another consideration is the creation and training of such systems, specifically around both the data and the power consumption. Primarily this comes down to how the training datasets are sourced, created and annotated (where annotation is required). The details of the annotation of the data used in this project can be found in the individual datasets' papers, however, it was ensured that all data from those datasets was annotated and sourced ethically. The sourcing of data is also an important consideration, especially surrounding the license under which the original data is released. This generally becomes a problem when using data for commercial purposes (i.e. creating commercially available models). Throughout this project it was ensured that all training data has been sourced within the terms of any licenses.

# Acknowledgements {-}

# References {-}

<div id="refs"></div>
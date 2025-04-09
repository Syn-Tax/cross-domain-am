---
title: A Cross-Domain Evaluation of Multimodal Argument Relation Identification
author: Oscar Morris
student: 2497790
supervisor: Dr. R. Ruiz-Dolz
date: April 2025

bibliography: [../Cross-Domain AM.bib]
numbersections: true
codeBlockCaptions: true
cref: false
header-includes: |
    \usepackage{graphicx}
    \graphicspath{ {./assets/} }

    \usepackage{fvextra}
    \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}

    \usepackage{longtable}
    \usepackage{multirow}

abstract: Recent advances in argument relation identification have begun to look beyond the domain in which they were trained such that these systems can robustly deal with the wide variety of domains they would see in a practical application. In many domains, the data stems from a dialogue (e.g. political debates), the additional paralinguistic features present in the audio data has, until recently, gone unexplored. Work exploring the addition of acoustic data has previously been hampered by a lack of available corpora. In this project, the largest multimodal argument mining dataset currently available is presented, along with a cross-domain dataset. These datasets are then used to evaluate nine different multimodal techniques in both an in-domain and a cross-domain environment. It was found that while the addition of acoustic features does not provide a significant improvement over text only solutions, the process used to combine data from each sequence in the pair is vitally important to the performance of the model.
---
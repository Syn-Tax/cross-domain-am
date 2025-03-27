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

abstract: Recent advances in argument relation identification have begun to look beyond the domain in which they were trained. It comes as no surprise that models which are able to generalise well into many different domains are much more useful in real-world contexts. Such work has previously been hampered by a lack of available data. In this project the largest multimodal argument mining dataset currently available is presented, along with a cross-domain dataset. These datasets are then used to evaluate different multimodal techniques in a cross-domain environment. It was found that while the addition of acoustic features does not provide a significant improvement over text only solutions, it was found that the process used to combine data from each sequence in the pair is vitally important to the performance of the model.
---
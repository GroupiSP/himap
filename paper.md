---
title: 'HiMAP:  Hidden Markov for Advanced Prognostics'
tags:
  -  Python
  -  hidden markov models
  -  stochastic models
  -  prognostics
  -  remaining useful life
authors:
  -  name: Thanos Kontogiannis
    orcid: 0009-0001-6720-1267
    equal-contrib: true
    affiliation: 1
  -  name: Mariana Salinas-Camus
	 orcid: 0009-0008-0315-8686
    equal-contrib: true 
    affiliation: 1
  -  name: Nick Eleftheroglou
    corresponding: true 
    affiliation: 1
affiliations:
  -  name: Intelligent Sustainable Prognostics Group, Aerospace Structures and Materials Department, Faculty of Aerospace Engineering, Delft University of Technology, Kluyverweg 1, 2629HS Delft, the Netherlands
   index: 1
date: 06 December 2024
bibliography: paper.bib

# Summary
Prognostics, the science of predicting systems' future health, performance, and remaining useful life (RUL), requires tools to model complex and often hidden degradation processes. Hidden Markov Models (HMMs) and Hidden Semi-Markov Models (HSMMs) excel in this domain by providing an unsupervised stochastic framework capable of uncovering the underlying states of a system, without requiring labeled data. These models inherently capture the probabilistic nature of degradation through their ability to represent transitions between hidden states over time. This makes them particularly powerful for analyzing dynamic, noisy time-series data and estimating system health even when the degradation patterns are not explicitly known in advance.

# Statement of need
This Python repository comprehensively implements Hidden Markov Models (HMMs) and Hidden Semi-Markov Models (HSMMs) tailored for prognostics applications. Each model is encapsulated within a dedicated Python class, offering an intuitive and modular design for easy integration into various workflows. These classes include essential methods such as `decode`, for inferring the most likely sequence of hidden states; `fit`, for parameter learning; and `sample`, for generating synthetic sequences. Beyond these core functionalities, the repository incorporates advanced features to calculate Remaining Useful Life (RUL) directly by using a prognostic measure introduced in [@phmeconference]. The prognostic measure uses the Viterbi-decoded state sequences of the HMMs and HSMMs and provides a pdf of RUL 

The repository also includes metrics critical for evaluating the performance and reliability of prognostic models, such as Root Mean Squared Error (RMSE) for accuracy, coverage for assessing prediction intervals, and Weighted Spread of Uncertainty (WSU) to quantify confidence in predictions weighted by time. To ensure practical usability and reproducibility, the repository provides two well-documented examples. The first example uses the C-MAPSS dataset, a widely recognized benchmark in prognostics. The second example contains synthetic data created from Monte Carlo Sampling to verify the models. This repository serves as a valuable resource for researchers, engineers, and practitioners aiming to implement and evaluate advanced stochastic models for prognostics in diverse domains.

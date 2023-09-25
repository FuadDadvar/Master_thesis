# master thesis
## Background
In the absence of suited experimental data at a large scale, in silico simulation
methods are a powerful alternative to generating synthetic data for evaluating
statistical models and machine learning methods. A primary challenge for
simulation methods is the ability to create synthetic data that truly reflects
real-world data to the extent possible. A failure to reflect the true nature of
real-world data in simulations results in poor generalisability of the evaluation
results of computational methods. Therefore, there is a greater need for benchmarking
the simulation methods themselves to know how well the simulation
methods are able to capture the inherent complexities of real-world data in
terms of both signal and noise.

## Project Description
In this thesis project, I will use high-dimensional data from a biomedical research
domain as a use case to benchmark the simulation methods. Specifically, the
state-of-the-art generative models like variational autoencoders (VAEs), generative
adversarial networks (GANs) and traditional baseline models like principal
component analysis (PCA), gaussian mixture models (GMMs) and other variations
of the aforementioned models will be used for the generation of synthetic
adaptive immune receptor repertoire (AIRR) data. By starting from simulated
data with known levels of ground truth signals and different levels of noise
(which will be referred to as real-world data hereafter), all the aforementioned
simulation methods will be evaluated in their ability to generate synthetic data
that deviates little from the real-world data. The methods will be compared
and contrasted in different scenarios to chart out the behaviour of the simulation
methods.
Notably, the data from the biomedical field is intended as a use case for suited
explorations and understanding of the behaviour of generative ML models. In
addition to biomedical data, suited datasets from other domains (e.g materials
science) will be used to demonstrate the reproducibility/transferability of the
findings of this project. The findings of the project will not only be useful for
the AIRR biomedical field but will shed light on the behaviour of generative
ML models in general and will be applicable across domains.

## Overview
This project revolves around T-Cell Receptor (TCR) beta amino acid sequences and aims to generate synthesized data using generative models like Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs). The TCR sequences are pivotal in understanding immune responses, and synthesizing such sequences can aid in exploring the vast landscape of potential TCR configurations.


### Data Preprocessing
#### One-Hot Encoding
The TCR beta amino acid sequences are one-hot encoded to convert them into a format suitable for training the models. One-hot encoding is performed using a custom Python script, which reads the sequences from a CSV file and converts each amino acid in a sequence into a unique binary vector, which is later converted to a .H5 file.

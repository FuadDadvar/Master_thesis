# master thesis
## Overview
This project revolves around T-Cell Receptor (TCR) beta amino acid sequences and aims to generate synthesized data using generative models like Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs). The TCR sequences are pivotal in understanding immune responses, and synthesizing such sequences can aid in exploring the vast landscape of potential TCR configurations.


### Data Preprocessing
#### One-Hot Encoding
The TCR beta amino acid sequences are one-hot encoded to convert them into a format suitable for training the models. One-hot encoding is performed using a custom Python script, which reads the sequences from a CSV file and converts each amino acid in a sequence into a unique binary vector, which is later converted to a .H5 file.

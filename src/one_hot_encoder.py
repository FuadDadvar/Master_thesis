import pandas as pd
import torch
from typing import List
from loguru import logger 
import string
import h5py

"""
One-Hot Encoding for TCR Beta Amino Acid Sequences using PyTorch

This script reads TCR (T-cell receptor) beta amino acid sequences from a file and 
performs one-hot encoding on them using PyTorch. One-hot encoding is a process by which categorical 
variables (in this case, amino acids) are converted into a binary matrix.

Attributes:
- AMINO_ACIDS (str): A string containing the 20 standard amino acids.

Functions:
- one_hot_encode(sequence: str) -> torch.Tensor:
    Returns a one-hot encoded torch tensor for a given amino acid sequence.
"""

# Define the amino acids
AMINO_ACIDS = string.ascii_uppercase + string.digits + string.punctuation + string.whitespace

df = pd.read_csv('/Users/fuaddadvar/MSc/simairr_output/baseline_repertoires/rep_1.tsv', header=None, names = ['sequence'])
sequences: List[str] = df['sequence'].tolist()
logger.info(f'Read {len(sequences)} sequences from the CSV file')

max_seq_length = max(map(len, sequences))

def one_hot_encode(sequence: str) -> torch.Tensor:
    """
    One-hot encode a TCR beta amino acid sequence using PyTorch.

    Given an amino acid sequence, this function returns its one-hot encoded representation 
    as a torch tensor. Each amino acid is represented as a unique binary vector.

    Parameters:
    - sequence (str): The amino acid sequence to be encoded.

    Returns:
    - torch.Tensor: The one-hot encoded representation of the sequence.
    """
    sequence = sequence.ljust(max_seq_length, '0')  # Here, '0' is the padding character. You can choose another if '0' is already in your sequences.

    encoding = torch.zeros(len(sequence), len(AMINO_ACIDS))
    for i, amino_acids in enumerate(sequence):
        if amino_acids in AMINO_ACIDS:
            position = AMINO_ACIDS.index(amino_acids)
            encoding[i, position] = 1
    return encoding

logger.info("Starting encoding sequences")

BATCH_SIZE = 10000  # Adjust this based on your available memory

num_batches = len(sequences) // BATCH_SIZE + 1
h5_output_path = '/Users/fuaddadvar/MSc/data/encoded_sequences.h5'

for batch_num in range(num_batches):
    start_idx = batch_num * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE
    
    batch_sequences = sequences[start_idx:end_idx]
    encoded_sequences = [one_hot_encode(seq) for seq in batch_sequences]
    
    # Save the 3D array to an HDF5 file
    with h5py.File(h5_output_path, 'a') as f:
        dataset_name = f'encoded_sequences_batch_{batch_num}'
        if dataset_name not in f:
            f.create_dataset(dataset_name, data=torch.stack(encoded_sequences).numpy(), compression="gzip")
            logger.info(f"Saved encoded sequences of batch {batch_num} to {h5_output_path}")
        else:
            logger.warning(f"Dataset {dataset_name} already exists in {h5_output_path}. Skipping this batch.")

logger.success("Done encoding sequences")
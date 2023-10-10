import pandas as pd
import torch
from typing import List
from loguru import logger 
import string
import h5py
import numpy as np
from matplotlib import pyplot as plt

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

ALL_CHARS = 'ACDEFGHIKLMNPQRSTVWY0'  # 20 amino acids + '0'

def center_pad_sequence(sequence: str, target_length: int = 30) -> str:
    """
    Pads a sequence by inserting zeros in the center to achieve the desired target length.
    
    Args:
    - sequence (str): The original sequence to be padded.
    - target_length (int, optional): The desired length of the padded sequence. Default is 30.

    Returns:
    - str: The padded sequence with zeros added in the center.
    """
    missing_length = target_length - len(sequence)
    split_point = len(sequence) // 2
    left_sequence = sequence[:split_point]
    right_sequence = sequence[split_point:]
    return left_sequence + '0' * missing_length + right_sequence

def read_and_pad_sequences(file_path: str) -> List[str]:
    """
    Reads sequences from a given file, pads them using center padding, and returns them as a list.
    
    Args:
    - file_path (str): Path to the file containing the sequences.

    Returns:
    - List[str]: A list of padded sequences.
    """
    df = pd.read_csv(file_path, header=None, sep='\t')
    sequences = df[1].apply(center_pad_sequence).tolist()
    logger.info(f'Read {len(sequences)} sequences from the CSV file')
    return sequences

def one_hot_encode(sequence: str) -> torch.Tensor:
    """
    One-hot encodes a given amino acid sequence into a torch tensor.
    
    The encoding uses the defined set of characters in ALL_CHARS, including amino acids and padding character.

    Args:
    - sequence (str): The amino acid sequence to be encoded.

    Returns:
    - torch.Tensor: The one-hot encoded representation of the sequence.
    """
    encoding = torch.zeros(len(sequence), len(ALL_CHARS))
    for i, amino_acid in enumerate(sequence):
        if amino_acid in ALL_CHARS:
            position = ALL_CHARS.index(amino_acid)
            encoding[i, position] = 1
    return encoding

def batch_encode_and_save(sequences: List[str], h5_output_path: str, batch_size: int = 100000):
    """
    Batch processes sequences, one-hot encodes each sequence, and saves the encoded sequences to an HDF5 file.
    
    The sequences are processed in batches to manage memory usage. Each batch of encoded sequences is saved
    as a dataset in the HDF5 file.

    Args:
    - sequences (List[str]): A list of sequences to be one-hot encoded.
    - h5_output_path (str): Path where the encoded sequences will be saved in HDF5 format.
    - batch_size (int, optional): The number of sequences to process in each batch. Default is 100,000.

    Returns:
    - None
    
    """
    num_batches = len(sequences) // batch_size + 1
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        
        batch_sequences = sequences[start_idx:end_idx]
        encoded_sequences = [one_hot_encode(seq) for seq in batch_sequences]
        
        with h5py.File(h5_output_path, 'a') as f:
            dataset_name = f'encoded_sequences_batch_{batch_num}'
            if dataset_name not in f:
                f.create_dataset(dataset_name, data=torch.stack(encoded_sequences).numpy(), compression="gzip")
                logger.info(f"Saved encoded sequences of batch {batch_num} to {h5_output_path}")
            else:
                logger.warning(f"Dataset {dataset_name} already exists in {h5_output_path}. Skipping this batch.")

def inspect_encoded_sequences(h5_output_path: str):
    """
    Load the one-hot encoded sequences from the HDF5 file and verify the data.
    
    Args:
    - h5_output_path (str): The path to the HDF5 file with the encoded sequences.
    
    Returns:
    None
    """
    with h5py.File(h5_output_path, 'r') as file:
        for dataset_name in file.keys():
            data = np.array(file[dataset_name])
            logger.info(f"Inspecting dataset: {dataset_name}")
            logger.info(f"Shape: {data.shape}")
            logger.info(f"Dimensions: {len(data.shape)}")
            logger.info(f"Min value: {data.min()}")
            logger.info(f"Max value: {data.max()}\n")

            # Checking for discrepancies (values other than 0 and 1, since it's one-hot encoded)
            unique_values = np.unique(data)
            for value in unique_values:
                if value not in [0, 1]:
                    logger.error(f"Unexpected value {value} detected in dataset {dataset_name}")

def plot_heatmap(probabilities: np.ndarray):
    """
    Plots a heatmap for amino acid occurrence probabilities.

    Args:
    - probabilities (np.ndarray): A 2D numpy array with occurrence probabilities.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(probabilities, cmap='Oranges', interpolation='nearest', aspect='auto')

    # Adding numbers inside the cells
    for i in range(probabilities.shape[0]):
        for j in range(probabilities.shape[1]):
            ax.text(j, i, f'{probabilities[i, j]:.1f}', ha='center', va='center', fontsize=6.2, 
                    color='black' if 0 <= probabilities[i, j] < 0.5 else 'white')

    plt.colorbar(cax, label="Probability")

    ax.set_yticks(range(len(ALL_CHARS)))
    ax.set_yticklabels(ALL_CHARS, fontsize=10, fontweight='bold')
    ax.set_xticks(range(30))
    ax.set_xticklabels(range(0, 30), fontsize=10)
    ax.set_ylabel("Amino Acid", fontsize=12, fontweight='bold')
    ax.set_xlabel("Position", fontsize=12, fontweight='bold')
    ax.set_title("Amino Acid Occurrence Probability Heatmap", fontsize=14, fontweight='bold')
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.set_xticks(np.arange(-.5, 30, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ALL_CHARS), 1), minor=True)

    plt.show()


def main():
    file_path = '/Users/fuaddadvar/MSc - test/simairr_output/baseline_repertoires/rep_1.tsv'
    sequences = read_and_pad_sequences(file_path)
    

    occurrences = np.zeros((len(ALL_CHARS), 30))
    for seq in sequences:
        for idx, char in enumerate(seq):
            if char in ALL_CHARS:
                position = ALL_CHARS.index(char)
                occurrences[position, idx] += 1

    #probabilities = occurrences / len(sequences)
    #plot_heatmap(probabilities)
    
    logger.info("Starting encoding sequences")
    h5_output_path = '/Users/fuaddadvar/MSc - test/data/encoded_sequences.h5'
    #batch_encode_and_save(sequences, h5_output_path)
    logger.success("Done encoding sequences")

    logger.info("Starting inspecting sequences")
    inspect_encoded_sequences(h5_output_path)
    logger.success("Done inspecting sequences")

if __name__ == '__main__':
    main()






import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#h5_file_path = '/Users/fuaddadvar/MSc/data/encoded_sequences.h5'
#batch_name = 'encoded_sequences_batch_10'  # replace with the batch you want to visualize

# with h5py.File(h5_file_path, 'r') as f:
#     if batch_name in f:
#         data = f[batch_name][()]  # Read the batch data

# # Sum across the sequences to get the frequency of each amino acid at each position
# frequencies = np.sum(data, axis=0)

# # Normalize the frequencies if needed
# frequencies = frequencies / np.sum(frequencies, axis=-1, keepdims=True)

# # Plot the heatmap
# plt.figure(figsize=(15, 8))
# sns.heatmap(frequencies.T, cmap='viridis', cbar_kws={'label': 'Frequency'})
# plt.xlabel('Position in Sequence')
# plt.ylabel('Amino Acid')
# plt.title(f'Amino Acid Frequencies at Each Position in Sequence for {batch_name}')
# plt.savefig('Amino_batch_10.pdf')
# plt.show()

# h5_output_path = '/Users/fuaddadvar/MSc/data/encoded_sequences.h5'

# with h5py.File(h5_output_path, 'r') as f:
#     for batch_num, dataset_name in enumerate(f.keys()):
#         data = f[dataset_name][()]  # Read the batch data        
        
#         # Printing the shape and dimensions of the entire data batch
#         print(f"Shape of batch {batch_num}: {data.shape}")
#         print(f"Number of dimensions of batch {batch_num}: {data.ndim}")
        
#         # If you want to print the shape and dimensions of individual sequences in the batch, you can do the following:
#         for seq_num, sequence in enumerate(data):
#             print(f"Shape of sequence {seq_num} in batch {batch_num}: {sequence.shape}")
#             print(f"Number of dimensions of sequence {seq_num} in batch {batch_num}: {sequence.ndim}")
            
#         # If you want to limit the number of printed sequences to avoid overwhelming output, you can break after a few sequences
#         if batch_num >= 2:  # Adjust as needed
#             break

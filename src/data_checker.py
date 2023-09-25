import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

h5_output_path = '/Users/fuaddadvar/MSc/data/encoded_sequences.h5'

with h5py.File(h5_output_path, 'r') as f:
    for batch_num, dataset_name in enumerate(f.keys()):
        data = f[dataset_name][()]  # Read the batch data        
        # Check different sequences in a batch
        print(f"First sequence of batch {batch_num}: {data[0]}")
        print(f"Second sequence of batch {batch_num}: {data[1]}")
        
        # Check the last sequence of each batch
        print(f"Last sequence of batch {batch_num}: {data[-1]}")
        
        # Check the sum of one-hot encoded sequences
        sum_of_sequences = np.sum(data, axis=1)
        print(f"Sum of sequences in batch {batch_num}: {sum_of_sequences}")
        
        # Check sequence lengths
        sequence_lengths = np.sum(data, axis=(1, 2))
        print(f"Sequence lengths in batch {batch_num}: {sequence_lengths}")

        # Assuming the second dimension represents the sequence, and the first sequence is at index 0
        first_sequence = data[0]
        
        print(f"First sequence of batch {batch_num}: {first_sequence}")

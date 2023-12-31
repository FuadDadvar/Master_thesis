import h5py
from typing import Tuple
import os

import seaborn as sns
import torch
import torch.nn as nn
from torch import Tensor
from VAE_ import VAE  # Replace with your actual import
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



h5_file_path = '/Users/fuaddadvar/MSc - test/data/encoded_sequences.h5'

# Load all datasets from h5 file and concatenate them
num_sequences = 1000
all_data = []

with h5py.File(h5_file_path, 'r') as f:
    
    count = 0
    for dataset_name in f.keys():
        if count >= num_sequences:
            break
        data = np.array(f[dataset_name])
        count += len(data)
        if count > num_sequences:
            diff = count - num_sequences
            data = data[:-diff]
        all_data.append(data)

all_data = np.concatenate(all_data, axis=0)
all_data = torch.from_numpy(all_data).float()

flattened_data = all_data.reshape(all_data.shape[0],-1)
input_dim = flattened_data.shape[1]
# The rest of the script remains mostly the same as it adapts to the length of all_data.
batch_size = 16
dataset = TensorDataset(flattened_data)


train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

hidden_dim = 128
latent_dim = 64

model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):

    #Training batch
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch_data, = batch
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(batch_data)
        loss = model.vae_loss(recon_batch, batch_data, mu, log_var)  # Corrected 'vae_loss' call
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader.dataset))
    
    #Validation batch.
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch_data, = batch
            recon_batch, mu, log_var = model(batch_data)
            loss = model.vae_loss(recon_batch, batch_data, mu, log_var)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader.dataset))
    print(f'Epoch {epoch}, Train Loss: {train_loss / len(train_loader.dataset)}, Val Loss: {val_loss / len(val_loader.dataset)}')

epochs = list(range(1, num_epochs + 1))  # Creating a list of epoch numbers
directory = 'figs'  # Or '/src' for an absolute path
if not os.path.exists(directory):
    os.makedirs(directory)

sns.set_theme(style="whitegrid")

# Create a DataFrame with your training and validation losses
data = {
    'Epoch': list(range(1, num_epochs + 1)),
    'Training Loss': train_losses,
    'Validation Loss': val_losses
}
df = pd.DataFrame(data)

# Melt DataFrame to have 'Loss' and 'Type' columns
df_melted = df.melt(id_vars=['Epoch'], value_vars=['Training Loss', 'Validation Loss'], 
                    var_name='Type', value_name='Loss')

# Plot the data
g = sns.relplot(
    data=df_melted,
    x="Epoch", y="Loss",
    hue="Type", 
    kind="scatter", 
    height=5, aspect=1.5,
)

# Adjusting the scales and adding grids if you want
g.set(xscale="linear", yscale="log") # use 'linear' if you want linear scale
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
g.despine(left=True, bottom=True)
plt.show()
# for i in range(min(4, len(all_data))):  
#     sequence = all_data[i].numpy()  # Adjusted to use all_data
    
#     plt.figure(figsize=(10, 2))
#     plt.title(f'Sequence {i}')
    
#     if sequence.ndim == 1:  # For 1D sequence
#         sns.lineplot(x=range(len(sequence)), y=sequence)
#     elif sequence.ndim == 2:  # For 2D sequence
#         sns.heatmap(sequence, cmap='viridis')
#     else:
#         print(f"Skipping sequence {i} as it has more than 2 dimensions.")
        
#     plt.savefig(f"{directory}/Sequence_{i}.pdf")
#     plt.show()

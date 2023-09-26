import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """
        Initializes the Variational Autoencoder (VAE) model.
        
        Args:
            input_dim (int): Dimensionality of the input data.
            hidden_dim (int): Dimensionality of the hidden layers.
            latent_dim (int): Dimensionality of the latent space.
        """
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )


    def reparameterize(self, mu: Tensor, log_variance: Tensor) -> Tensor:
        """
        Applies the reparameterization trick: z = mu + std*epsilon where epsilon is sampled from N(0,1).
        
        Args:
            mu (Tensor): The mean of the latent space.
            log_variance (Tensor): The log variance of the latent space.

        Returns:
            Tensor: The sampled latent vector.
        """
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Defines the computation performed at every call.
        
        Args:
            x (Tensor): The input data.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the reconstructed data, mu, and log_var.
        """
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


    def vae_loss(self, recon_x: Tensor, x: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Computes the VAE loss = reconstruction_loss + KL_divergence_loss.
        
        Args:
            recon_x (Tensor): The reconstructed data.
            x (Tensor): The input data.
            mu (Tensor): The mean of the latent space.
            log_var (Tensor): The log variance of the latent space.

        Returns:
            Tensor: The computed VAE loss.
        """
        recon_loss = nn.BCELoss(reduction='sum')(recon_x, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kld_loss


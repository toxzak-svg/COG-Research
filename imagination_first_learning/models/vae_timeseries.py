"""
Improved VAE for time series data.
Fixes issues with the original VAE:
1. Removes Sigmoid activation (time series can be negative after normalization)
2. Adds configurable hidden dimensions
3. Better initialization
4. Option for layer normalization
"""

import torch
from torch import nn


class TimeseriesVAE(nn.Module):
    """VAE designed for time series with proper activations and scaling."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list = None,
        use_layer_norm: bool = False,
    ):
        super(TimeseriesVAE, self).__init__()
        
        if hidden_dims is None:
            # Default architecture scales with input dimension
            hidden_dims = [
                min(512, input_dim // 2),
                min(256, input_dim // 4),
            ]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        
        # Final layer - NO activation for time series
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Better initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(
    recon_x,
    x,
    mu,
    logvar,
    kl_weight: float = 1.0,
    normalize_by_dim: bool = True,
):
    """
    Compute VAE loss with proper scaling.
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Latent mean
        logvar: Latent log variance
        kl_weight: Weight for KL divergence term
        normalize_by_dim: If True, normalize losses by input dimension
    """
    batch_size = x.shape[0]
    input_dim = x.shape[1]
    
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Normalize by batch size
    recon_loss = recon_loss / batch_size
    kl_loss = kl_loss / batch_size
    
    # Optionally normalize by input dimension for better scaling
    if normalize_by_dim:
        recon_loss = recon_loss / input_dim
    
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss

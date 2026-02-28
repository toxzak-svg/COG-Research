"""
Image-Based VAE for Website Aesthetics

Implements a CNN-based VAE that learns from webpage screenshots rather than
structural features. Uses the Beta-TC-VAE loss for disentanglement.

Architecture:
- Encoder: CNN -> flatten -> latent (mu, logvar)
- Decoder: latent -> unflatten -> TransposeCNN -> image
- Outcome head: latent -> aesthetic score prediction

Input: RGB images (C, H, W)
Output: Reconstructed images + aesthetic score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class ImageVAEEncoder(nn.Module):
    """CNN encoder for webpage screenshots"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 64, image_size: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Progressive downsampling: 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv_blocks = nn.Sequential(
            # Block 1: 256 -> 128
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Block 2: 128 -> 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Block 3: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Block 4: 32 -> 16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Block 5: 16 -> 8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Calculate flattened size: 512 * 8 * 8 = 32768
        self.feature_size = 512 * (image_size // 32) * (image_size // 32)
        
        # Latent projection
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_size, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            mu: Latent mean (B, latent_dim)
            logvar: Latent log-variance (B, latent_dim)
        """
        # CNN feature extraction
        features = self.conv_blocks(x)  # (B, 512, 8, 8)
        features = features.view(features.size(0), -1)  # (B, 32768)
        
        # Latent parameters
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        return mu, logvar


class ImageVAEDecoder(nn.Module):
    """Transposed CNN decoder for webpage screenshots"""
    
    def __init__(self, latent_dim: int = 64, output_channels: int = 3, image_size: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Initial spatial size after unprojection
        self.init_size = image_size // 32  # 8 for 256x256
        self.feature_channels = 512
        
        # Unproject latent to feature map
        self.fc = nn.Linear(latent_dim, self.feature_channels * self.init_size * self.init_size)
        
        # Progressive upsampling: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.deconv_blocks = nn.Sequential(
            # Block 1: 8 -> 16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Block 2: 16 -> 32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block 3: 32 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 4: 64 -> 128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Block 5: 128 -> 256
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # 3 x 256 x 256
            nn.Sigmoid(),  # Output range [0, 1]
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vectors (B, latent_dim)
            
        Returns:
            Reconstructed images (B, C, H, W)
        """
        # Unproject and reshape
        x = self.fc(z)
        x = x.view(x.size(0), self.feature_channels, self.init_size, self.init_size)
        
        # Transposed convolutions
        x = self.deconv_blocks(x)
        
        return x


class ImageWebpageVAE(nn.Module):
    """Complete image-based VAE for webpage aesthetics"""
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 64,
        image_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder and decoder
        self.encoder = ImageVAEEncoder(input_channels, latent_dim, image_size)
        self.decoder = ImageVAEDecoder(latent_dim, input_channels, image_size)
        
        # Outcome prediction head (aesthetic score)
        self.outcome_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Single aesthetic score
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images to latent parameters"""
        return self.encoder(x)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to images"""
        return self.decoder(z)
    
    def predict_outcome(self, z: torch.Tensor) -> torch.Tensor:
        """Predict aesthetic score from latent"""
        return self.outcome_head(z)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input images (B, C, H, W)
            return_latent: Whether to return latent codes
            
        Returns:
            Dictionary with:
                - recon: Reconstructed images
                - mu: Latent means
                - logvar: Latent log-variances
                - z: Sampled latent codes (if return_latent=True)
                - outcome: Predicted aesthetic scores
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decode(z)
        
        # Predict outcome
        outcome = self.predict_outcome(z)
        
        result = {
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
            'outcome': outcome,
        }
        
        if return_latent:
            result['z'] = z
            
        return result
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from prior N(0, I)"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Linear interpolation between two images in latent space"""
        # Encode both images
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        
        # Interpolate in latent space
        alphas = torch.linspace(0, 1, num_steps, device=x1.device)
        z_interp = torch.stack([
            (1 - alpha) * mu1 + alpha * mu2
            for alpha in alphas
        ])
        
        # Decode interpolated latents
        return self.decode(z_interp)


def test_image_vae():
    """Test the image VAE architecture"""
    print("Testing Image VAE...")
    
    # Create model
    model = ImageWebpageVAE(
        input_channels=3,
        latent_dim=64,
        image_size=256,
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)
    
    print(f"Input shape: {x.shape}")
    
    output = model(x, return_latent=True)
    
    print(f"Reconstruction shape: {output['recon'].shape}")
    print(f"Latent mu shape: {output['mu'].shape}")
    print(f"Latent logvar shape: {output['logvar'].shape}")
    print(f"Latent z shape: {output['z'].shape}")
    print(f"Outcome shape: {output['outcome'].shape}")
    
    # Test sampling
    samples = model.sample(num_samples=4)
    print(f"Sample shape: {samples.shape}")
    
    # Test interpolation
    interp = model.interpolate(x[0:1], x[1:2], num_steps=10)
    print(f"Interpolation shape: {interp.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✓ Image VAE test passed!")


if __name__ == '__main__':
    test_image_vae()

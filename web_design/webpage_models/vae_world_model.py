"""
VAE World-Model for Webpage Design

Variational Autoencoder that learns to:
1. Compress WebpageState (190D) → Semantic latent space (32D)
2. Reconstruct state from latent representation
3. Predict business outcomes (conversion rate, bounce rate, etc.)

Based on VAE_WORLD_MODEL_DESIGN.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import numpy as np


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, 1)
    
    Args:
        mu: (batch_size, latent_dim) - Latent mean
        logvar: (batch_size, latent_dim) - Log variance
    
    Returns:
        z: (batch_size, latent_dim) - Sampled latent vector
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class WebpageVAEEncoder(nn.Module):
    """
    VAE Encoder: WebpageState (190D) → Latent distribution (32D)
    
    Architecture:
        Input (190D)
          ↓
        Linear(190 → 128) + BatchNorm + ReLU + Dropout(0.1)
          ↓
        Linear(128 → 96) + BatchNorm + ReLU + Dropout(0.1)
          ↓
        Linear(96 → 64) + BatchNorm + ReLU + Dropout(0.1)
          ↓
        ┌─────────────────┬─────────────────┐
        ↓                 ↓
    Linear(64 → 32)   Linear(64 → 32)
    μ (mean)          log_σ² (variance)
    """
    
    def __init__(
        self,
        input_dim: int = 190,
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 96, 64]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output heads for μ and log_σ²
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, 190) - Vectorized webpage state
        
        Returns:
            mu: (batch_size, 32) - Latent mean
            logvar: (batch_size, 32) - Latent log-variance
        """
        # Encode
        h = self.encoder(x)
        
        # Project to latent distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class WebpageVAEDecoder(nn.Module):
    """
    VAE Decoder: Latent (32D) → Reconstructed state (190D) + Outcomes (10D)
    
    Architecture:
        Latent z (32D)
          ↓
        Linear(32 → 64) + BatchNorm + ReLU
          ↓
        Linear(64 → 96) + BatchNorm + ReLU
          ↓
        Linear(96 → 128) + BatchNorm + ReLU
          ↓
        ┌────────────────────────────┬──────────────────────┐
        ↓                            ↓
    Linear(128 → 190)           Linear(128 → 10)
    Reconstructed State         Outcome Predictions
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        output_dim: int = 190,
        outcome_dim: int = 10,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 96, 128]
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.outcome_dim = outcome_dim
        self.hidden_dims = hidden_dims
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Output heads
        self.fc_recon = nn.Linear(hidden_dims[-1], output_dim)
        self.fc_outcomes = nn.Linear(hidden_dims[-1], outcome_dim)
        
        # Feature-specific output layers for mixed activations
        # We'll apply activations in forward pass based on feature type
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch_size, 32) - Latent representation
        
        Returns:
            x_recon: (batch_size, 190) - Reconstructed state
            outcomes: (batch_size, 10) - Predicted outcomes
                [conversion_rate, bounce_rate, avg_time_on_page,
                 scroll_depth, cta_click_rate, form_start_rate,
                 form_completion_rate, return_visitor_rate, 
                 exit_rate, pages_per_session]
        """
        # Decode
        h = self.decoder(z)
        
        # Reconstruct state (apply sigmoid for normalized features)
        # Note: For simplicity, using sigmoid for all reconstruction features
        # In production, would split by feature type (continuous, count, categorical)
        x_recon = torch.sigmoid(self.fc_recon(h))
        
        # Predict outcomes (all in [0, 1] range)
        outcomes = torch.sigmoid(self.fc_outcomes(h))
        
        return x_recon, outcomes


class WebpageVAE(nn.Module):
    """
    Complete VAE World-Model for webpage design
    
    Combines encoder and decoder with training/inference utilities.
    """
    
    def __init__(
        self,
        input_dim: int = 190,
        latent_dim: int = 32,
        outcome_dim: int = 10,
        hidden_dims_encoder: List[int] = None,
        hidden_dims_decoder: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if hidden_dims_encoder is None:
            hidden_dims_encoder = [128, 96, 64]
        if hidden_dims_decoder is None:
            hidden_dims_decoder = [64, 96, 128]
        
        self.encoder = WebpageVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims_encoder,
            dropout=dropout,
        )
        
        self.decoder = WebpageVAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            outcome_dim=outcome_dim,
            hidden_dims=hidden_dims_decoder,
            dropout=dropout,
        )
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.outcome_dim = outcome_dim
    
    def forward(
        self, 
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → sample → decode
        
        Args:
            x: (batch_size, 190) - Input state vector
            deterministic: If True, use mean (no sampling) for evaluation
        
        Returns:
            x_recon: (batch_size, 190) - Reconstructed state
            outcomes: (batch_size, 10) - Predicted outcomes
            mu: (batch_size, 32) - Latent mean
            logvar: (batch_size, 32) - Latent log-variance
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample latent (or use mean if deterministic)
        if deterministic:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        
        # Decode
        x_recon, outcomes = self.decoder(z)
        
        return x_recon, outcomes, mu, logvar
    
    def encode(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Encode state to latent representation
        
        Args:
            x: (batch_size, 190) - Input state
            deterministic: If True, return mean; else sample
        
        Returns:
            z: (batch_size, 32) - Latent representation
        """
        mu, logvar = self.encoder(x)
        
        if deterministic:
            return mu
        else:
            return reparameterize(mu, logvar)
    
    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent to state + outcomes
        
        Args:
            z: (batch_size, 32) - Latent representation
        
        Returns:
            x_recon: (batch_size, 190) - Reconstructed state
            outcomes: (batch_size, 10) - Predicted outcomes
        """
        return self.decoder(z)
    
    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct state (deterministic - no sampling)
        
        Args:
            x: (batch_size, 190) - Input state
        
        Returns:
            x_recon: (batch_size, 190) - Reconstructed state
            outcomes: (batch_size, 10) - Predicted outcomes
        """
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            z = self.encode(x, deterministic=True)
            result = self.decode(z)
        
        if was_training:
            self.train()
        
        return result
    
    def interpolate(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor, 
        num_steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between two states in latent space
        
        Args:
            x1: (1, 190) - Start state
            x2: (1, 190) - End state
            num_steps: Number of interpolation steps
        
        Returns:
            x_interp: (num_steps, 190) - Interpolated states
            outcomes_interp: (num_steps, 10) - Interpolated outcomes
        """
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # Encode both states
            z1 = self.encode(x1, deterministic=True)
            z2 = self.encode(x2, deterministic=True)
            
            # Linear interpolation
            alphas = torch.linspace(0, 1, num_steps, device=z1.device).unsqueeze(1)
            z_interp = (1 - alphas) * z1 + alphas * z2
            
            # Decode interpolated latents
            x_interp, outcomes_interp = self.decode(z_interp)
        
        if was_training:
            self.train()
        
        return x_interp, outcomes_interp
    
    def sample(self, num_samples: int = 1, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample random states from prior p(z) = N(0, I)
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            x_samples: (num_samples, 190) - Generated states
            outcomes_samples: (num_samples, 10) - Predicted outcomes
        """
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            result = self.decode(z)
        
        if was_training:
            self.train()
        
        return result


class VAELoss(nn.Module):
    """
    Combined VAE loss function
    
    Loss = β_recon * L_reconstruction 
         + β_kl * L_kl_divergence 
         + β_outcome * L_outcome_prediction
         + β_consistency * L_consistency
    """
    
    def __init__(
        self,
        beta_recon: float = 1.0,
        beta_kl: float = 0.5,
        beta_outcome: float = 1.0,
        beta_consistency: float = 0.2,
        feature_masks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Args:
            beta_recon: Weight for reconstruction loss
            beta_kl: Weight for KL divergence (β-VAE parameter)
            beta_outcome: Weight for outcome prediction loss
            beta_consistency: Weight for consistency loss
            feature_masks: Dictionary defining feature groups for weighted reconstruction
        """
        super().__init__()
        
        self.beta_recon = beta_recon
        self.beta_kl = beta_kl
        self.beta_outcome = beta_outcome
        self.beta_consistency = beta_consistency
        # Only use default masks if feature_masks is not explicitly provided
        # For non-structural inputs (e.g., images), feature_masks should be None
        if feature_masks is False:
            # Explicitly disable feature masks
            self.feature_masks = None
        elif feature_masks is None:
            # Default behavior: use structural webpage masks
            self.feature_masks = self._default_masks()
        else:
            # User provided custom masks
            self.feature_masks = feature_masks
    
    def _default_masks(self) -> Dict[str, List[int]]:
        """
        Default feature masks for WebpageState (190D)
        
        Based on vectorization in dataset_builder.py:
        - Visual/Design: [0:80]
        - Content: [80:110]
        - Performance: [110:130]
        - Outcomes: [130:140]
        - Cognitive Load: [140:150]
        - Context: [150:170]
        - Portfolio/Metadata: [170:190]
        """
        return {
            'visual': list(range(0, 80)),
            'content': list(range(80, 110)),
            'performance': list(range(110, 130)),
            'outcomes': list(range(130, 140)),
            'cognitive': list(range(140, 150)),
            'context': list(range(150, 170)),
            'portfolio': list(range(170, 190)),
        }
    
    def reconstruction_loss(
        self, 
        x_recon: torch.Tensor, 
        x_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted reconstruction loss by feature group
        
        Args:
            x_recon: (batch_size, 190) - Reconstructed state OR (batch_size, C, H, W) for images
            x_true: (batch_size, 190) - True state OR (batch_size, C, H, W) for images
        
        Returns:
            loss: Scalar tensor
        """
        # If no feature masks provided, use simple MSE (for images or generic inputs)
        if self.feature_masks is None:
            return F.mse_loss(x_recon, x_true)
        
        loss = 0.0
        
        # Visual/Design features (high weight)
        visual_mask = self.feature_masks['visual']
        loss += 2.0 * F.mse_loss(x_recon[:, visual_mask], x_true[:, visual_mask])
        
        # Content features (high weight)
        content_mask = self.feature_masks['content']
        loss += 2.0 * F.mse_loss(x_recon[:, content_mask], x_true[:, content_mask])
        
        # Performance features (medium weight)
        perf_mask = self.feature_masks['performance']
        loss += 1.0 * F.mse_loss(x_recon[:, perf_mask], x_true[:, perf_mask])
        
        # Outcome features (low weight - predicted separately)
        outcome_mask = self.feature_masks['outcomes']
        loss += 0.5 * F.mse_loss(x_recon[:, outcome_mask], x_true[:, outcome_mask])
        
        # Cognitive load (medium weight)
        cognitive_mask = self.feature_masks['cognitive']
        loss += 1.0 * F.mse_loss(x_recon[:, cognitive_mask], x_true[:, cognitive_mask])
        
        # Context (low weight)
        context_mask = self.feature_masks['context']
        loss += 0.5 * F.mse_loss(x_recon[:, context_mask], x_true[:, context_mask])
        
        # Portfolio (medium weight)
        portfolio_mask = self.feature_masks['portfolio']
        loss += 1.0 * F.mse_loss(x_recon[:, portfolio_mask], x_true[:, portfolio_mask])
        
        # Normalize by number of groups
        return loss / 7
    
    def kl_divergence_loss(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        
        = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        
        Args:
            mu: (batch_size, latent_dim) - Latent mean
            logvar: (batch_size, latent_dim) - Latent log-variance
        
        Returns:
            loss: Scalar tensor
        """
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    def outcome_prediction_loss(
        self,
        outcomes_pred: torch.Tensor,
        outcomes_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Outcome prediction loss with MSE + ranking constraint
        
        Args:
            outcomes_pred: (batch_size, outcome_dim) - Predicted outcomes
            outcomes_true: (batch_size, outcome_dim) - True outcomes
        
        Returns:
            loss: Scalar tensor
        """
        # MSE loss
        mse_loss = F.mse_loss(outcomes_pred, outcomes_true)
        
        # Ranking loss: Preserve relative ordering by conversion rate
        # Higher conversion samples should have higher predictions
        ranking_loss = 0.0
        batch_size = outcomes_true.size(0)
        
        if batch_size > 1:
            # Focus on conversion rate (first outcome dimension)
            conv_true = outcomes_true[:, 0]
            conv_pred = outcomes_pred[:, 0]
            
            # Compute pairwise ranking violations
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    if conv_true[i] > conv_true[j]:
                        # i should predict higher than j
                        violation = F.relu(conv_pred[j] - conv_pred[i])
                        ranking_loss += violation
            
            # Normalize by number of pairs
            num_pairs = batch_size * (batch_size - 1) / 2
            ranking_loss = ranking_loss / num_pairs
        
        return mse_loss + 0.1 * ranking_loss
    
    def consistency_loss(
        self,
        model: nn.Module,
        z: torch.Tensor,
        x_recon: torch.Tensor,
        noise_scale: float = 0.1,
    ) -> torch.Tensor:
        """
        Consistency loss: Nearby latents should produce nearby reconstructions
        
        Args:
            model: VAE decoder
            z: (batch_size, latent_dim) - Latent samples
            x_recon: (batch_size, output_dim) - Reconstructed states
            noise_scale: Standard deviation of Gaussian noise
        
        Returns:
            loss: Scalar tensor
        """
        # Add small noise to latent
        z_noise = z + torch.randn_like(z) * noise_scale
        
        # Decode noisy latent
        x_recon_noise, _ = model.decoder(z_noise)
        
        # Nearby latents should have nearby reconstructions
        smoothness_loss = F.mse_loss(x_recon, x_recon_noise)
        
        return smoothness_loss
    
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        outcomes_pred: torch.Tensor,
        outcomes_true: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and component losses
        
        Args:
            model: VAE model (for consistency loss)
            x: (batch_size, input_dim) - Input state
            x_recon: (batch_size, input_dim) - Reconstructed state
            outcomes_pred: (batch_size, outcome_dim) - Predicted outcomes
            outcomes_true: (batch_size, outcome_dim) - True outcomes
            mu: (batch_size, latent_dim) - Latent mean
            logvar: (batch_size, latent_dim) - Latent log-variance
            z: (batch_size, latent_dim) - Sampled latent
        
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary of component losses
        """
        # Component losses
        loss_recon = self.reconstruction_loss(x_recon, x)
        loss_kl = self.kl_divergence_loss(mu, logvar)
        loss_outcome = self.outcome_prediction_loss(outcomes_pred, outcomes_true)
        loss_consist = self.consistency_loss(model, z, x_recon)
        
        # Total loss
        total_loss = (
            self.beta_recon * loss_recon +
            self.beta_kl * loss_kl +
            self.beta_outcome * loss_outcome +
            self.beta_consistency * loss_consist
        )
        
        # Return loss and components for logging
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': loss_recon.item(),
            'kl_divergence': loss_kl.item(),
            'outcome_prediction': loss_outcome.item(),
            'consistency': loss_consist.item(),
        }
        
        return total_loss, loss_dict
    
    def update_beta_kl(self, new_beta: float):
        """Update KL weight (for β-annealing)"""
        self.beta_kl = new_beta


# Utility functions for training

def anneal_beta_kl(epoch: int, max_beta: float = 0.5, anneal_epochs: int = 20) -> float:
    """
    Linear annealing schedule for β_kl
    
    Args:
        epoch: Current epoch (0-indexed)
        max_beta: Maximum β value
        anneal_epochs: Number of epochs to anneal over
    
    Returns:
        beta: Current β value
    """
    return min(max_beta, epoch / anneal_epochs * max_beta)


def get_feature_masks(device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Get feature masks as tensors for efficient indexing
    
    Returns:
        Dictionary mapping feature group names to index tensors
    """
    masks = {
        'visual': torch.arange(0, 80, device=device),
        'content': torch.arange(80, 110, device=device),
        'performance': torch.arange(110, 130, device=device),
        'outcomes': torch.arange(130, 140, device=device),
        'cognitive': torch.arange(140, 150, device=device),
        'context': torch.arange(150, 170, device=device),
        'portfolio': torch.arange(170, 190, device=device),
    }
    return masks


class BetaTCVAELoss(VAELoss):
    """
    Enhanced VAE loss with β-TC-VAE decomposition and contrastive learning
    
    Loss = β_recon * L_recon
         + β_index * I(z;n)          [index-code mutual information]
         + β_tc * TC(z)               [total correlation - prevents collapse]
         + β_dwkl * ∑KL(q(z_j)||p(z_j))  [dimension-wise KL]
         + β_outcome * L_outcome
         + β_contrast * L_contrastive [semantic structure]
    
    Reference: "Isolating Sources of Disentanglement in VAEs" (Chen et al., 2018)
    """
    
    def __init__(
        self,
        beta_recon: float = 1.0,
        beta_index: float = 1.0,
        beta_tc: float = 6.0,  # Higher weight on TC prevents collapse
        beta_dwkl: float = 1.0,
        beta_outcome: float = 1.0,
        beta_contrast: float = 0.5,
        beta_consistency: float = 0.2,
        free_bits: float = 0.5,  # Minimum nats per dimension
        target_kl: float = 160.0,  # Lagrangian target: 5 nats * 32 dims
        feature_masks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Args:
            beta_recon: Weight for reconstruction loss
            beta_index: Weight for index-code MI
            beta_tc: Weight for total correlation (key for preventing collapse)
            beta_dwkl: Weight for dimension-wise KL
            beta_outcome: Weight for outcome prediction
            beta_contrast: Weight for contrastive loss
            beta_consistency: Weight for consistency loss
            free_bits: Minimum KL per dimension (prevents dimension collapse)
            target_kl: Target KL for Lagrangian constraint
            feature_masks: Feature group masks
        """
        super().__init__(
            beta_recon=beta_recon,
            beta_kl=0.0,  # Not used in β-TC-VAE
            beta_outcome=beta_outcome,
            beta_consistency=beta_consistency,
            feature_masks=feature_masks,
        )
        
        self.beta_index = beta_index
        self.beta_tc = beta_tc
        self.beta_dwkl = beta_dwkl
        self.beta_contrast = beta_contrast
        self.free_bits = free_bits
        self.target_kl = target_kl
    
    def log_density_gaussian(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log p(z) for Gaussian distribution
        
        log N(z; mu, sigma^2) = -0.5 * (log(2π) + log(σ²) + (z-μ)²/σ²)
        
        Args:
            z: (batch_size, latent_dim) - Latent samples
            mu: (batch_size, latent_dim) - Mean
            logvar: (batch_size, latent_dim) - Log variance
        
        Returns:
            log_density: (batch_size, latent_dim) - Log density per dimension
        """
        normalization = -0.5 * torch.log(torch.tensor(2.0 * torch.pi))
        inv_var = torch.exp(-logvar)
        log_density = normalization - 0.5 * logvar - 0.5 * ((z - mu) ** 2 * inv_var)
        return log_density
    
    def tc_vae_decomposition(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose KL[q(z)|p(z)] into three terms using minibatch weighted sampling
        
        KL[q(z)||p(z)] = I(z;n) + TC(z) + ∑_j KL[q(z_j)||p(z_j)]
        
        Where:
        - I(z;n): Index-code mutual information (how much does data index matter)
        - TC(z): Total correlation (dependency between latent dimensions)
        - ∑KL: Sum of dimension-wise KL divergences
        
        Args:
            z: (batch_size, latent_dim) - Sampled latents
            mu: (batch_size, latent_dim) - Encoder means
            logvar: (batch_size, latent_dim) - Encoder log-variances
        
        Returns:
            index_code_mi: Scalar - I(z;n)
            total_correlation: Scalar - TC(z)
            dimension_wise_kl: Scalar - ∑KL
        """
        batch_size, latent_dim = z.shape
        
        # Compute log q(z|x) for each sample
        # log q(z_i|x_i) = ∑_j log N(z_ij; mu_ij, sigma_ij^2)
        log_qz_given_x = self.log_density_gaussian(z, mu, logvar).sum(dim=1)  # (batch_size,)
        
        # Compute log q(z) using minibatch weighted sampling
        # log q(z) ≈ log(1/NM * ∑_i ∑_j q(z_i|x_j))
        # We need to compute q(z_sample_i | x_j) for all pairs
        
        # Expand dimensions for broadcasting
        z_expand = z.unsqueeze(1)  # (batch_size, 1, latent_dim)
        mu_expand = mu.unsqueeze(0)  # (1, batch_size, latent_dim)
        logvar_expand = logvar.unsqueeze(0)  # (1, batch_size, latent_dim)
        
        # Compute log q(z_i|x_j) for all pairs - shape (batch_size, batch_size, latent_dim)
        log_qz_given_x_all = self.log_density_gaussian(z_expand, mu_expand, logvar_expand)
        
        # log q(z) = log(1/N * ∑_j q(z|x_j)) = log(1/N * ∑_j exp(∑_d log q(z_d|x_j)))
        log_qz = torch.logsumexp(
            log_qz_given_x_all.sum(dim=2),  # Sum over dimensions first
            dim=1  # Then logsumexp over batch (data indices)
        ) - torch.log(torch.tensor(batch_size, dtype=torch.float32))
        
        # Compute log ∏_j q(z_j) = ∑_j log q(z_j)
        # log q(z_j) = log(1/N * ∑_i q(z_j|x_i))
        log_qz_prod = torch.logsumexp(
            log_qz_given_x_all,  # (batch_size, batch_size, latent_dim)
            dim=1  # logsumexp over data indices
        ) - torch.log(torch.tensor(batch_size, dtype=torch.float32))
        log_qz_prod = log_qz_prod.sum(dim=1)  # Sum over dimensions (batch_size,)
        
        # Compute log p(z) = ∑_j log N(z_j; 0, 1)
        log_pz = self.log_density_gaussian(
            z,
            torch.zeros_like(z),
            torch.zeros_like(z)
        ).sum(dim=1)  # (batch_size,)
        
        # Decomposition:
        # I(z;n) = E[log q(z|x) - log q(z)]
        index_code_mi = (log_qz_given_x - log_qz).mean()
        
        # TC(z) = E[log q(z) - log ∏_j q(z_j)]
        total_correlation = (log_qz - log_qz_prod).mean()
        
        # ∑KL = E[log ∏_j q(z_j) - log p(z)]
        dimension_wise_kl = (log_qz_prod - log_pz).mean()
        
        return index_code_mi, total_correlation, dimension_wise_kl
    
    def dimension_wise_kl_with_free_bits(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dimension-wise KL with free-bits regularization
        
        Free-bits: Each dimension must have at least 'free_bits' nats of information.
        This prevents individual dimensions from collapsing.
        
        Args:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        
        Returns:
            kl_loss: Scalar - KL with free-bits constraint
        """
        # KL per dimension and per sample
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (batch_size, latent_dim)
        
        # Average over batch
        kl_per_dim_mean = kl_per_dim.mean(dim=0)  # (latent_dim,)
        
        # Apply free-bits: max(KL, free_bits)
        kl_per_dim_free = torch.maximum(
            kl_per_dim_mean,
            torch.tensor(self.free_bits, device=mu.device)
        )
        
        # Sum over dimensions
        return kl_per_dim_free.sum()
    
    def contrastive_loss(
        self,
        z: torch.Tensor,
        num_components: torch.Tensor,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        """
        Contrastive loss for semantic structure
        
        Similar states (same/near component count) should be close in latent space.
        Uses InfoNCE-style contrastive objective.
        
        Args:
            z: (batch_size, latent_dim) - Latent codes
            num_components: (batch_size,) - Number of components per state
            temperature: Temperature parameter for softmax
        
        Returns:
            loss: Scalar - Contrastive loss
        """
        batch_size = z.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)
        
        # Normalize latents for cosine similarity
        z_norm = F.normalize(z, p=2, dim=1)
        
        # Compute pairwise cosine similarities
        similarity_matrix = torch.mm(z_norm, z_norm.t()) / temperature  # (batch_size, batch_size)
        
        # Define semantic similarity based on component count
        # Similar: within ±1 component
        num_components_expanded = num_components.unsqueeze(1)  # (batch_size, 1)
        component_diff = torch.abs(
            num_components_expanded - num_components_expanded.t()
        )  # (batch_size, batch_size)
        
        # Create positive mask (similar states)
        positive_mask = (component_diff <= 1).float()
        # Remove self-similarities
        positive_mask.fill_diagonal_(0)
        
        # Create negative mask (dissimilar states)
        negative_mask = (component_diff > 1).float()
        
        # Check if we have positives and negatives
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        # InfoNCE loss: log(exp(sim(z_i, z_i+)) / sum_j exp(sim(z_i, z_j)))
        # For each anchor, pull positives close and push negatives away
        
        loss = 0.0
        num_anchors = 0
        
        for i in range(batch_size):
            # Get positives for anchor i
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue
            
            # Similarities for anchor i
            anchor_similarity = similarity_matrix[i]  # (batch_size,)
            
            # For each positive
            for pos_idx in pos_indices:
                # Positive similarity
                pos_sim = anchor_similarity[pos_idx]
                
                # Denominator: sum over all except anchor itself
                # Use mask to exclude diagonal
                mask = torch.ones(batch_size, device=z.device)
                mask[i] = 0
                
                # Log-sum-exp for numerical stability
                log_sum_exp = torch.logsumexp(anchor_similarity[mask.bool()], dim=0)
                
                # InfoNCE loss
                loss += -pos_sim + log_sum_exp
                num_anchors += 1
        
        if num_anchors > 0:
            loss = loss / num_anchors
        else:
            loss = torch.tensor(0.0, device=z.device)
        
        return loss
    
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        outcomes_pred: torch.Tensor,
        outcomes_true: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        num_components: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute β-TC-VAE loss with contrastive learning
        
        Args:
            model: VAE model
            x: Input state
            x_recon: Reconstructed state
            outcomes_pred: Predicted outcomes
            outcomes_true: True outcomes
            mu: Latent mean
            logvar: Latent log-variance
            z: Sampled latent
            num_components: Number of components per state (for contrastive loss)
        
        Returns:
            total_loss: Scalar
            loss_dict: Dictionary of component losses
        """
        # Reconstruction loss
        loss_recon = self.reconstruction_loss(x_recon, x)
        
        # β-TC-VAE decomposition
        index_code_mi, total_correlation, dimension_wise_kl = self.tc_vae_decomposition(z, mu, logvar)
        
        # Free-bits regularization (alternative dim-wise KL)
        # dwkl_free = self.dimension_wise_kl_with_free_bits(mu, logvar)
        
        # Outcome prediction loss
        loss_outcome = self.outcome_prediction_loss(outcomes_pred, outcomes_true)
        
        # Consistency loss (only if beta > 0)
        if self.beta_consistency > 0:
            loss_consist = self.consistency_loss(model, z, x_recon)
        else:
            loss_consist = torch.tensor(0.0, device=z.device)
        
        # Contrastive loss (if num_components provided and beta > 0)
        if num_components is not None and self.beta_contrast > 0:
            loss_contrast = self.contrastive_loss(z, num_components)
        else:
            loss_contrast = torch.tensor(0.0, device=z.device)
        
        # Lagrangian constraint on total KL
        total_kl = index_code_mi + total_correlation + dimension_wise_kl
        lagrangian_penalty = (total_kl - self.target_kl).pow(2)
        
        # Total loss
        total_loss = (
            self.beta_recon * loss_recon +
            self.beta_index * index_code_mi +
            self.beta_tc * total_correlation +
            self.beta_dwkl * dimension_wise_kl +
            self.beta_outcome * loss_outcome +
            self.beta_contrast * loss_contrast +
            self.beta_consistency * loss_consist
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': loss_recon.item(),
            'index_code_mi': index_code_mi.item(),
            'total_correlation': total_correlation.item(),
            'dimension_wise_kl': dimension_wise_kl.item(),
            'total_kl': total_kl.item(),
            'outcome_prediction': loss_outcome.item(),
            'contrastive': loss_contrast.item(),
            'consistency': loss_consist.item(),
            'lagrangian_penalty': lagrangian_penalty.item(),
        }
        
        return total_loss, loss_dict
    
    def update_beta_tc(self, new_beta: float):
        """Update total correlation weight (for annealing)"""
        self.beta_tc = new_beta

"""
Flask App World Model
VAE-based model that learns to predict outcomes from schema changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class AppEncoder(nn.Module):
    """
    VAE Encoder: Maps app state to latent representation
    
    Input: App state vector (170D)
    Output: latent mean (16D), latent log variance (16D)
    """
    
    def __init__(self, state_dim: int = 170, latent_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # Encoder network
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Latent projection
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input state (batch, state_dim)
            
        Returns:
            mu: Latent mean (batch, latent_dim)
            logvar: Latent log variance (batch, latent_dim)
        """
        # Encode
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        
        # Project to latent
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class AppDecoder(nn.Module):
    """
    VAE Decoder: Maps latent representation back to app state
    
    Input: Latent vector (16D)
    Output: Reconstructed state (170D)
    """
    
    def __init__(self, latent_dim: int = 16, state_dim: int = 170, hidden_dim: int = 128):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        
        # Decoder network
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector (batch, latent_dim)
            
        Returns:
            Reconstructed state (batch, state_dim)
        """
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        
        # Reconstruct
        x_recon = self.fc_out(h)
        
        return x_recon


class TransitionModel(nn.Module):
    """
    Transition Model: Predicts next state given current state and edit
    
    Input: (latent_state, edit) -> latent_delta
    Output: predicted next latent state
    """
    
    def __init__(self, latent_dim: int = 16, edit_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.edit_dim = edit_dim
        
        # Transition network
        self.fc1 = nn.Linear(latent_dim + edit_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_delta = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, z: torch.Tensor, edit: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Current latent state (batch, latent_dim)
            edit: Edit vector (batch, edit_dim)
            
        Returns:
            Predicted next latent state (batch, latent_dim)
        """
        # Concatenate state and edit
        combined = torch.cat([z, edit], dim=-1)
        
        # Compute delta
        h = F.relu(self.fc1(combined))
        h = F.relu(self.fc2(h))
        delta = self.fc_delta(h)
        
        # Apply delta
        z_next = z + delta
        
        return z_next


class OutcomePredictor(nn.Module):
    """
    Outcome Predictor: Predicts test outcomes from latent state and edit
    
    Input: (latent_state, edit)
    Output: test_pass_probability
    """
    
    def __init__(self, latent_dim: int = 16, edit_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        
        # Outcome network
        self.fc1 = nn.Linear(latent_dim + edit_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, z: torch.Tensor, edit: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Current latent state (batch, latent_dim)
            edit: Edit vector (batch, edit_dim)
            
        Returns:
            Test pass probability (batch, 1)
        """
        # Concatenate
        combined = torch.cat([z, edit], dim=-1)
        
        # Predict
        h = F.relu(self.fc1(combined))
        h = F.relu(self.fc2(h))
        
        # Output: probability
        prob = torch.sigmoid(self.fc_out(h))
        
        return prob


class AppWorldModel(nn.Module):
    """
    Complete World Model for Flask App Evolution
    
    Components:
    - Encoder: state -> latent
    - Decoder: latent -> state
    - Transition: (state, edit) -> next_state
    - Outcome: (state, edit) -> test_pass_probability
    
    Training objective:
    - Reconstruction loss: reconstruct state after transition
    - KL loss: regularize latent space
    - Outcome loss: predict test outcomes
    """
    
    def __init__(
        self,
        state_dim: int = 170,
        latent_dim: int = 16,
        edit_dim: int = 32,
        hidden_dim: int = 128,
        kl_weight: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.edit_dim = edit_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        
        # Components
        self.encoder = AppEncoder(state_dim, latent_dim, hidden_dim)
        self.decoder = AppDecoder(latent_dim, state_dim, hidden_dim)
        self.transition = TransitionModel(latent_dim, edit_dim, hidden_dim // 2)
        self.outcome = OutcomePredictor(latent_dim, edit_dim, hidden_dim // 2)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode state to latent"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to state"""
        return self.decoder(z)
    
    def predict_outcome(
        self,
        state: torch.Tensor,
        edit: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state and test outcome
        
        Args:
            state: Current state (batch, state_dim)
            edit: Edit vector (batch, edit_dim)
            
        Returns:
            z_next: Predicted next latent state (batch, latent_dim)
            state_recon: Reconstructed next state (batch, state_dim)
            outcome_prob: Test pass probability (batch, 1)
        """
        # Encode current state
        mu, logvar = self.encode(state)
        z = self.reparameterize(mu, logvar)
        
        # Predict next state via transition
        z_next = self.transition(z, edit)
        
        # Decode to get reconstructed state
        state_recon = self.decode(z_next)
        
        # Predict outcome
        outcome_prob = self.outcome(z, edit)
        
        return z_next, state_recon, outcome_prob
    
    def forward(
        self,
        state_before: torch.Tensor,
        edit: torch.Tensor,
        state_after: torch.Tensor,
        outcome: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with losses
        
        Args:
            state_before: State before edit (batch, state_dim)
            edit: Edit vector (batch, edit_dim)
            state_after: State after edit (batch, state_dim)
            outcome: Test pass outcome (batch,) - optional, for supervised training
            
        Returns:
            Dictionary of losses and outputs
        """
        # Encode before state
        mu, logvar = self.encode(state_before)
        z = self.reparameterize(mu, logvar)
        
        # Predict next state
        z_next, state_recon, outcome_prob = self.predict_outcome(state_before, edit)
        
        # Compute losses
        # 1. Reconstruction loss
        recon_loss = F.mse_loss(state_recon, state_after, reduction='mean')
        
        # 2. KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 3. Outcome loss (if provided)
        if outcome is not None:
            outcome_target = outcome.unsqueeze(-1)
            outcome_loss = F.binary_cross_entropy(outcome_prob, outcome_target, reduction='mean')
        else:
            outcome_loss = torch.tensor(0.0, device=state_before.device)
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss + outcome_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'outcome_loss': outcome_loss,
            'outcome_prob': outcome_prob,
            'z_next': z_next,
            'state_recon': state_recon,
        }
    
    def predict(
        self,
        state: torch.Tensor,
        edit: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: predict outcome without ground truth
        
        Args:
            state: Current state (batch, state_dim)
            edit: Edit vector (batch, edit_dim)
            
        Returns:
            Dictionary with predictions
        """
        with torch.no_grad():
            z_next, state_recon, outcome_prob = self.predict_outcome(state, edit)
            
            return {
                'predicted_state': state_recon,
                'test_pass_probability': outcome_prob,
                'is_safe': outcome_prob > 0.5,
            }


def create_app_world_model(
    state_dim: int = 170,
    latent_dim: int = 16,
    edit_dim: int = 32,
    hidden_dim: int = 128,
    device: str = 'cpu',
) -> AppWorldModel:
    """Create and initialize the world model"""
    model = AppWorldModel(
        state_dim=state_dim,
        latent_dim=latent_dim,
        edit_dim=edit_dim,
        hidden_dim=hidden_dim,
    )
    model.to(device)
    return model


def test_world_model():
    """Test the world model"""
    print("Testing AppWorldModel...")
    
    # Create model
    model = create_app_world_model(
        state_dim=170,
        latent_dim=16,
        edit_dim=32,
    )
    
    # Create dummy input
    batch_size = 4
    state_before = torch.randn(batch_size, 170)
    edit = torch.randn(batch_size, 32)
    state_after = torch.randn(batch_size, 170)
    outcome = torch.tensor([1.0, 0.0, 1.0, 1.0])
    
    # Forward pass
    result = model(state_before, edit, state_after, outcome)
    
    print(f"Total loss: {result['total_loss'].item():.4f}")
    print(f"Reconstruction loss: {result['recon_loss'].item():.4f}")
    print(f"KL loss: {result['kl_loss'].item():.4f}")
    print(f"Outcome loss: {result['outcome_loss'].item():.4f}")
    print(f"Outcome probs: {result['outcome_prob'].squeeze().tolist()}")
    
    # Test prediction
    pred = model.predict(state_before, edit)
    print(f"\nPrediction test_pass_probability: {pred['test_pass_probability'].squeeze().tolist()}")
    print(f"Prediction is_safe: {pred['is_safe'].squeeze().tolist()}")
    
    print("\n✓ World model works!")


if __name__ == "__main__":
    test_world_model()


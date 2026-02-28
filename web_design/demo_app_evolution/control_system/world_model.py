"""
World-Model: Predict app state transitions

Based on research VAE architecture from imagination_first_learning/models/vae.py
Predicts outcomes of edit sequences.
"""

import torch
import torch.nn as nn
import numpy as np


class WorldModel(nn.Module):
    """
    World-Model predicts next app state given current state and edit.
    
    Architecture mirrors research VAE:
    - Encoder: (current_state, edit) → latent representation
    - Decoder: latent → predicted next state
    - Transition: learns latent dynamics z_next = f(z_current, edit)
    """
    
    def __init__(self, latent_dim=16, edit_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.edit_dim = edit_dim
        
        # Transition model: (z_current, edit) → z_next
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + edit_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        
        # Decoder: z → predicted outcomes
        # Predicts: test pass rate, coverage, auth violations, etc.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),  # Output: [test_pass_rate, coverage, ...]
            nn.Sigmoid(),
        )
    
    def predict_transition(self, z_current, edit_embedding):
        """Predict next latent state given current state and edit."""
        combined = torch.cat([z_current, edit_embedding], dim=0)
        z_next = self.transition(combined)
        return z_next
    
    def decode_outcomes(self, z):
        """Decode latent state to predicted outcomes."""
        outcomes = self.decoder(z)
        return {
            'test_pass_rate': outcomes[0].item(),
            'coverage': outcomes[1].item(),
            'auth_violations': outcomes[2].item(),
            'performance_delta': outcomes[3].item(),
            'schema_consistency': outcomes[4].item(),
            'endpoint_health': outcomes[5].item(),
        }
    
    def predict_sequence_outcome(self, z_start, edit_sequence, state_encoder):
        """
        Predict outcome of applying an edit sequence.
        
        This is the key capability: given a proposed sequence of edits,
        predict the final app state without actually applying them.
        """
        z_current = z_start
        
        for edit in edit_sequence:
            # Encode edit to vector
            edit_embedding = self._encode_edit(edit)
            
            # Predict transition
            z_next = self.predict_transition(z_current, edit_embedding)
            z_current = z_next
        
        # Decode final state to outcomes
        outcomes = self.decode_outcomes(z_current)
        
        # Add confidence score (based on prediction uncertainty)
        # In production, this would be learned from VAE variance
        confidence = self._compute_confidence(z_start, z_current, edit_sequence)
        outcomes['confidence'] = confidence
        
        return outcomes
    
    def _encode_edit(self, edit):
        """Encode an edit to a vector representation."""
        # Simplified encoding for demo
        # In production, this would be a learned embedding (e.g., CodeBERT)
        
        edit_type_map = {
            'add_column': 0,
            'add_endpoint': 1,
            'add_test': 2,
            'remove_column': 3,
            'remove_endpoint': 4,
        }
        
        embedding = torch.zeros(self.edit_dim)
        
        # One-hot encode edit type
        edit_type = edit.edit_type
        if edit_type in edit_type_map:
            embedding[edit_type_map[edit_type]] = 1.0
        
        # Add edit-specific features
        if hasattr(edit, 'table'):
            embedding[10] = hash(edit.table) % 10 / 10.0
        if hasattr(edit, 'path'):
            embedding[11] = hash(edit.path) % 10 / 10.0
        if hasattr(edit, 'auth'):
            embedding[12] = 1.0 if edit.auth else 0.0
        
        return embedding
    
    def _compute_confidence(self, z_start, z_end, edit_sequence):
        """
        Compute confidence in prediction.
        
        Lower confidence for:
        - Large latent state changes (z_end far from z_start)
        - Long edit sequences (more uncertainty)
        - Novel edit patterns (different from training data)
        """
        # Distance in latent space
        latent_distance = torch.norm(z_end - z_start).item()
        
        # Sequence length penalty
        length_penalty = len(edit_sequence) / 10.0
        
        # Compute confidence (inverse of uncertainty)
        # In production, this would be learned from VAE log_var
        base_confidence = 0.9
        confidence = base_confidence * (1.0 - min(latent_distance / 5.0, 0.5)) * (1.0 - min(length_penalty, 0.3))
        
        # Add noise to simulate realistic predictions
        confidence += np.random.uniform(-0.1, 0.1)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return confidence

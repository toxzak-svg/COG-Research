"""
Self-Model: Learn stable editing policy

Based on research RNN architecture from minimal_self_model/models/self_model.py
Learns to generate edits that keep the system stable.
"""

import torch
import torch.nn as nn


class SelfModel(nn.Module):
    """
    Self-Model learns its own editing policy.
    
    Architecture mirrors research RNN:
    - Input: Sequence of (latent_state, edit) pairs
    - RNN: Learn temporal dependencies in edit history
    - Output: Next edit that maintains stability
    
    Key insight from research: Self-model-first is more stable than world-model-first
    because it learns to self-correct rather than just predict.
    """
    
    def __init__(self, latent_dim=16, edit_dim=32, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.edit_dim = edit_dim
        self.hidden_dim = hidden_dim
        
        # RNN for sequence modeling
        # Input: (latent_state, edit) concatenated
        self.rnn = nn.LSTM(
            input_size=latent_dim + edit_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Output layer: predict next edit
        self.fc = nn.Linear(hidden_dim, edit_dim)
    
    def forward(self, state_edit_history):
        """
        Predict next edit given history.
        
        Args:
            state_edit_history: Tensor of shape (batch, seq_len, latent_dim + edit_dim)
        
        Returns:
            next_edit: Tensor of shape (batch, edit_dim)
        """
        rnn_out, (h_n, c_n) = self.rnn(state_edit_history)
        
        # Use last hidden state to predict next edit
        next_edit = self.fc(rnn_out[:, -1, :])
        
        return next_edit
    
    def predict_next_edit(self, latent_state, edit_history):
        """
        Predict next edit that maintains stability.
        
        This is the key capability: given current state and past edits,
        what's the next safe edit to make?
        """
        # Prepare input sequence
        # Concatenate latent_state with each past edit
        history_with_state = []
        for edit_embedding in edit_history:
            combined = torch.cat([latent_state, edit_embedding], dim=0)
            history_with_state.append(combined)
        
        # Stack into sequence (1, seq_len, latent_dim + edit_dim)
        input_seq = torch.stack(history_with_state).unsqueeze(0)
        
        # Predict next edit
        next_edit = self.forward(input_seq).squeeze(0)
        
        return next_edit
    
    def get_hidden_state(self, state_edit_history):
        """
        Get RNN hidden state for Jacobian computation.
        
        This is used by stability monitor to compute spectral radius.
        """
        _, (h_n, c_n) = self.rnn(state_edit_history)
        return h_n, c_n

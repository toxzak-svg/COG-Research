# Implementation of a Minimal Self-Model using RNN for Minimal Self-Model First Architectures

import torch
from torch import nn


class SelfModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SelfModel, self).__init__()
        
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        # Fully connected layer for output
        output = self.fc(rnn_out)  # Remove slicing to return predictions for all time steps
        return output


class HierarchicalSelfModel(nn.Module):
    """
    Hierarchical Self-Model Architecture for multi-timescale predictions.
    
    Level 0: Fast self-model for 1-10 step predictions (primitive actions)
    Level 1: Meta-learner that modulates Level 0 behavior based on context/goals
    Level 2: Goal encoder that predicts high-level objectives
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = None,
        n_levels: int = 2,
        level0_horizon: int = 10,
        level1_horizon: int = 50,
    ):
        super(HierarchicalSelfModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.n_levels = n_levels
        self.level0_horizon = level0_horizon
        self.level1_horizon = level1_horizon
        
        # Level 0: Fast self-model (same as original SelfModel)
        self.level0 = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.level0_fc = nn.Linear(hidden_dim, self.output_dim)
        
        if n_levels >= 2:
            # Level 1: Meta-learner for medium-term predictions
            self.level1_context_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.level1_meta = nn.GRUCell(hidden_dim + input_dim, hidden_dim)
            self.level1_output = nn.Linear(hidden_dim, self.output_dim)
            self.level1_iterations = 5
        
        if n_levels >= 3:
            # Level 2: Goal encoder for high-level objectives
            self.level2_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.level2_goal_predictor = nn.Linear(hidden_dim, hidden_dim)
    
    def forward_level0(self, x: torch.Tensor) -> torch.Tensor:
        """Level 0: Fast predictions (1-10 steps)."""
        rnn_out, hidden = self.level0(x)
        output = self.level0_fc(rnn_out)
        return output
    
    def forward_level1(self, x: torch.Tensor, goal_embedding: torch.Tensor = None) -> torch.Tensor:
        """Level 1: Medium-term predictions (10-50 steps)."""
        level0_output, _ = self.level0(x)
        context_encoded, _ = self.level1_context_encoder(level0_output[:, -self.level0_horizon:, :])
        context_summary = context_encoded[:, -1, :]
        
        predictions = []
        current_state = x[:, -1, :]
        
        for _ in range(self.level1_iterations):
            if goal_embedding is not None:
                meta_input = torch.cat([context_summary, current_state, goal_embedding], dim=-1)
            else:
                meta_input = torch.cat([context_summary, current_state], dim=-1)
            
            meta_hidden = self.level1_meta(meta_input, context_summary)
            pred = self.level1_output(meta_hidden)
            predictions.append(pred)
            current_state = pred
        
        return torch.stack(predictions, dim=1)
    
    def forward_level2(self, x: torch.Tensor) -> torch.Tensor:
        """Level 2: High-level goal prediction."""
        encoded, hidden = self.level2_encoder(x)
        goal_embedding = self.level2_goal_predictor(hidden.squeeze(0))
        return goal_embedding
    
    def forward(self, x: torch.Tensor, level: int = 0, return_goal: bool = False) -> torch.Tensor:
        """Forward pass through hierarchical self-model."""
        if level == 0:
            return self.forward_level0(x)
        elif level == 1:
            return self.forward_level1(x)
        elif level == 2:
            goal = self.forward_level2(x)
            if return_goal:
                return goal
            return self.forward_level1(x, goal_embedding=goal)
        else:
            raise ValueError(f"Invalid level: {level}")


class MultiTimescaleSelfModel(nn.Module):
    """
    Multi-timescale self-model that predicts at multiple horizons simultaneously.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = None,
        horizons: list = None,
    ):
        super(MultiTimescaleSelfModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.horizons = horizons or [1, 5, 10, 24, 48]
        
        # Shared encoder
        self.encoder = nn.RNN(input_dim, hidden_dim, batch_first=True)
        
        # Separate prediction heads for each horizon
        self.heads = nn.ModuleDict()
        for h in self.horizons:
            self.heads[str(h)] = nn.Linear(hidden_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor) -> dict:
        """Predict at multiple horizons simultaneously."""
        encoded, hidden = self.encoder(x)
        final_hidden = hidden.squeeze(0)
        
        predictions = {}
        for h in self.horizons:
            predictions[h] = self.heads[str(h)](final_hidden)
        
        return predictions


# Loss function for self-model
def self_model_loss(predicted, target):
    return nn.functional.mse_loss(predicted, target)
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

# Loss function for self-model
def self_model_loss(predicted, target):
    return nn.functional.mse_loss(predicted, target)  # Ensure dimensions match for all time steps
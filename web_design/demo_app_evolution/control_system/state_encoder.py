"""
State Encoder: App State → Latent Vector

Converts high-dimensional app state into compressed latent representation.
Based on research VAE encoder.
"""

import torch
import torch.nn as nn
import numpy as np


class StateEncoder(nn.Module):
    """Encode app state into latent vector using learned embeddings."""
    
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # For demo purposes, use simple feature extractors
        # In production, these would be learned encoders (e.g., CodeBERT for code)
        self.schema_dim = 64
        self.endpoint_dim = 32
        self.test_dim = 16
        self.auth_dim = 8
        
        total_dim = self.schema_dim + self.endpoint_dim + self.test_dim + self.auth_dim
        
        # Simple MLP encoder (in production, would be pre-trained VAE)
        self.encoder = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
    
    def encode(self, app_state):
        """Encode app state to latent vector."""
        # Extract features (simplified for demo)
        schema_features = self._encode_schema(app_state.schema)
        endpoint_features = self._encode_endpoints(app_state.endpoints)
        test_features = self._encode_tests(app_state.tests)
        auth_features = self._encode_auth(app_state.auth_config)
        
        # Concatenate all features
        features = torch.cat([
            schema_features,
            endpoint_features,
            test_features,
            auth_features
        ], dim=0)
        
        # Encode to latent space
        latent = self.encoder(features)
        return latent
    
    def _encode_schema(self, schema):
        """Encode database schema."""
        # Simplified: random projection based on schema structure
        num_tables = len(schema['tables'])
        num_columns = sum(len(cols) for cols in schema['columns'].values())
        num_indices = sum(len(idxs) for idxs in schema['indices'].values())
        
        # Create feature vector (in production, use graph neural network)
        features = torch.zeros(self.schema_dim)
        features[0] = num_tables / 10.0  # Normalize
        features[1] = num_columns / 20.0
        features[2] = num_indices / 10.0
        
        # Add some hash-based features for table names
        for i, table in enumerate(schema['tables'][:10]):
            idx = (hash(table) % (self.schema_dim - 3)) + 3
            features[idx] = 1.0
        
        return features
    
    def _encode_endpoints(self, endpoints):
        """Encode API endpoints."""
        # Simplified: count-based features
        features = torch.zeros(self.endpoint_dim)
        features[0] = len(endpoints) / 10.0
        
        num_post = sum(1 for ep in endpoints if 'POST' in ep['methods'])
        num_get = sum(1 for ep in endpoints if 'GET' in ep['methods'])
        num_auth = sum(1 for ep in endpoints if ep['auth'])
        
        features[1] = num_post / 5.0
        features[2] = num_get / 5.0
        features[3] = num_auth / 5.0
        
        # Add path-based features
        for i, ep in enumerate(endpoints[:10]):
            idx = (hash(ep['path']) % (self.endpoint_dim - 4)) + 4
            features[idx] = 1.0
        
        return features
    
    def _encode_tests(self, tests):
        """Encode test suite state."""
        features = torch.zeros(self.test_dim)
        features[0] = tests['total'] / 20.0
        features[1] = tests['passing'] / 20.0
        features[2] = tests['coverage']
        
        if tests['total'] > 0:
            features[3] = tests['passing'] / tests['total']  # Pass rate
        
        return features
    
    def _encode_auth(self, auth_config):
        """Encode authentication configuration."""
        features = torch.zeros(self.auth_dim)
        features[0] = 1.0 if auth_config['enabled'] else 0.0
        features[1] = len(auth_config['protected_endpoints']) / 10.0
        
        return features

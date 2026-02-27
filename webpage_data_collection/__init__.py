"""
Webpage Data Collection Infrastructure
Phase 1: Foundation - Data Collection for Training

This module provides tools to capture webpage states, edit sequences,
and outcomes from real sites to train the VAE world-model and RNN self-model.
"""

from .state_schema import WebpageState, LayoutTree, Component, StyleSheet
from .state_collector import WebpageStateCollector
from .outcome_tracker import OutcomeTracker
from .dataset_builder import WebpageEvolutionDataset

__all__ = [
    'WebpageState',
    'LayoutTree', 
    'Component',
    'StyleSheet',
    'WebpageStateCollector',
    'OutcomeTracker',
    'WebpageEvolutionDataset',
]

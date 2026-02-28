"""
Webpage Data Collection Infrastructure
Phase 1: Foundation - Data Collection for Training

This module provides tools to capture webpage states, edit sequences,
and outcomes from real sites to train the VAE world-model and RNN self-model.
"""

from .state_schema import (
    WebpageState, LayoutTree, Component, StyleSheet,
    SiteSnapshot, EditRecord, OutcomeMetrics, PerformanceMetrics,
    SEOMetadata, SocialProof, ComponentType, LayoutType
)
from .state_collector import WebpageStateCollector
from .outcome_tracker import (
    OutcomeTracker, GA4OutcomeTracker, MixpanelOutcomeTracker,
    SegmentOutcomeTracker, CustomOutcomeTracker, create_outcome_tracker
)
from .dataset_builder import (
    WebpageStateDataset, EditSequenceDataset, create_dataloaders
)
from .layout_algorithms import (
    LayoutAlgorithms, LayoutPattern, LayoutPosition,
    suggest_layout_for_components
)
from .awwwards_loader import AwwwardsLoader

__all__ = [
    # State schema
    'WebpageState',
    'LayoutTree',
    'Component',
    'StyleSheet',
    'SiteSnapshot',
    'EditRecord',
    'OutcomeMetrics',
    'PerformanceMetrics',
    'SEOMetadata',
    'SocialProof',
    'ComponentType',
    'LayoutType',
    
    # State collector
    'WebpageStateCollector',
    
    # Outcome trackers
    'OutcomeTracker',
    'GA4OutcomeTracker',
    'MixpanelOutcomeTracker',
    'SegmentOutcomeTracker',
    'CustomOutcomeTracker',
    'create_outcome_tracker',
    
    # Dataset builders
    'WebpageStateDataset',
    'EditSequenceDataset',
    'create_dataloaders',
    
    # Layout algorithms
    'LayoutAlgorithms',
    'LayoutPattern',
    'LayoutPosition',
    'suggest_layout_for_components',
    
    # Data loaders
    'AwwwardsLoader',
]

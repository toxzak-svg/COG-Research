"""
Semantic Augmentation for WebpageState

Generates diverse training samples while preserving semantic identity.
Simplified version that works with the actual state_schema.
"""

import random
import copy
from typing import List
from dataclasses import replace

from .state_schema import WebpageState, Component, PerformanceMetrics, OutcomeMetrics


class SemanticAugmenter:
    """
    Semantic-preserving augmentation for webpage states
    """
    
    def __init__(
        self,
        augment_factor: int = 10,
        emphasis_jitter_prob: float = 0.4,
        size_jitter_prob: float = 0.5,
        performance_jitter_prob: float = 0.5,
        outcome_jitter_prob: float = 0.3,
        seed: int = None,
    ):
        """
        Args:
            augment_factor: Number of augmented versions per original state
            emphasis_jitter_prob: Probability of jittering emphasis levels
            size_jitter_prob: Probability of jittering component sizes
            performance_jitter_prob: Probability of jittering performance metrics
            outcome_jitter_prob: Probability of jittering outcome values
            seed: Random seed for reproducibility
        """
        self.augment_factor = augment_factor
        self.emphasis_jitter_prob = emphasis_jitter_prob
        self.size_jitter_prob = size_jitter_prob
        self.performance_jitter_prob = performance_jitter_prob
        self.outcome_jitter_prob = outcome_jitter_prob
        
        if seed is not None:
            random.seed(seed)
    
    def augment_state(self, state: WebpageState) -> List[WebpageState]:
        """
        Generate augmented versions of a webpage state
        
        Args:
            state: Original WebpageState
        
        Returns:
            List of augmented states (length = augment_factor)
        """
        augmented = [state]  # Include original
        
        for i in range(self.augment_factor - 1):
            aug_state = copy.deepcopy(state)
            aug_state.snapshot_version = state.snapshot_version + i + 1
            
            # Apply augmentations with probabilities
            if random.random() < self.emphasis_jitter_prob:
                aug_state = self._jitter_emphasis(aug_state)
            
            if random.random() < self.size_jitter_prob:
                aug_state = self._jitter_sizes(aug_state)
            
            if random.random() < self.performance_jitter_prob:
                aug_state = self._jitter_performance(aug_state)
            
            if random.random() < self.outcome_jitter_prob:
                aug_state = self._jitter_outcomes(aug_state)
            
            augmented.append(aug_state)
        
        return augmented
    
    def _jitter_emphasis(self, state: WebpageState) -> WebpageState:
        """Randomly shift component emphasis levels by ±1 (bounded 0-5)"""
        if not state.components:
            return state
        
        jittered_components = []
        for comp in state.components:
            shift = random.choice([-1, 0, 1])
            new_emphasis = max(0, min(5, comp.emphasis_level + shift))
            jittered_comp = replace(comp, emphasis_level=new_emphasis)
            jittered_components.append(jittered_comp)
        
        return replace(state, components=jittered_components)
    
    def _jitter_sizes(self, state: WebpageState) -> WebpageState:
        """Randomly adjust component size levels by ±1 (bounded 0-5)"""
        if not state.components:
            return state
        
        jittered_components = []
        for comp in state.components:
            shift = random.choice([-1, 0, 1])
            new_size = max(0, min(5, comp.size_level + shift))
            jittered_comp = replace(comp, size_level=new_size)
            jittered_components.append(jittered_comp)
        
        return replace(state, components=jittered_components)
    
    def _jitter_performance(self, state: WebpageState) -> WebpageState:
        """Add noise to performance metrics (±10%)"""
        if not state.performance:
            return state
        
        perf = state.performance
        
        def jitter_value(value, min_val=0.0, max_val=100.0):
            if value is None:
                return None
            noise = random.uniform(-0.1, 0.1)
            jittered = value * (1 + noise)
            return max(min_val, min(max_val, jittered))
        
        new_perf = replace(
           perf,
            performance_score=jitter_value(perf.performance_score, 0, 100),
            accessibility_score=jitter_value(perf.accessibility_score, 0, 100),
            seo_score=jitter_value(perf.seo_score, 0, 100),
            best_practices_score=jitter_value(perf.best_practices_score, 0, 100),
            lcp=jitter_value(perf.lcp, 0, 10),
            fid=jitter_value(perf.fid, 0, 1000),
            cls=jitter_value(perf.cls, 0, 1),
        )
        
        return replace(state, performance=new_perf)
    
    def _jitter_outcomes(self, state: WebpageState) -> WebpageState:
        """Add small noise to outcome values (±5%)"""
        if not state.outcomes:
            return state
        
        outcomes = state.outcomes
        
        def jitter_value(value, min_val=0.0, max_val=1.0):
            if value is None:
                return None
            noise = random.uniform(-0.05, 0.05)
            jittered = value * (1 + noise)
            return max(min_val, min(max_val, jittered))
        
        new_outcomes = replace(
            outcomes,
            conversion_rate=jitter_value(outcomes.conversion_rate),
            bounce_rate=jitter_value(outcomes.bounce_rate),
            avg_time_on_page=jitter_value(outcomes.avg_time_on_page, 0, 1000),
            scroll_depth_avg=jitter_value(outcomes.scroll_depth_avg),
            clicks_per_session=jitter_value(outcomes.clicks_per_session, 0, 50),
            pages_per_session=jitter_value(outcomes.pages_per_session, 0, 20),
            exit_rate=jitter_value(outcomes.exit_rate),
        )
        
        return replace(state, outcomes=new_outcomes)
    
    def augment_batch(self, states: List[WebpageState]) -> List[WebpageState]:
        """
        Augment a batch of states
        
        Args:
            states: List of original WebpageStates
        
        Returns:
            List of all original + augmented states
        """
        all_states = []
        for state in states:
            augmented = self.augment_state(state)
            all_states.extend(augmented)
        return all_states


def create_semantic_augmenter(
    augment_factor: int = 10,
    aggressive: bool = False,
    seed: int = 42,
) -> SemanticAugmenter:
    """
    Factory function for creating augmenter with preset configurations
    
    Args:
        augment_factor: Number of augmented versions per original
        aggressive: If True, use higher augmentation probabilities
        seed: Random seed
    
    Returns:
        Configured SemanticAugmenter
    """
    if aggressive:
        return SemanticAugmenter(
            augment_factor=augment_factor,
            emphasis_jitter_prob=0.7,
            size_jitter_prob=0.7,
            performance_jitter_prob=0.6,
            outcome_jitter_prob=0.5,
            seed=seed,
        )
    else:
        return SemanticAugmenter(
            augment_factor=augment_factor,
            seed=seed,
        )

"""
Awwwards Design Loader
Parse Awwwards and portfolio databases to create high-quality seed training data

Award-winning websites represent proven high-outcome examples that can:
1. Seed VAE training before collecting from 1000 partner sites
2. Serve as "gold standard" examples (quality_score = 1.0)
3. Provide design patterns for component categorization
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from .state_schema import (
    WebpageState, LayoutTree, Component, StyleSheet,
    OutcomeMetrics, PerformanceMetrics, ComponentType, LayoutType,
    SEOMetadata, SocialProof
)


class AwwwardsLoader:
    """
    Load design patterns from Awwwards and portfolio databases
    
    Converts technical patterns and portfolio examples into WebpageState format
    """
    
    def __init__(self, webalchemist_path: str = "c:/dev/projects/WebAlchemist"):
        """
        Args:
            webalchemist_path: Path to WebAlchemist project root
        """
        self.base_path = Path(webalchemist_path)
        self.awwwards_db = self.base_path / "organized_rag/awwwards-webgpu-rag.json"
        self.portfolio_db = self.base_path / "organized_rag/portfolio-builder-database.json"
    
    def load_awwwards_patterns(self) -> List[Dict]:
        """
        Load Awwwards design patterns from database
        
        Returns:
            List of pattern dicts with id, category, title, summary, tags
        """
        if not self.awwwards_db.exists():
            print(f"Warning: Awwwards database not found at {self.awwwards_db}")
            return []
        
        with open(self.awwwards_db, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        patterns = data.get('records', [])
        print(f"Loaded {len(patterns)} Awwwards patterns")
        
        return patterns
    
    def load_portfolio_schema(self) -> Dict:
        """
        Load portfolio builder database schema
        
        Returns:
            Schema dict with collections and field definitions
        """
        if not self.portfolio_db.exists():
            print(f"Warning: Portfolio database not found at {self.portfolio_db}")
            return {}
        
        with open(self.portfolio_db, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded portfolio schema: {data.get('source', 'unknown')}")
        
        return data
    
    def create_seed_states_from_patterns(
        self,
        patterns: Optional[List[Dict]] = None
    ) -> List[WebpageState]:
        """
        Convert Awwwards patterns to WebpageState objects
        
        These are simplified states representing proven design patterns
        Used as seed data for initial model training
        
        Args:
            patterns: Pattern dicts (loads from DB if None)
        
        Returns:
            List of WebpageState objects with quality_score = 1.0
        """
        if patterns is None:
            patterns = self.load_awwwards_patterns()
        
        states = []
        
        for pattern in patterns:
            state = self._pattern_to_state(pattern)
            if state:
                states.append(state)
        
        print(f"Created {len(states)} seed states from Awwwards patterns")
        
        return states
    
    def _pattern_to_state(self, pattern: Dict) -> Optional[WebpageState]:
        """
        Convert a single Awwwards pattern to WebpageState
        
        Patterns are technical implementations (canvas sizing, responsive design, etc.)
        We create simplified states capturing the design principles
        
        Args:
            pattern: Pattern dict with id, category, title, summary, tags
        
        Returns:
            WebpageState object or None if conversion fails
        """
        pattern_id = pattern.get('id', 'unknown')
        category = pattern.get('category', 'web_integration')
        title = pattern.get('title', 'Untitled Pattern')
        summary = pattern.get('summary', '')
        tags = pattern.get('tags', [])
        
        # Create synthetic site_id and page_id
        site_id = f"awwwards_{pattern_id}"
        page_id = pattern_id
        timestamp = datetime.utcnow().isoformat()
        
        # Infer components from pattern type
        components = self._infer_components_from_pattern(pattern)
        
        # Create simplified layout
        layout = self._create_layout_from_category(category, len(components))
        
        # Create basic styling based on category
        styling = self._create_styling_from_category(category)
        
        # Assign high-quality outcome metrics
        # Award-winning = proven effectiveness
        outcomes = OutcomeMetrics(
            conversion_rate=0.085,  # Assume high conversion (8.5%)
            bounce_rate=0.35,  # Assume low bounce
            avg_time_on_page=180.0,  # Assume good engagement (3 min)
            unique_visitors=None,  # Unknown traffic
            pageviews=None,
            measurement_start=None,
            measurement_end=None,
            measurement_duration_hours=None,
        )
        
        # Create SEO metadata from pattern info
        seo = SEOMetadata(
            meta_title=title,
            meta_description=summary[:160] if len(summary) > 160 else summary,
            og_title=title,
            og_description=summary[:200] if len(summary) > 200 else summary,
            og_image="",  # Unknown from pattern
            keywords=tags,
            canonical_url="",
            schema_markup=None,
        )
        
        # Create social proof (high for award-winning designs)
        social_proof = SocialProof(
            views=None,  # Unknown from pattern
            likes=None,
            shares=None,
            comments=None,
            github_stars=None,
            upvotes=None,
        )
        
        # Infer tech stack from tags
        tech_stack = []
        tech_keywords = ['webgpu', 'three.js', 'canvas', 'webgl', 'react', 'vue', 'svelte']
        for tag in tags:
            tag_lower = tag.lower()
            for tech in tech_keywords:
                if tech in tag_lower:
                    tech_stack.append(tech)
        
        # Infer project category from pattern category
        category_map = {
            'web_integration': 'landing',
            'production_deployment': 'saas',
            'advanced_visual_pipelines': 'portfolio',
            'interaction_orchestration': 'interactive',
        }
        project_category = category_map.get(category, 'landing')
        
        # Create WebpageState
        state = WebpageState(
            site_id=site_id,
            page_id=page_id,
            timestamp=timestamp,
            layout=layout,
            components=components,
            styling=styling,
            assets=[],
            events=[],
            animations=[],
            outcomes=outcomes,
            performance=None,
            cognitive_load=None,
            page_title=title,
            meta_description=summary,
            headline=title,
            subheadline=summary[:100] if len(summary) > 100 else summary,
            cta_primary="Learn More",
            cta_secondary="View Demo",
            body_copy={'main': summary},
            project_category=project_category,
            tech_stack=tech_stack,
            tags=tags,
            seo_metadata=seo,
            social_proof=social_proof,
        )
        
        return state
    
    def _infer_components_from_pattern(self, pattern: Dict) -> List[Component]:
        """
        Infer webpage components from pattern type
        
        E.g., "canvas embed" pattern -> HERO component with canvas
             "fallback UI" pattern -> HERO + CTA components
        """
        components = []
        category = pattern.get('category', 'web_integration')
        tags = pattern.get('tags', [])
        pattern_id = pattern.get('id', 'unknown')
        
        # Web integration patterns typically have hero sections
        if category == 'web_integration':
            components.append(Component(
                component_id=f"{pattern_id}_hero",
                component_type=ComponentType.HERO,
                element_id='hero-section',
                text_content=pattern.get('title', ''),
                emphasis_level=5,  # High prominence
                color_role='primary',
                size_level=5,
                clickable=False,
            ))
            
            # If it has interactive elements, add a CTA
            if any(tag in tags for tag in ['interaction', 'animation', 'responsive']):
                components.append(Component(
                    component_id=f"{pattern_id}_cta",
                    component_type=ComponentType.CTA,
                    element_id='primary-cta',
                    text_content='Try Demo',
                    emphasis_level=4,
                    color_role='accent',
                    size_level=4,
                    clickable=True,
                    href='#demo',
                ))
        
        # Production deployment patterns have more structured layouts
        elif category == 'production_deployment':
            # Navigation
            components.append(Component(
                component_id=f"{pattern_id}_nav",
                component_type=ComponentType.NAV,
                element_id='main-nav',
                text_content='Navigation',
                emphasis_level=3,
                color_role='neutral',
                size_level=3,
                clickable=True,
            ))
            
            # Content blocks
            components.append(Component(
                component_id=f"{pattern_id}_content",
                component_type=ComponentType.TEXT_BLOCK,
                element_id='content',
                text_content=pattern.get('summary', ''),
                emphasis_level=3,
                color_role='neutral',
                size_level=3,
                clickable=False,
            ))
        
        # Advanced visual pipelines -> showcases
        elif category == 'advanced_visual_pipelines':
            components.append(Component(
                component_id=f"{pattern_id}_hero",
                component_type=ComponentType.HERO,
                element_id='hero',
                text_content=pattern.get('title', ''),
                emphasis_level=5,
                color_role='primary',
                size_level=5,
                clickable=False,
            ))
            
            # Gallery/showcase component
            components.append(Component(
                component_id=f"{pattern_id}_showcase",
                component_type=ComponentType.CARD,
                element_id='showcase',
                text_content='Visual Examples',
                emphasis_level=4,
                color_role='secondary',
                size_level=4,
                clickable=True,
            ))
        
        # Interaction orchestration -> forms and buttons
        elif category == 'interaction_orchestration':
            components.append(Component(
                component_id=f"{pattern_id}_form",
                component_type=ComponentType.FORM,
                element_id='interaction-form',
                text_content='Configure',
                emphasis_level=4,
                color_role='primary',
                size_level=4,
                clickable=True,
            ))
            
            components.append(Component(
                component_id=f"{pattern_id}_button",
                component_type=ComponentType.BUTTON,
                element_id='submit',
                text_content='Apply',
                emphasis_level=4,
                color_role='accent',
                size_level=3,
                clickable=True,
            ))
        
        # Default: add at least a hero and CTA
        if not components:
            components.append(Component(
                component_id=f"{pattern_id}_hero",
                component_type=ComponentType.HERO,
                element_id='hero',
                text_content=pattern.get('title', ''),
                emphasis_level=5,
                color_role='primary',
                size_level=5,
                clickable=False,
            ))
            
            components.append(Component(
                component_id=f"{pattern_id}_cta",
                component_type=ComponentType.CTA,
                element_id='cta',
                text_content='Get Started',
                emphasis_level=4,
                color_role='accent',
                size_level=4,
                clickable=True,
            ))
        
        return components
    
    def _create_layout_from_category(self, category: str, num_components: int) -> List[LayoutTree]:
        """
        Create simplified layout structure based on category
        """
        layout = []
        
        # Root container
        layout.append(LayoutTree(
            element_id='root',
            tag='div',
            layout_type=LayoutType.FLEX,
            parent_id=None,
            children=['hero', 'content'],
            is_container=True,
            flex_direction='column',
            padding_level=3,
            margin_level=0,
            gap_level=3,
        ))
        
        # Hero section
        layout.append(LayoutTree(
            element_id='hero',
            tag='section',
            layout_type=LayoutType.FLEX,
            parent_id='root',
            children=[],
            is_container=True,
            flex_direction='column',
            padding_level=5,
            margin_level=2,
            gap_level=2,
        ))
        
        # Content section
        layout.append(LayoutTree(
            element_id='content',
            tag='section',
            layout_type=LayoutType.GRID if num_components > 3 else LayoutType.FLEX,
            parent_id='root',
            children=[],
            is_container=True,
            columns=2 if num_components > 3 else None,
            flex_direction='row' if num_components <= 3 else None,
            padding_level=4,
            margin_level=2,
            gap_level=3,
        ))
        
        return layout
    
    def _create_styling_from_category(self, category: str) -> StyleSheet:
        """
        Create basic styling based on category
        """
        # Different categories have different color schemes
        color_schemes = {
            'web_integration': {
                'primary': '#2563eb',  # Blue
                'secondary': '#64748b',  # Slate
                'accent': '#f59e0b',  # Amber
            },
            'production_deployment': {
                'primary': '#059669',  # Green
                'secondary': '#6b7280',  # Gray
                'accent': '#8b5cf6',  # Purple
            },
            'advanced_visual_pipelines': {
                'primary': '#dc2626',  # Red
                'secondary': '#737373',  # Neutral
                'accent': '#ec4899',  # Pink
            },
            'interaction_orchestration': {
                'primary': '#7c3aed',  # Violet
                'secondary': '#71717a',  # Zinc
                'accent': '#06b6d4',  # Cyan
            },
        }
        
        scheme = color_schemes.get(category, color_schemes['web_integration'])
        
        return StyleSheet(
            primary_color=scheme['primary'],
            secondary_color=scheme['secondary'],
            accent_color=scheme['accent'],
            background_color='#ffffff',
            text_color='#1f2937',
            font_family_heading='Inter, system-ui, sans-serif',
            font_family_body='Inter, system-ui, sans-serif',
            base_font_size=16,
            line_height=1.6,
            border_radius=8,
        )


def load_seed_data(webalchemist_path: str = "c:/dev/projects/WebAlchemist") -> List[WebpageState]:
    """
    Convenience function to load all seed data from WebAlchemist
    
    Usage:
        >>> from webpage_data_collection import load_seed_data
        >>> seed_states = load_seed_data()
        >>> print(f"Loaded {len(seed_states)} high-quality seed examples")
    
    Args:
        webalchemist_path: Path to WebAlchemist project
    
    Returns:
        List of WebpageState objects representing award-winning designs
    """
    loader = AwwwardsLoader(webalchemist_path)
    return loader.create_seed_states_from_patterns()

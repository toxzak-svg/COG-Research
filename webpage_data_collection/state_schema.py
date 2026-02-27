"""
Webpage State Schema
Defines the complete state representation for a webpage at time t

This schema captures:
1. Visual/Design Layer: Layout, components, styling
2. Content Layer: Copy, media assets
3. Interaction Layer: Events, animations
4. Performance Metrics: Lighthouse scores, Core Web Vitals
5. Business Outcomes: Conversion rates, engagement metrics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class LayoutType(Enum):
    """Layout system types"""
    FLEX = "flex"
    GRID = "grid"
    ABSOLUTE = "absolute"
    FLOAT = "float"
    BLOCK = "block"


class ComponentType(Enum):
    """Common webpage component types"""
    BUTTON = "button"
    FORM = "form"
    INPUT = "input"
    NAV = "nav"
    HERO = "hero"
    CARD = "card"
    MODAL = "modal"
    DROPDOWN = "dropdown"
    TESTIMONIAL = "testimonial"
    SOCIAL_PROOF = "social_proof"
    CTA = "cta"
    IMAGE = "image"
    VIDEO = "video"
    TEXT_BLOCK = "text_block"
    HEADER = "header"
    FOOTER = "footer"


@dataclass
class LayoutTree:
    """
    DOM structure representation
    Captures hierarchy and positioning without pixel-level details
    """
    element_id: str
    tag: str  # div, section, header, etc.
    layout_type: LayoutType
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    
    # Layout properties
    is_container: bool = False
    columns: Optional[int] = None  # For grid layouts
    flex_direction: Optional[str] = None  # row, column
    
    # Positioning
    position_relative: bool = False
    z_index: int = 0
    
    # Dimensions (semantic, not pixel-perfect)
    width_type: str = "full"  # full, auto, fixed, percentage
    height_type: str = "auto"
    
    # Spacing (semantic levels)
    padding_level: int = 2  # 0-5 (none to xl)
    margin_level: int = 2
    gap_level: int = 1  # For flex/grid gaps


@dataclass
class Component:
    """
    Individual webpage component
    Focus on semantic properties, not implementation details
    """
    component_id: str
    component_type: ComponentType
    element_id: str  # Links to LayoutTree
    
    # Content
    text_content: Optional[str] = None
    placeholder: Optional[str] = None
    
    # Visual properties
    emphasis_level: int = 3  # 0-5 (low to high visual weight)
    color_role: str = "neutral"  # primary, secondary, accent, neutral, danger
    size_level: int = 3  # 0-5 (xs to xl)
    
    # Interaction
    clickable: bool = False
    href: Optional[str] = None
    form_action: Optional[str] = None
    
    # State
    disabled: bool = False
    required: bool = False
    
    # A/B testing variant (if applicable)
    variant: Optional[str] = None


@dataclass
class StyleSheet:
    """
    Global style properties
    Captures design system, not individual CSS rules
    """
    # Color system
    primary_color: str  # Hex color
    secondary_color: str
    accent_color: str
    background_color: str
    text_color: str
    
    # Typography
    font_family_heading: str
    font_family_body: str
    base_font_size: int  # pixels
    line_height: float
    
    # Spacing system (multiplier values)
    spacing_unit: int = 8  # Base unit in pixels
    
    # Visual style
    border_radius: int = 4  # pixels
    shadow_style: str = "subtle"  # none, subtle, medium, strong
    
    # Brand coherence score (computed)
    color_consistency: float = 0.0  # 0-1
    spacing_consistency: float = 0.0  # 0-1


@dataclass
class Asset:
    """Media asset (image, video, icon)"""
    asset_id: str
    asset_type: str  # image, video, icon, logo
    url: str
    alt_text: Optional[str] = None
    
    # Performance
    file_size_kb: Optional[float] = None
    optimized: bool = False
    lazy_loaded: bool = False


@dataclass
class EventHandler:
    """Interaction event handler"""
    event_id: str
    element_id: str
    event_type: str  # click, submit, scroll, hover, focus
    action: str  # navigate, show_modal, submit_form, etc.
    
    # Analytics tracking
    tracked: bool = False
    track_as: Optional[str] = None  # GA event name


@dataclass
class Animation:
    """Animation or transition"""
    animation_id: str
    element_id: str
    trigger: str  # on_load, on_scroll, on_hover, on_click
    effect: str  # fade_in, slide_in, scale, rotate
    duration_ms: int = 300
    easing: str = "ease"


@dataclass
class PerformanceMetrics:
    """
    Performance and technical quality metrics
    """
    # Lighthouse scores (0-100)
    performance_score: Optional[float] = None
    accessibility_score: Optional[float] = None
    seo_score: Optional[float] = None
    best_practices_score: Optional[float] = None
    
    # Core Web Vitals
    lcp: Optional[float] = None  # Largest Contentful Paint (seconds)
    fid: Optional[float] = None  # First Input Delay (milliseconds)
    cls: Optional[float] = None  # Cumulative Layout Shift (score)
    
    # Load metrics
    page_size_kb: Optional[float] = None
    load_time_seconds: Optional[float] = None
    time_to_interactive: Optional[float] = None
    
    # Mobile
    mobile_friendly: bool = True
    responsive_design: bool = True


@dataclass
class OutcomeMetrics:
    """
    Business outcome metrics
    Ground truth for training world-model predictions
    """
    # Conversion metrics
    conversion_rate: Optional[float] = None  # Percentage
    conversion_count: Optional[int] = None
    conversion_goal: Optional[str] = None  # signup, purchase, demo, download
    
    # Engagement metrics
    avg_time_on_page: Optional[float] = None  # Seconds
    scroll_depth_avg: Optional[float] = None  # Percentage
    clicks_per_session: Optional[float] = None
    pages_per_session: Optional[float] = None
    
    # Bounce and exit
    bounce_rate: Optional[float] = None  # Percentage
    exit_rate: Optional[float] = None
    
    # Traffic
    unique_visitors: Optional[int] = None
    pageviews: Optional[int] = None
    
    # Attribution window
    measurement_start: Optional[str] = None  # ISO timestamp
    measurement_end: Optional[str] = None
    measurement_duration_hours: Optional[float] = None


@dataclass
class CognitiveLoadMetrics:
    """
    Computed metrics for cognitive load and visual hierarchy
    These are derived from the state, not directly observed
    """
    # Visual complexity
    element_count: int = 0
    visual_clutter_score: float = 0.0  # 0-1 (low to high clutter)
    color_variety_score: float = 0.0  # 0-1 (monochrome to rainbow)
    
    # Hierarchy quality
    hierarchy_depth: int = 0
    emphasis_distribution: List[int] = field(default_factory=list)  # Histogram
    
    # Attention flow
    f_pattern_score: float = 0.0  # 0-1 (how well it follows F-pattern)
    cta_visibility: float = 0.0  # 0-1 (how prominent CTAs are)
    
    # Information architecture
    navigation_complexity: float = 0.0  # 0-1
    content_density: float = 0.0  # words per viewport


@dataclass
class WebpageState:
    """
    Complete state representation of a webpage at time t
    
    This is the main state object that gets compressed to latent space
    by the VAE world-model.
    """
    # Metadata
    site_id: str
    page_id: str
    timestamp: str  # ISO format
    snapshot_version: int = 1
    
    # 1. Visual/Design Layer
    layout: List[LayoutTree] = field(default_factory=list)
    components: List[Component] = field(default_factory=list)
    styling: Optional[StyleSheet] = None
    
    # 2. Content Layer
    page_title: str = ""
    meta_description: str = ""
    headline: str = ""
    subheadline: str = ""
    cta_primary: str = ""
    cta_secondary: str = ""
    body_copy: Dict[str, str] = field(default_factory=dict)  # section_id -> text
    assets: List[Asset] = field(default_factory=list)
    
    # 3. Interaction Layer
    events: List[EventHandler] = field(default_factory=list)
    animations: List[Animation] = field(default_factory=list)
    
    # 4. Performance Metrics
    performance: Optional[PerformanceMetrics] = None
    
    # 5. Outcome Metrics
    outcomes: Optional[OutcomeMetrics] = None
    
    # 6. Computed Metrics
    cognitive_load: Optional[CognitiveLoadMetrics] = None
    
    # 7. Context
    device_type: str = "desktop"  # desktop, mobile, tablet
    traffic_source: Optional[str] = None  # organic, paid, direct, social
    user_segment: Optional[str] = None  # new, returning, high_intent
    
    # 8. A/B Test Context
    experiment_id: Optional[str] = None
    variant_id: Optional[str] = None
    control_group: bool = False
    
    def get_component_by_id(self, component_id: str) -> Optional[Component]:
        """Helper: Find component by ID"""
        for comp in self.components:
            if comp.component_id == component_id:
                return comp
        return None
    
    def get_components_by_type(self, component_type: ComponentType) -> List[Component]:
        """Helper: Find all components of a given type"""
        return [c for c in self.components if c.component_type == component_type]
    
    def count_elements(self) -> int:
        """Helper: Total DOM element count"""
        return len(self.layout)
    
    def count_interactive_elements(self) -> int:
        """Helper: Count clickable elements"""
        return sum(1 for c in self.components if c.clickable)
    
    def get_primary_cta(self) -> Optional[Component]:
        """Helper: Find the primary CTA component"""
        ctas = self.get_components_by_type(ComponentType.CTA)
        if ctas:
            # Return highest emphasis CTA
            return max(ctas, key=lambda c: c.emphasis_level)
        return None


@dataclass
class EditRecord:
    """
    Record of a single edit applied to a webpage
    Used to build edit sequences for training self-model
    """
    edit_id: str
    timestamp: str
    
    # Edit specification
    edit_type: str  # add_component, modify_layout, change_copy, etc.
    target_element_id: Optional[str] = None
    
    # Edit details (structured)
    edit_params: Dict = field(default_factory=dict)
    
    # State delta
    before_state_id: str = ""
    after_state_id: str = ""
    
    # Outcome impact (measured after sufficient traffic)
    outcome_delta: Optional[Dict[str, float]] = None
    
    # Stability metrics
    spectral_radius: Optional[float] = None
    perturbation_return_rate: Optional[float] = None
    
    # Metadata
    applied_by: str = "human"  # human, autonomous, assisted
    rollback: bool = False


@dataclass 
class SiteSnapshot:
    """
    Complete snapshot of a site including state and edit history
    This is what we collect from each site for training
    """
    site_id: str
    domain: str
    industry: str  # saas, ecommerce, content, etc.
    
    # State history
    states: List[WebpageState] = field(default_factory=list)
    
    # Edit history
    edits: List[EditRecord] = field(default_factory=list)
    
    # Metadata
    collection_start: str = ""  # ISO timestamp
    collection_end: str = ""
    total_snapshots: int = 0
    total_edits: int = 0
    
    # Analytics integration
    analytics_provider: str = ""  # google_analytics, mixpanel, segment
    analytics_property_id: str = ""
    
    def get_state_sequence(self) -> List[WebpageState]:
        """Return states in chronological order"""
        return sorted(self.states, key=lambda s: s.timestamp)
    
    def get_edit_sequence(self) -> List[EditRecord]:
        """Return edits in chronological order"""
        return sorted(self.edits, key=lambda e: e.timestamp)
    
    def get_outcome_trajectory(self, metric: str) -> List[Tuple[str, float]]:
        """
        Extract trajectory of a specific outcome metric over time
        Returns: [(timestamp, value), ...]
        """
        trajectory = []
        for state in self.get_state_sequence():
            if state.outcomes:
                value = getattr(state.outcomes, metric, None)
                if value is not None:
                    trajectory.append((state.timestamp, value))
        return trajectory

"""
Layout Algorithm Reference Patterns
Inspired by WebAlchemist portfolio builder

Provides semantic layout generation algorithms for:
- Grid layouts (cards, galleries, dashboards)
- Ring/radial layouts (navigation, feature showcases)
- Spiral layouts (storytelling, progressive disclosure)
- Arc layouts (curved hero sections, testimonials)

These patterns are used for:
1. Synthetic edit generation (what edits are possible?)
2. Layout validation (is this arrangement sensible?)
3. Component positioning (where should this element go?)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
import math

from .state_schema import LayoutTree, Component, LayoutType, ComponentType


class LayoutPattern(Enum):
    """Semantic layout patterns"""
    GRID = "grid"  # Equally-spaced grid (cards, galleries)
    RING = "ring"  # Circular arrangement around center
    SPIRAL = "spiral"  # Spiral outward from center
    ARC = "arc"  # Curved arrangement (hero sections)
    HERO_CONTENT = "hero_content"  # Hero + content blocks
    SIDEBAR = "sidebar"  # Sidebar + main content
    MASONRY = "masonry"  # Pinterest-style staggered grid
    CENTERED = "centered"  # Single centered element


@dataclass
class LayoutPosition:
    """
    Semantic position for a component
    Not pixel-perfect, but relative positioning levels
    """
    row: int  # Vertical position (0-based)
    col: int  # Horizontal position (0-based)
    span_row: int = 1  # Spans multiple rows
    span_col: int = 1  # Spans multiple columns
    z_index: int = 0  # Stacking order


class LayoutAlgorithms:
    """
    Layout generation algorithms
    
    All algorithms return semantic positioning, not pixel coordinates
    This allows the rendering layer to adapt to device/viewport
    """
    
    @staticmethod
    def grid_layout(
        num_components: int,
        columns: int = 3,
        gap_level: int = 3
    ) -> List[LayoutPosition]:
        """
        Generate grid layout positions
        
        Args:
            num_components: Number of components to layout
            columns: Number of columns in grid (3 = typical card grid)
            gap_level: Spacing between items (0-5 semantic scale)
        
        Returns:
            List of LayoutPosition objects
        """
        positions = []
        
        for i in range(num_components):
            row = i // columns
            col = i % columns
            
            positions.append(LayoutPosition(
                row=row,
                col=col,
                span_row=1,
                span_col=1,
                z_index=0,
            ))
        
        return positions
    
    @staticmethod
    def hero_content_layout(
        num_content_blocks: int = 3
    ) -> List[LayoutPosition]:
        """
        Hero section at top, followed by content blocks
        
        Common pattern: Hero (full-width) + 3 feature cards
        
        Returns:
            [hero_position, content_block_1, content_block_2, ...]
        """
        positions = []
        
        # Hero: Row 0, spans all columns
        positions.append(LayoutPosition(
            row=0,
            col=0,
            span_row=1,
            span_col=3,  # Full width
            z_index=0,
        ))
        
        # Content blocks: Grid below hero
        for i in range(num_content_blocks):
            positions.append(LayoutPosition(
                row=1 + (i // 3),
                col=i % 3,
                span_row=1,
                span_col=1,
                z_index=0,
            ))
        
        return positions
    
    @staticmethod
    def sidebar_layout() -> Tuple[LayoutPosition, LayoutPosition]:
        """
        Sidebar + main content area
        
        Returns:
            (sidebar_position, main_content_position)
        """
        sidebar = LayoutPosition(
            row=0,
            col=0,
            span_row=10,  # Extends vertically
            span_col=1,
            z_index=0,
        )
        
        main = LayoutPosition(
            row=0,
            col=1,
            span_row=10,
            span_col=2,  # Main content is wider
            z_index=0,
        )
        
        return sidebar, main
    
    @staticmethod
    def ring_layout(
        num_components: int,
        radius_level: int = 3
    ) -> List[LayoutPosition]:
        """
        Circular arrangement around center point
        
        Good for: Feature showcases, radial navigation, project galleries
        
        Args:
            num_components: Number of items to arrange
            radius_level: Distance from center (0-5 semantic scale)
        
        Returns:
            List of positions (center is row=5, col=5 in a 10x10 grid)
        """
        positions = []
        
        center_row = 5
        center_col = 5
        
        for i in range(num_components):
            angle = (2 * math.pi * i) / num_components
            
            # Convert polar to grid coordinates
            # Radius_level maps to 0-5 grid units
            row_offset = int(radius_level * math.sin(angle))
            col_offset = int(radius_level * math.cos(angle))
            
            positions.append(LayoutPosition(
                row=center_row + row_offset,
                col=center_col + col_offset,
                span_row=1,
                span_col=1,
                z_index=0,
            ))
        
        return positions
    
    @staticmethod
    def spiral_layout(
        num_components: int,
        turns: float = 2.0
    ) -> List[LayoutPosition]:
        """
        Spiral arrangement outward from center
        
        Good for: Storytelling, progressive disclosure, project timelines
        
        Args:
            num_components: Number of items
            turns: Number of full rotations (2.0 = two full spirals)
        
        Returns:
            List of positions starting from center
        """
        positions = []
        
        center_row = 5
        center_col = 5
        
        for i in range(num_components):
            t = i / max(num_components - 1, 1)  # 0 to 1
            angle = turns * 2 * math.pi * t
            radius = 5 * t  # Expand outward (0 to 5 grid units)
            
            row_offset = int(radius * math.sin(angle))
            col_offset = int(radius * math.cos(angle))
            
            positions.append(LayoutPosition(
                row=center_row + row_offset,
                col=center_col + col_offset,
                span_row=1,
                span_col=1,
                z_index=0,
            ))
        
        return positions
    
    @staticmethod
    def arc_layout(
        num_components: int,
        arc_angle: float = 180.0,  # Degrees
        radius_level: int = 4
    ) -> List[LayoutPosition]:
        """
        Arc/curved arrangement (portion of circle)
        
        Good for: Curved hero sections, testimonials, feature highlights
        
        Args:
            num_components: Number of items
            arc_angle: Angle of arc in degrees (180 = semicircle)
            radius_level: Radius of arc (0-5 semantic scale)
        
        Returns:
            List of positions along arc
        """
        positions = []
        
        center_row = 8  # Arc curves upward from bottom
        center_col = 5
        
        start_angle = 90 - (arc_angle / 2)  # Center the arc
        
        for i in range(num_components):
            if num_components == 1:
                angle_deg = 90  # Single item at top of arc
            else:
                angle_deg = start_angle + (arc_angle * i) / (num_components - 1)
            
            angle_rad = math.radians(angle_deg)
            
            row_offset = -int(radius_level * math.sin(angle_rad))
            col_offset = int(radius_level * math.cos(angle_rad))
            
            positions.append(LayoutPosition(
                row=center_row + row_offset,
                col=center_col + col_offset,
                span_row=1,
                span_col=1,
                z_index=0,
            ))
        
        return positions
    
    @staticmethod
    def masonry_layout(
        num_components: int,
        columns: int = 3,
        height_pattern: Optional[List[int]] = None
    ) -> List[LayoutPosition]:
        """
        Masonry/Pinterest-style staggered grid
        
        Items have varying heights, fill columns evenly
        
        Args:
            num_components: Number of items
            columns: Number of columns
            height_pattern: List of heights (1-3 units), cycles if too short
        
        Returns:
            List of positions with variable row spans
        """
        if height_pattern is None:
            # Default: Alternate between 1 and 2 unit heights
            height_pattern = [1, 2, 1, 2, 1, 3]
        
        positions = []
        
        # Track current height of each column
        column_heights = [0] * columns
        
        for i in range(num_components):
            # Find shortest column
            min_col = min(range(columns), key=lambda c: column_heights[c])
            
            # Get height for this item
            height = height_pattern[i % len(height_pattern)]
            
            positions.append(LayoutPosition(
                row=column_heights[min_col],
                col=min_col,
                span_row=height,
                span_col=1,
                z_index=0,
            ))
            
            # Update column height
            column_heights[min_col] += height
        
        return positions
    
    @staticmethod
    def centered_layout() -> LayoutPosition:
        """
        Single centered element
        
        Good for: Login forms, error pages, CTAs
        """
        return LayoutPosition(
            row=5,
            col=5,
            span_row=1,
            span_col=1,
            z_index=0,
        )
    
    @staticmethod
    def infer_layout_pattern(
        layout: List[LayoutTree],
        components: List[Component]
    ) -> LayoutPattern:
        """
        Infer which layout pattern is being used
        
        Useful for:
        - Understanding existing pages
        - Suggesting similar layouts
        - Validating edits (does this fit the pattern?)
        
        Args:
            layout: Current layout tree
            components: Current components
        
        Returns:
            Best matching LayoutPattern
        """
        if len(layout) == 0:
            return LayoutPattern.CENTERED
        
        # Check for hero pattern (one large top element)
        root_layouts = [l for l in layout if l.parent_id is None]
        if len(root_layouts) > 0:
            first = root_layouts[0]
            if first.layout_type == LayoutType.FLEX:
                # Check if first component is HERO
                hero_components = [c for c in components if c.component_type == ComponentType.HERO]
                if len(hero_components) > 0:
                    return LayoutPattern.HERO_CONTENT
        
        # Check for grid (multiple equally-spaced flex items)
        grid_layouts = [l for l in layout if l.layout_type == LayoutType.GRID]
        if len(grid_layouts) > 0:
            return LayoutPattern.GRID
        
        # Check for sidebar (nested flex with one narrow column)
        flex_layouts = [l for l in layout if l.layout_type == LayoutType.FLEX]
        if len(flex_layouts) >= 2:
            # Look for nested flex structures (sidebar pattern)
            for flex in flex_layouts:
                if len(flex.children) >= 2:
                    return LayoutPattern.SIDEBAR
        
        # Default to grid if multiple components
        if len(components) > 3:
            return LayoutPattern.GRID
        
        return LayoutPattern.CENTERED
    
    @staticmethod
    def apply_layout_pattern(
        pattern: LayoutPattern,
        components: List[Component],
        **kwargs: int | float | List[int] | None
    ) -> List[LayoutTree]:
        """
        Generate LayoutTree structure from pattern and components
        
        This creates the actual layout tree that can be assigned to WebpageState
        
        Args:
            pattern: Which layout pattern to use
            components: Components to layout
            **kwargs: Pattern-specific parameters
        
        Returns:
            List of LayoutTree objects
        """
        num_components = len(components)
        
        if pattern == LayoutPattern.GRID:
            columns = kwargs.get('columns', 3)
            positions = LayoutAlgorithms.grid_layout(num_components, columns)
        
        elif pattern == LayoutPattern.HERO_CONTENT:
            positions = LayoutAlgorithms.hero_content_layout(num_components - 1)
        
        elif pattern == LayoutPattern.RING:
            radius = kwargs.get('radius_level', 3)
            positions = LayoutAlgorithms.ring_layout(num_components, radius)
        
        elif pattern == LayoutPattern.SPIRAL:
            turns = kwargs.get('turns', 2.0)
            positions = LayoutAlgorithms.spiral_layout(num_components, turns)
        
        elif pattern == LayoutPattern.ARC:
            arc_angle = kwargs.get('arc_angle', 180.0)
            radius = kwargs.get('radius_level', 4)
            positions = LayoutAlgorithms.arc_layout(num_components, arc_angle, radius)
        
        elif pattern == LayoutPattern.MASONRY:
            columns = kwargs.get('columns', 3)
            height_pattern = kwargs.get('height_pattern', None)
            positions = LayoutAlgorithms.masonry_layout(num_components, columns, height_pattern)
        
        elif pattern == LayoutPattern.SIDEBAR:
            # Two sections: sidebar + main
            sidebar_pos, main_pos = LayoutAlgorithms.sidebar_layout()
            positions = [sidebar_pos, main_pos]
        
        else:  # CENTERED
            positions = [LayoutAlgorithms.centered_layout()]
        
        # Convert positions to LayoutTree objects
        layout_trees = []
        
        for component, _position in zip(components, positions):
            tree = LayoutTree(
                element_id=component.element_id,
                tag="div",  # Default tag
                layout_type=LayoutType.FLEX if pattern == LayoutPattern.HERO_CONTENT else LayoutType.GRID,
                parent_id=None,  # Simplified: All at root level
                children=[],
                padding_level=2,  # Default padding
                margin_level=1,  # Default margin
                gap_level=3,  # Default gap
            )
            layout_trees.append(tree)
        
        return layout_trees


def suggest_layout_for_components(components: List[Component]) -> LayoutPattern:
    """
    Suggest best layout pattern for given components
    
    Uses heuristics based on component types and count
    """
    num_components = len(components)
    
    if num_components == 0:
        return LayoutPattern.CENTERED
    
    # Check component types
    has_hero = any(c.component_type == ComponentType.HERO for c in components)
    has_nav = any(c.component_type == ComponentType.NAV for c in components)
    has_cards = len([c for c in components if c.component_type == ComponentType.CARD]) >= 3
    
    # Hero + other components → HERO_CONTENT
    if has_hero and num_components > 1:
        return LayoutPattern.HERO_CONTENT
    
    # Many cards → GRID or MASONRY
    if has_cards:
        return LayoutPattern.GRID
    
    # Navigation + content → SIDEBAR
    if has_nav and num_components >= 2:
        return LayoutPattern.SIDEBAR
    
    # Few components → CENTERED
    if num_components <= 2:
        return LayoutPattern.CENTERED
    
    # Default → GRID
    return LayoutPattern.GRID

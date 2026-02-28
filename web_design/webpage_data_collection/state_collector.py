"""
Webpage State Collector
Captures webpage states from running sites for training data

This module provides tools to:
1. Instrument websites to capture DOM + analytics
2. Extract state snapshots at regular intervals
3. Record edit sequences when designers make changes
4. Measure outcomes before/after edits
"""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .state_schema import (
    WebpageState, SiteSnapshot, EditRecord,
    LayoutTree, Component, StyleSheet, ComponentType, LayoutType,
    PerformanceMetrics, OutcomeMetrics, CognitiveLoadMetrics, Asset,
    EventHandler, Animation
)


class WebpageStateCollector:
    """
    Collects webpage state snapshots from instrumented sites
    
    Integration approaches:
    1. JavaScript instrumentation (client-side DOM capture)
    2. Headless browser automation (Playwright/Puppeteer)
    3. Analytics API integration (GA4, Mixpanel, Segment)
    """
    
    def __init__(self, storage_dir: str = "./collected_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def capture_snapshot(
        self,
        site_id: str,
        page_url: str,
        dom_data: Dict,
        analytics_data: Optional[Dict] = None,
        performance_data: Optional[Dict] = None
    ) -> WebpageState:
        """
        Capture a single snapshot of webpage state
        
        Args:
            site_id: Unique site identifier
            page_url: URL of the page
            dom_data: DOM structure from instrumentation
            analytics_data: Outcome metrics from analytics API
            performance_data: Lighthouse/CWV metrics
        
        Returns:
            WebpageState object
        """
        timestamp = datetime.utcnow().isoformat()
        page_id = self._generate_page_id(page_url)
        
        # Parse DOM into structured format
        layout = self._parse_layout(dom_data.get('dom', []))
        components = self._parse_components(dom_data.get('components', []))
        styling = self._parse_styling(dom_data.get('styles', {}))
        assets = self._parse_assets(dom_data.get('assets', []))
        events = self._parse_events(dom_data.get('events', []))
        animations = self._parse_animations(dom_data.get('animations', []))
        
        # Parse analytics outcomes
        outcomes = None
        if analytics_data:
            outcomes = self._parse_outcomes(analytics_data)
        
        # Parse performance metrics
        performance = None
        if performance_data:
            performance = self._parse_performance(performance_data)
        
        # Compute cognitive load metrics
        cognitive_load = self._compute_cognitive_load(layout, components, styling)
        
        # Build state object
        state = WebpageState(
            site_id=site_id,
            page_id=page_id,
            timestamp=timestamp,
            layout=layout,
            components=components,
            styling=styling,
            assets=assets,
            events=events,
            animations=animations,
            outcomes=outcomes,
            performance=performance,
            cognitive_load=cognitive_load,
            page_title=dom_data.get('title', ''),
            meta_description=dom_data.get('meta_description', ''),
            headline=dom_data.get('headline', ''),
            subheadline=dom_data.get('subheadline', ''),
            cta_primary=dom_data.get('cta_primary', ''),
            cta_secondary=dom_data.get('cta_secondary', ''),
            body_copy=dom_data.get('body_copy', {}),
        )
        
        # Save snapshot
        self._save_snapshot(state)
        
        return state
    
    def capture_edit(
        self,
        edit_type: str,
        target_element_id: Optional[str],
        edit_params: Dict,
        before_state: WebpageState,
        after_state: WebpageState,
        applied_by: str = "human"
    ) -> EditRecord:
        """
        Record an edit that was applied to the webpage
        
        Args:
            edit_type: Type of edit (add_component, modify_layout, etc.)
            target_element_id: Element that was edited
            edit_params: Parameters of the edit
            before_state: State before edit
            after_state: State after edit  
            applied_by: Who applied the edit (human, autonomous, assisted)
        
        Returns:
            EditRecord object
        """
        timestamp = datetime.utcnow().isoformat()
        edit_id = self._generate_edit_id(edit_type, timestamp)
        
        edit = EditRecord(
            edit_id=edit_id,
            timestamp=timestamp,
            edit_type=edit_type,
            target_element_id=target_element_id,
            edit_params=edit_params,
            before_state_id=f"{before_state.site_id}_{before_state.timestamp}",
            after_state_id=f"{after_state.site_id}_{after_state.timestamp}",
            applied_by=applied_by
        )
        
        # Save edit record
        self._save_edit(edit)
        
        return edit
    
    def measure_outcome_delta(
        self,
        edit: EditRecord,
        before_outcomes: OutcomeMetrics,
        after_outcomes: OutcomeMetrics,
        duration_hours: float = 24.0
    ) -> Dict[str, float]:
        """
        Measure the outcome change after an edit
        
        Waits for sufficient traffic (duration_hours) then compares metrics
        
        Args:
            edit: The edit to measure
            before_outcomes: Outcomes before edit
            after_outcomes: Outcomes after edit (measured after duration)
            duration_hours: How long to wait for measurement
        
        Returns:
            Dict of metric_name -> delta_value
        """
        deltas = {}
        
        # Conversion rate delta
        if before_outcomes.conversion_rate is not None and after_outcomes.conversion_rate is not None:
            deltas['conversion_rate'] = after_outcomes.conversion_rate - before_outcomes.conversion_rate
        
        # Engagement deltas
        if before_outcomes.avg_time_on_page is not None and after_outcomes.avg_time_on_page is not None:
            deltas['avg_time_on_page'] = after_outcomes.avg_time_on_page - before_outcomes.avg_time_on_page
        
        if before_outcomes.scroll_depth_avg is not None and after_outcomes.scroll_depth_avg is not None:
            deltas['scroll_depth_avg'] = after_outcomes.scroll_depth_avg - before_outcomes.scroll_depth_avg
        
        if before_outcomes.clicks_per_session is not None and after_outcomes.clicks_per_session is not None:
            deltas['clicks_per_session'] = after_outcomes.clicks_per_session - before_outcomes.clicks_per_session
        
        # Bounce rate delta (negative is good)
        if before_outcomes.bounce_rate is not None and after_outcomes.bounce_rate is not None:
            deltas['bounce_rate'] = after_outcomes.bounce_rate - before_outcomes.bounce_rate
        
        # Update edit record with outcomes
        edit.outcome_delta = deltas
        self._save_edit(edit)
        
        return deltas
    
    def collect_site_snapshot(
        self,
        site_id: str,
        domain: str,
        industry: str,
        duration_days: int = 30
    ) -> SiteSnapshot:
        """
        Collect a complete snapshot of a site over a period
        
        This is used for the initial Phase 1 data collection from 1000 sites
        
        Args:
            site_id: Unique identifier for the site
            domain: Website domain
            industry: Industry category (saas, ecommerce, etc.)
            duration_days: How many days to collect data
        
        Returns:
            SiteSnapshot with complete state and edit history
        """
        # Load all states and edits for this site
        states = self._load_states_for_site(site_id)
        edits = self._load_edits_for_site(site_id)
        
        snapshot = SiteSnapshot(
            site_id=site_id,
            domain=domain,
            industry=industry,
            states=states,
            edits=edits,
            total_snapshots=len(states),
            total_edits=len(edits),
            collection_start=states[0].timestamp if states else "",
            collection_end=states[-1].timestamp if states else "",
        )
        
        # Save combined snapshot
        snapshot_path = self.storage_dir / f"snapshots/{site_id}_snapshot.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize to JSON (would need custom encoder for dataclasses)
        # For now, placeholder
        with open(snapshot_path, 'w') as f:
            json.dump({'site_id': site_id, 'states': len(states), 'edits': len(edits)}, f)
        
        return snapshot
    
    # Private helper methods
    
    def _parse_layout(self, dom_list: List[Dict]) -> List[LayoutTree]:
        """Parse DOM data into LayoutTree objects"""
        layout_trees = []
        
        for elem in dom_list:
            layout = LayoutTree(
                element_id=elem.get('id', ''),
                tag=elem.get('tag', 'div'),
                layout_type=LayoutType(elem.get('layout_type', 'block')),
                parent_id=elem.get('parent_id'),
                children=elem.get('children', []),
                is_container=elem.get('is_container', False),
                columns=elem.get('columns'),
                flex_direction=elem.get('flex_direction'),
                padding_level=elem.get('padding_level', 2),
                margin_level=elem.get('margin_level', 2),
                gap_level=elem.get('gap_level', 1),
            )
            layout_trees.append(layout)
        
        return layout_trees
    
    def _parse_components(self, comp_list: List[Dict]) -> List[Component]:
        """Parse component data into Component objects"""
        components = []
        
        for comp in comp_list:
            component = Component(
                component_id=comp.get('id', ''),
                component_type=ComponentType(comp.get('type', 'text_block')),
                element_id=comp.get('element_id', ''),
                text_content=comp.get('text'),
                emphasis_level=comp.get('emphasis_level', 3),
                color_role=comp.get('color_role', 'neutral'),
                size_level=comp.get('size_level', 3),
                clickable=comp.get('clickable', False),
                href=comp.get('href'),
            )
            components.append(component)
        
        return components
    
    def _parse_styling(self, style_dict: Dict) -> Optional[StyleSheet]:
        """Parse style data into StyleSheet object"""
        if not style_dict:
            return None
        
        return StyleSheet(
            primary_color=style_dict.get('primary_color', '#000000'),
            secondary_color=style_dict.get('secondary_color', '#666666'),
            accent_color=style_dict.get('accent_color', '#0066cc'),
            background_color=style_dict.get('background_color', '#ffffff'),
            text_color=style_dict.get('text_color', '#000000'),
            font_family_heading=style_dict.get('font_family_heading', 'sans-serif'),
            font_family_body=style_dict.get('font_family_body', 'sans-serif'),
            base_font_size=style_dict.get('base_font_size', 16),
            line_height=style_dict.get('line_height', 1.5),
            border_radius=style_dict.get('border_radius', 4),
        )
    
    def _parse_assets(self, asset_list: List[Dict]) -> List[Asset]:
        """Parse asset data"""
        return [
            Asset(
                asset_id=a.get('id', ''),
                asset_type=a.get('type', 'image'),
                url=a.get('url', ''),
                alt_text=a.get('alt'),
                file_size_kb=a.get('size_kb'),
                optimized=a.get('optimized', False),
                lazy_loaded=a.get('lazy_loaded', False),
            )
            for a in asset_list
        ]
    
    def _parse_events(self, event_list: List[Dict]) -> List[EventHandler]:
        """Parse event handler data"""
        return [
            EventHandler(
                event_id=e.get('id', ''),
                element_id=e.get('element_id', ''),
                event_type=e.get('type', 'click'),
                action=e.get('action', ''),
                tracked=e.get('tracked', False),
                track_as=e.get('track_as'),
            )
            for e in event_list
        ]
    
    def _parse_animations(self, anim_list: List[Dict]) -> List[Animation]:
        """Parse animation data"""
        return [
            Animation(
                animation_id=a.get('id', ''),
                element_id=a.get('element_id', ''),
                trigger=a.get('trigger', 'on_load'),
                effect=a.get('effect', 'fade_in'),
                duration_ms=a.get('duration_ms', 300),
            )
            for a in anim_list
        ]
    
    def _parse_outcomes(self, analytics_dict: Dict) -> OutcomeMetrics:
        """Parse analytics data into OutcomeMetrics"""
        return OutcomeMetrics(
            conversion_rate=analytics_dict.get('conversion_rate'),
            conversion_count=analytics_dict.get('conversion_count'),
            conversion_goal=analytics_dict.get('conversion_goal'),
            avg_time_on_page=analytics_dict.get('avg_time_on_page'),
            scroll_depth_avg=analytics_dict.get('scroll_depth_avg'),
            clicks_per_session=analytics_dict.get('clicks_per_session'),
            bounce_rate=analytics_dict.get('bounce_rate'),
            exit_rate=analytics_dict.get('exit_rate'),
            unique_visitors=analytics_dict.get('unique_visitors'),
            pageviews=analytics_dict.get('pageviews'),
            measurement_start=analytics_dict.get('measurement_start'),
            measurement_end=analytics_dict.get('measurement_end'),
            measurement_duration_hours=analytics_dict.get('measurement_duration_hours'),
        )
    
    def _parse_performance(self, perf_dict: Dict) -> PerformanceMetrics:
        """Parse performance data into PerformanceMetrics"""
        return PerformanceMetrics(
            performance_score=perf_dict.get('performance_score'),
            accessibility_score=perf_dict.get('accessibility_score'),
            seo_score=perf_dict.get('seo_score'),
            best_practices_score=perf_dict.get('best_practices_score'),
            lcp=perf_dict.get('lcp'),
            fid=perf_dict.get('fid'),
            cls=perf_dict.get('cls'),
            page_size_kb=perf_dict.get('page_size_kb'),
            load_time_seconds=perf_dict.get('load_time_seconds'),
            mobile_friendly=perf_dict.get('mobile_friendly', True),
            responsive_design=perf_dict.get('responsive_design', True),
        )
    
    def _compute_cognitive_load(
        self,
        layout: List[LayoutTree],
        components: List[Component],
        styling: Optional[StyleSheet]
    ) -> CognitiveLoadMetrics:
        """
        Compute cognitive load metrics from state
        
        These are heuristics based on design principles
        """
        element_count = len(layout)
        
        # Visual clutter: more elements = higher clutter
        visual_clutter_score = min(1.0, element_count / 100.0)
        
        # Color variety: count unique colors in components
        colors_used = set()
        for comp in components:
            colors_used.add(comp.color_role)
        color_variety_score = min(1.0, len(colors_used) / 5.0)
        
        # Hierarchy depth: max nesting level
        hierarchy_depth = self._compute_max_depth(layout)
        
        # Emphasis distribution: histogram of emphasis levels
        emphasis_distribution = [0] * 6
        for comp in components:
            emphasis_distribution[comp.emphasis_level] += 1
        
        # CTA visibility: average emphasis of CTA components
        cta_components = [c for c in components if c.component_type == ComponentType.CTA]
        cta_visibility = sum(c.emphasis_level for c in cta_components) / max(len(cta_components), 1) / 5.0
        
        return CognitiveLoadMetrics(
            element_count=element_count,
            visual_clutter_score=visual_clutter_score,
            color_variety_score=color_variety_score,
            hierarchy_depth=hierarchy_depth,
            emphasis_distribution=emphasis_distribution,
            cta_visibility=cta_visibility,
            navigation_complexity=0.5,  # Placeholder
            content_density=0.5,  # Placeholder
            f_pattern_score=0.7,  # Placeholder
        )
    
    def _compute_max_depth(self, layout: List[LayoutTree]) -> int:
        """Compute maximum nesting depth of layout tree"""
        # Build parent-child map
        children_map = {}
        for elem in layout:
            if elem.parent_id:
                if elem.parent_id not in children_map:
                    children_map[elem.parent_id] = []
                children_map[elem.parent_id].append(elem.element_id)
        
        # Find roots (no parent)
        roots = [e.element_id for e in layout if e.parent_id is None]
        
        # DFS to find max depth
        def dfs(node_id: str, depth: int) -> int:
            if node_id not in children_map:
                return depth
            return max(dfs(child, depth + 1) for child in children_map[node_id])
        
        if not roots:
            return 0
        
        return max(dfs(root, 1) for root in roots)
    
    def _generate_page_id(self, url: str) -> str:
        """Generate unique page ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def _generate_edit_id(self, edit_type: str, timestamp: str) -> str:
        """Generate unique edit ID"""
        combined = f"{edit_type}_{timestamp}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def _save_snapshot(self, state: WebpageState):
        """Save state snapshot to disk"""
        state_dir = self.storage_dir / f"states/{state.site_id}"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = state_dir / f"{state.timestamp.replace(':', '-')}.json"
        
        # Placeholder: Would need proper serialization
        with open(state_file, 'w') as f:
            json.dump({'site_id': state.site_id, 'timestamp': state.timestamp}, f)
    
    def _save_edit(self, edit: EditRecord):
        """Save edit record to disk"""
        edit_dir = self.storage_dir / "edits"
        edit_dir.mkdir(parents=True, exist_ok=True)
        
        edit_file = edit_dir / f"{edit.edit_id}.json"
        
        # Placeholder: Would need proper serialization
        with open(edit_file, 'w') as f:
            json.dump({'edit_id': edit.edit_id, 'edit_type': edit.edit_type}, f)
    
    def _load_states_for_site(self, site_id: str) -> List[WebpageState]:
        """Load all states for a site"""
        # Placeholder: Would load from disk and deserialize
        return []
    
    def _load_edits_for_site(self, site_id: str) -> List[EditRecord]:
        """Load all edits for a site"""
        # Placeholder: Would load from disk and deserialize
        return []

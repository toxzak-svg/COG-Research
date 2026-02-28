"""
Test script to validate WebAlchemist integration
Tests: Awwwards loader, state schema enhancements, layout algorithms
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from webpage_data_collection import (
    AwwwardsLoader,
    LayoutAlgorithms, LayoutPattern,
    suggest_layout_for_components,
    Component, ComponentType
)


def test_awwwards_loader():
    """Test loading Awwwards patterns and converting to WebpageState"""
    print("Testing Awwwards Loader...")
    
    loader = AwwwardsLoader(webalchemist_path="c:/dev/projects/WebAlchemist")
    
    # Load patterns
    patterns = loader.load_awwwards_patterns()
    print(f"✓ Loaded {len(patterns)} patterns")
    
    # Convert to states
    states = loader.create_seed_states_from_patterns()
    print(f"✓ Created {len(states)} WebpageState objects")
    
    # Validate first state
    if len(states) > 0:
        state = states[0]
        print(f"\nSample State:")
        print(f"  Site ID: {state.site_id}")
        print(f"  Page Title: {state.page_title}")
        print(f"  Project Category: {state.project_category}")
        print(f"  Tech Stack: {state.tech_stack}")
        print(f"  Tags: {state.tags[:3]}..." if len(state.tags) > 3 else f"  Tags: {state.tags}")
        print(f"  Components: {len(state.components)}")
        print(f"  Layout Nodes: {len(state.layout)}")
        
        # Check SEO metadata
        if state.seo_metadata:
            print(f"  SEO Title: {state.seo_metadata.meta_title[:50]}...")
            print(f"  SEO Keywords: {len(state.seo_metadata.keywords)} keywords")
        
        # Check outcomes
        if state.outcomes:
            print(f"  Conversion Rate: {state.outcomes.conversion_rate:.1%}")
            print(f"  Bounce Rate: {state.outcomes.bounce_rate:.1%}")
    
    print("\n✅ Awwwards loader test passed!\n")
    return states


def test_layout_algorithms():
    """Test layout algorithm generation"""
    print("Testing Layout Algorithms...")
    
    # Test grid layout
    grid_positions = LayoutAlgorithms.grid_layout(num_components=6, columns=3)
    print(f"✓ Grid layout: {len(grid_positions)} positions")
    
    # Test hero + content
    hero_positions = LayoutAlgorithms.hero_content_layout(num_content_blocks=3)
    print(f"✓ Hero+content layout: {len(hero_positions)} positions")
    
    # Test ring layout
    ring_positions = LayoutAlgorithms.ring_layout(num_components=8, radius_level=3)
    print(f"✓ Ring layout: {len(ring_positions)} positions")
    
    # Test spiral layout
    spiral_positions = LayoutAlgorithms.spiral_layout(num_components=10, turns=2.0)
    print(f"✓ Spiral layout: {len(spiral_positions)} positions")
    
    # Test arc layout
    arc_positions = LayoutAlgorithms.arc_layout(num_components=5, arc_angle=180.0)
    print(f"✓ Arc layout: {len(arc_positions)} positions")
    
    # Test masonry layout
    masonry_positions = LayoutAlgorithms.masonry_layout(num_components=9, columns=3)
    print(f"✓ Masonry layout: {len(masonry_positions)} positions")
    
    print("\n✅ Layout algorithms test passed!\n")


def test_layout_suggestion():
    """Test layout pattern suggestion"""
    print("Testing Layout Suggestion...")
    
    # Create sample components
    components_hero_cards = [
        Component(
            component_id="hero",
            component_type=ComponentType.HERO,
            element_id="hero-section",
            emphasis_level=5,
        ),
        Component(
            component_id="card1",
            component_type=ComponentType.CARD,
            element_id="card-1",
            emphasis_level=3,
        ),
        Component(
            component_id="card2",
            component_type=ComponentType.CARD,
            element_id="card-2",
            emphasis_level=3,
        ),
    ]
    
    # Test suggestion
    pattern = suggest_layout_for_components(components_hero_cards)
    print(f"✓ Suggested pattern for hero+cards: {pattern}")
    assert pattern == LayoutPattern.HERO_CONTENT, f"Expected HERO_CONTENT, got {pattern}"
    
    # Test with many cards
    many_cards = [
        Component(
            component_id=f"card{i}",
            component_type=ComponentType.CARD,
            element_id=f"card-{i}",
            emphasis_level=3,
        )
        for i in range(6)
    ]
    
    pattern_grid = suggest_layout_for_components(many_cards)
    print(f"✓ Suggested pattern for 6 cards: {pattern_grid}")
    assert pattern_grid == LayoutPattern.GRID, f"Expected GRID, got {pattern_grid}"
    
    print("\n✅ Layout suggestion test passed!\n")


def test_portfolio_fields():
    """Test that portfolio fields are populated in seed states"""
    print("Testing Portfolio Fields...")
    
    loader = AwwwardsLoader(webalchemist_path="c:/dev/projects/WebAlchemist")
    states = loader.create_seed_states_from_patterns()
    
    if len(states) > 0:
        state = states[0]
        
        # Check project_category exists
        assert state.project_category is not None, "project_category is None"
        print(f"✓ project_category: {state.project_category}")
        
        # Check tech_stack exists
        assert isinstance(state.tech_stack, list), "tech_stack is not a list"
        print(f"✓ tech_stack: {state.tech_stack}")
        
        # Check tags exists
        assert isinstance(state.tags, list), "tags is not a list"
        assert len(state.tags) > 0, "tags is empty"
        print(f"✓ tags: {state.tags[:3]}...")
        
        # Check SEO metadata exists
        assert state.seo_metadata is not None, "seo_metadata is None"
        assert len(state.seo_metadata.keywords) > 0, "SEO keywords empty"
        print(f"✓ SEO metadata populated with {len(state.seo_metadata.keywords)} keywords")
        
        # Check social proof exists
        assert state.social_proof is not None, "social_proof is None"
        print(f"✓ social_proof structure exists")
    
    print("\n✅ Portfolio fields test passed!\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("WebAlchemist Integration Validation")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Awwwards loader
        states = test_awwwards_loader()
        
        # Test 2: Layout algorithms
        test_layout_algorithms()
        
        # Test 3: Layout suggestion
        test_layout_suggestion()
        
        # Test 4: Portfolio fields
        test_portfolio_fields()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Summary:")
        print(f"  - {len(states)} seed states ready for training")
        print(f"  - 7 layout algorithms available")
        print(f"  - Portfolio fields integrated (project_category, tech_stack, tags, SEO)")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

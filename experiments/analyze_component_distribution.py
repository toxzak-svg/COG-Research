"""
Analyze component count distribution in seed data
to understand why contrastive loss was 0.0
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from webpage_data_collection.awwwards_loader import load_seed_data
from collections import Counter
import json

def analyze_distribution(webalchemist_path: str):
    """Analyze component count distribution"""
    
    # Load seed data
    print("Loading Awwwards seed data...")
    states = load_seed_data(webalchemist_path)
    print(f"Loaded {len(states)} states\n")
    
    # Extract component counts
    component_counts = []
    for state in states:
        num_components = len(state.components) if state.components else 0
        component_counts.append(num_components)
    
    # Statistics
    counter = Counter(component_counts)
    
    print("="*60)
    print("Component Count Distribution")
    print("="*60)
    print(f"\nTotal states: {len(states)}")
    print(f"Min components: {min(component_counts)}")
    print(f"Max components: {max(component_counts)}")
    print(f"Mean components: {sum(component_counts)/len(component_counts):.2f}")
    print(f"\nDistribution:")
    for count, freq in sorted(counter.items()):
        bar = "█" * freq
        print(f"  {count:2d} components: {freq:2d} states {bar}")
    
    # Analyze contrastive pairs
    print("\n" + "="*60)
    print("Contrastive Pair Analysis")
    print("="*60)
    
    # Count positive pairs (diff <= 1)
    positive_pairs = 0
    negative_pairs = 0
    
    for i in range(len(component_counts)):
        for j in range(i+1, len(component_counts)):
            diff = abs(component_counts[i] - component_counts[j])
            if diff <= 1:
                positive_pairs += 1
            else:
                negative_pairs += 1
    
    total_pairs = len(component_counts) * (len(component_counts) - 1) // 2
    
    print(f"\nWith similarity metric: |diff| <= 1")
    print(f"  Positive pairs (similar): {positive_pairs} ({100*positive_pairs/total_pairs:.1f}%)")
    print(f"  Negative pairs (dissimilar): {negative_pairs} ({100*negative_pairs/total_pairs:.1f}%)")
    print(f"  Total pairs: {total_pairs}")
    
    if negative_pairs < 10:
        print(f"\n⚠️  ISSUE CONFIRMED: Only {negative_pairs} negative pairs!")
        print("   Contrastive loss will be ineffective with so few negatives.")
    
    # Check with different thresholds
    print(f"\n" + "="*60)
    print("Alternative Similarity Thresholds")
    print("="*60)
    
    for threshold in [0, 2, 3]:
        pos = sum(1 for i in range(len(component_counts)) 
                  for j in range(i+1, len(component_counts))
                  if abs(component_counts[i] - component_counts[j]) <= threshold)
        neg = total_pairs - pos
        print(f"  |diff| <= {threshold}: {pos} positive ({100*pos/total_pairs:.1f}%), {neg} negative ({100*neg/total_pairs:.1f}%)")
    
    print("\n" + "="*60)
    print("Recommendation")
    print("="*60)
    
    if negative_pairs < 20:
        print("\n✗ Current contrastive metric (|diff| <= 1) is too narrow")
        print("  → Almost all pairs are positive, no diversity to learn from")
        print("  → This explains why contrastive loss was 0.0 every epoch")
        print("\n✓ Solution: Use aggressive β-TC without contrastive loss")
        print("  → β_tc: 20-30 (much stronger than current 10)")
        print("  → latent_dim: 64 (more capacity)")
        print("  → free_bits: 1.0 (force information retention)")
    else:
        print("\n✓ Contrastive metric has sufficient diversity")
        print("  → Issue must be elsewhere")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--webalchemist-path", required=True)
    args = parser.parse_args()
    
    analyze_distribution(args.webalchemist_path)

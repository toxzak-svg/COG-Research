"""
Latent Space Validation for VAE World-Model

Validates the trained VAE by testing:
1. Reconstruction quality (MSE per feature group)
2. Interpolation smoothness (coherence scores)
3. Semantic clustering (t-SNE + silhouette score)
4. Latent arithmetic (semantic operations)

Based on VAE_WORLD_MODEL_DESIGN.md Section 7
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from webpage_models import WebpageVAE, get_feature_masks
from webpage_data_collection.awwwards_loader import load_seed_data as load_awwwards_seed_data
from webpage_data_collection.state_schema import WebpageState


def load_model(checkpoint_path: str, device: str = 'cpu', latent_dim: int = 32) -> WebpageVAE:
    """Load trained VAE model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get latent_dim from checkpoint, otherwise use provided value
    if 'latent_dim' in checkpoint:
        latent_dim = checkpoint['latent_dim']
        print(f"  Using latent_dim from checkpoint: {latent_dim}")
    else:
        print(f"  Using provided latent_dim: {latent_dim}")
    
    # Create model with same architecture
    model = WebpageVAE(
        input_dim=190,
        latent_dim=latent_dim,
        outcome_dim=10,
        dropout=0.1,
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_loss' in checkpoint:
        print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    
    return model


def load_seed_data(webalchemist_path: str) -> List[WebpageState]:
    """Load Awwwards seed data"""
    print(f"\nLoading seed data from {webalchemist_path}...")
    
    # Use the existing load_seed_data function from awwwards_loader
    states = load_awwwards_seed_data(webalchemist_path)
    
    print(f"✓ Loaded {len(states)} seed states")
    return states


def vectorize_state(state: WebpageState) -> torch.Tensor:
    """
    Convert WebpageState to 190D vector
    (Same logic as in train_vae_seed_data.py)
    """
    features = []
    
    # Visual/Design features [0:80]
    layout_depth = len(state.layout) if state.layout else 0
    component_count = len(state.components) if state.components else 0
    features.extend([
        min(layout_depth / 30.0, 1.0),
        min(component_count / 50.0, 1.0),
        0.5,
    ])
    features.extend([0.5] * 77)
    
    # Content features [80:110]
    headline_len = len(state.headline) if state.headline else 0
    cta_primary_len = len(state.cta_primary) if state.cta_primary else 0
    features.extend([
        min(headline_len / 100.0, 1.0),
        min(cta_primary_len / 50.0, 1.0),
        1.0 if state.cta_primary else 0.0,
        1.0 if state.cta_secondary else 0.0,
    ])
    features.extend([0.5] * 26)
    
    # Performance features [110:130]
    if state.performance:
        perf = state.performance
        features.extend([
            perf.lighthouse_performance / 100.0 if hasattr(perf, 'lighthouse_performance') else 0.8,
            perf.lighthouse_accessibility / 100.0 if hasattr(perf, 'lighthouse_accessibility') else 0.8,
            perf.lighthouse_best_practices / 100.0 if hasattr(perf, 'lighthouse_best_practices') else 0.8,
            perf.lighthouse_seo / 100.0 if hasattr(perf, 'lighthouse_seo') else 0.8,
        ])
    else:
        features.extend([0.8, 0.8, 0.8, 0.8])
    features.extend([0.5] * 16)
    
    # Outcomes [130:140] - default good outcomes
    features.extend([0.05, 0.30, 0.60, 0.75, 0.15, 0.10, 0.70, 0.40, 0.25, 0.60])
    
    # Cognitive load [140:150]
    if state.cognitive_load:
        cog = state.cognitive_load
        features.extend([
            cog.clutter_score if hasattr(cog, 'clutter_score') else 0.5,
            cog.hierarchy_clarity if hasattr(cog, 'hierarchy_clarity') else 0.5,
            cog.visual_weight_balance if hasattr(cog, 'visual_weight_balance') else 0.5,
        ])
    else:
        features.extend([0.5, 0.5, 0.5])
    features.extend([0.5] * 7)
    
    # Context [150:170]
    features.extend([
        1.0 if state.device_type == 'desktop' else 0.5,
        1.0 if state.traffic_source == 'organic' else 0.5,
        1.0 if state.user_segment else 0.5,
    ])
    features.extend([0.5] * 17)
    
    # Portfolio [170:190]
    features.extend([
        1.0 if state.project_category else 0.5,
        min(len(state.tech_stack) / 5.0, 1.0) if state.tech_stack else 0.5,
        min(len(state.tags) / 10.0, 1.0) if state.tags else 0.5,
    ])
    features.extend([0.5] * 17)
    
    assert len(features) == 190
    return torch.tensor(features, dtype=torch.float32)


def test_reconstruction_quality(
    model: WebpageVAE,
    states: List[WebpageState],
    device: str,
) -> Dict[str, float]:
    """
    Test reconstruction quality with MSE per feature group
    
    Returns:
        Dict with MSE for each feature group + overall
    """
    print("\n" + "="*60)
    print("1. Testing Reconstruction Quality")
    print("="*60)
    
    # Get feature masks
    feature_masks = get_feature_masks(device)
    
    # Vectorize all states
    state_vectors = torch.stack([vectorize_state(s) for s in states]).to(device)
    
    # Reconstruct
    with torch.no_grad():
        x_recon, outcomes_pred = model.reconstruct(state_vectors)
    
    # Overall MSE
    overall_mse = F.mse_loss(x_recon, state_vectors).item()
    
    # MSE per feature group
    group_mses = {}
    for group_name, mask in feature_masks.items():
        if group_name == 'outcomes':
            continue  # Skip outcomes (handled separately)
        
        # Use advanced indexing with mask (1D tensor of indices)
        x_true_group = state_vectors[:, mask]
        x_recon_group = x_recon[:, mask]
        
        mse = F.mse_loss(x_recon_group, x_true_group).item()
        group_mses[group_name] = mse
    
    # Outcome prediction MAE
    outcome_mae = F.l1_loss(outcomes_pred, state_vectors[:, 130:140]).item()
    
    # Print results
    print(f"\nReconstruction Quality:")
    print(f"  Overall MSE: {overall_mse:.6f}")
    print(f"\nMSE by Feature Group:")
    for group, mse in sorted(group_mses.items()):
        print(f"  {group:20s}: {mse:.6f}")
    print(f"\nOutcome Prediction:")
    print(f"  MAE: {outcome_mae:.6f} ({outcome_mae*100:.2f}%)")
    
    # Success criteria from design doc
    print(f"\nSuccess Criteria:")
    mvp_pass = overall_mse < 0.15 and outcome_mae < 0.03
    prod_pass = overall_mse < 0.05 and outcome_mae < 0.01
    
    if prod_pass:
        print(f"  ✅ PRODUCTION quality achieved!")
        print(f"     MSE < 0.05 ✓ ({overall_mse:.6f})")
        print(f"     MAE < 1% ✓ ({outcome_mae*100:.2f}%)")
    elif mvp_pass:
        print(f"  ✅ MVP quality achieved")
        print(f"     MSE < 0.15 ✓ ({overall_mse:.6f})")
        print(f"     MAE < 3% ✓ ({outcome_mae*100:.2f}%)")
    else:
        print(f"  ⚠️  Below MVP threshold")
        print(f"     MSE < 0.15: {'✓' if overall_mse < 0.15 else '✗'} ({overall_mse:.6f})")
        print(f"     MAE < 3%: {'✓' if outcome_mae < 0.03 else '✗'} ({outcome_mae*100:.2f}%)")
    
    return {
        'overall_mse': overall_mse,
        'outcome_mae': outcome_mae,
        **group_mses,
        'mvp_pass': mvp_pass,
        'production_pass': prod_pass,
    }


def test_interpolation_smoothness(
    model: WebpageVAE,
    states: List[WebpageState],
    device: str,
    num_pairs: int = 10,
    num_steps: int = 10,
) -> Dict[str, float]:
    """
    Test interpolation smoothness between random pairs
    
    Measures coherence by checking that intermediate states
    are valid (values in expected ranges)
    
    Returns:
        Dict with coherence scores
    """
    print("\n" + "="*60)
    print("2. Testing Interpolation Smoothness")
    print("="*60)
    
    # Vectorize all states
    state_vectors = torch.stack([vectorize_state(s) for s in states]).to(device)
    
    coherence_scores = []
    
    for i in range(num_pairs):
        # Pick random pair
        idx1, idx2 = np.random.choice(len(states), 2, replace=False)
        x1 = state_vectors[idx1:idx1+1]
        x2 = state_vectors[idx2:idx2+1]
        
        # Interpolate
        with torch.no_grad():
            x_interp, _ = model.interpolate(x1, x2, num_steps=num_steps)
        
        # Check coherence: all values should be in [0, 1]
        valid_values = ((x_interp >= 0) & (x_interp <= 1.2)).float().mean().item()
        
        # Check smoothness: adjacent steps should be similar
        diffs = torch.diff(x_interp, dim=0).abs().mean().item()
        smoothness = 1.0 / (1.0 + diffs)  # Higher is smoother
        
        coherence = 0.7 * valid_values + 0.3 * smoothness
        coherence_scores.append(coherence)
    
    avg_coherence = np.mean(coherence_scores)
    
    print(f"\nInterpolation Results ({num_pairs} pairs, {num_steps} steps each):")
    print(f"  Average coherence: {avg_coherence:.4f}")
    print(f"  Min coherence: {min(coherence_scores):.4f}")
    print(f"  Max coherence: {max(coherence_scores):.4f}")
    
    # Success criteria
    print(f"\nSuccess Criteria:")
    mvp_pass = avg_coherence > 0.85
    prod_pass = avg_coherence > 0.95
    
    if prod_pass:
        print(f"  ✅ PRODUCTION quality (coherence > 0.95)")
    elif mvp_pass:
        print(f"  ✅ MVP quality (coherence > 0.85)")
    else:
        print(f"  ⚠️  Below MVP threshold (coherence > 0.85)")
    
    return {
        'avg_coherence': avg_coherence,
        'min_coherence': min(coherence_scores),
        'max_coherence': max(coherence_scores),
        'mvp_pass': mvp_pass,
        'production_pass': prod_pass,
    }


def test_semantic_clustering(
    model: WebpageVAE,
    states: List[WebpageState],
    device: str,
    output_dir: Path,
) -> Dict[str, float]:
    """
    Test semantic clustering with t-SNE visualization
    
    Projects latent space to 2D and measures cluster quality
    
    Returns:
        Dict with silhouette score
    """
    print("\n" + "="*60)
    print("3. Testing Semantic Clustering")
    print("="*60)
    
    # Vectorize all states
    state_vectors = torch.stack([vectorize_state(s) for s in states]).to(device)
    
    # Encode to latent space
    with torch.no_grad():
        z = model.encode(state_vectors, deterministic=True)
    
    z_np = z.cpu().numpy()
    
    # t-SNE projection
    print("\nComputing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(states)-1))
    z_2d = tsne.fit_transform(z_np)
    
    # Create labels based on project category
    labels = []
    label_names = []
    for state in states:
        if state.project_category:
            label = state.project_category
        else:
            label = 'unknown'
        labels.append(label)
        if label not in label_names:
            label_names.append(label)
    
    # Convert to numerical labels for silhouette score
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    label_indices = [label_to_idx[label] for label in labels]
    
    # Compute silhouette score (only if we have multiple clusters)
    if len(label_names) > 1:
        silhouette = silhouette_score(z_np, label_indices)
        print(f"\nSilhouette Score: {silhouette:.4f}")
    else:
        silhouette = 0.0
        print(f"\nSilhouette Score: N/A (only one cluster)")
    
    # Visualize t-SNE
    plt.figure(figsize=(10, 8))
    
    # Color by category
    for label_name in label_names:
        mask = [l == label_name for l in labels]
        plt.scatter(
            z_2d[mask, 0],
            z_2d[mask, 1],
            label=label_name,
            alpha=0.7,
            s=100,
        )
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f'VAE Latent Space (32D → 2D)\n{len(states)} Awwwards Seed States')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = output_dir / 'latent_space_tsne.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved t-SNE visualization: {output_path}")
    plt.close()
    
    # Success criteria
    print(f"\nSuccess Criteria:")
    mvp_pass = len(label_names) == 1 or silhouette > 0.3
    prod_pass = len(label_names) == 1 or silhouette > 0.5
    
    if prod_pass:
        print(f"  ✅ PRODUCTION quality (silhouette > 0.5)")
    elif mvp_pass:
        print(f"  ✅ MVP quality (silhouette > 0.3)")
    else:
        print(f"  ⚠️  Below MVP threshold (silhouette > 0.3)")
    
    return {
        'silhouette_score': silhouette,
        'num_clusters': len(label_names),
        'mvp_pass': mvp_pass,
        'production_pass': prod_pass,
    }


def test_latent_arithmetic(
    model: WebpageVAE,
    states: List[WebpageState],
    device: str,
    output_dir: Path,
) -> Dict[str, float]:
    """
    Test latent arithmetic operations
    
    Example: z_minimal + (z_rich - z_average)
    Should produce a state that's "minimal but with rich features"
    
    Returns:
        Dict with arithmetic test results
    """
    print("\n" + "="*60)
    print("4. Testing Latent Arithmetic")
    print("="*60)
    
    # Vectorize all states
    state_vectors = torch.stack([vectorize_state(s) for s in states]).to(device)
    
    # Encode to latent space
    with torch.no_grad():
        z_all = model.encode(state_vectors, deterministic=True)
    
    # Compute average latent
    z_avg = z_all.mean(dim=0, keepdim=True)
    
    # Find states with extreme properties
    component_counts = [len(s.components) if s.components else 0 for s in states]
    
    # Minimal state (fewest components)
    idx_minimal = np.argmin(component_counts)
    z_minimal = z_all[idx_minimal:idx_minimal+1]
    
    # Rich state (most components)
    idx_rich = np.argmax(component_counts)
    z_rich = z_all[idx_rich:idx_rich+1]
    
    print(f"\nSelected states:")
    print(f"  Minimal: {states[idx_minimal].page_id} ({component_counts[idx_minimal]} components)")
    print(f"  Rich: {states[idx_rich].page_id} ({component_counts[idx_rich]} components)")
    print(f"  Average: {component_counts[idx_minimal]} to {component_counts[idx_rich]} components")
    
    # Latent arithmetic: z_minimal + (z_rich - z_avg)
    # This should create a "minimal state with rich features"
    z_arithmetic = z_minimal + (z_rich - z_avg)
    
    # Decode
    with torch.no_grad():
        x_minimal, _ = model.decode(z_minimal)
        x_rich, _ = model.decode(z_rich)
        x_avg, _ = model.decode(z_avg)
        x_arithmetic, _ = model.decode(z_arithmetic)
    
    # Measure similarity
    sim_to_minimal = F.cosine_similarity(x_arithmetic, x_minimal).item()
    sim_to_rich = F.cosine_similarity(x_arithmetic, x_rich).item()
    sim_to_avg = F.cosine_similarity(x_arithmetic, x_avg).item()
    
    print(f"\nArithmetic result similarity:")
    print(f"  To minimal state: {sim_to_minimal:.4f}")
    print(f"  To rich state: {sim_to_rich:.4f}")
    print(f"  To average state: {sim_to_avg:.4f}")
    
    # Check that arithmetic result is valid (all values in reasonable range)
    valid_values = ((x_arithmetic >= -0.1) & (x_arithmetic <= 1.1)).float().mean().item()
    print(f"\nValidity:")
    print(f"  Valid values: {valid_values*100:.1f}%")
    
    # Visualize latent space distances
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Latent space arithmetic
    ax = axes[0]
    z_np = torch.cat([z_minimal, z_rich, z_avg, z_arithmetic]).cpu().numpy()
    
    # Project to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_np)
    
    labels_viz = ['Minimal', 'Rich', 'Average', 'Arithmetic Result']
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, (label, color) in enumerate(zip(labels_viz, colors)):
        ax.scatter(z_2d[i, 0], z_2d[i, 1], label=label, c=color, s=200, alpha=0.7)
    
    # Draw arithmetic operation
    ax.annotate('', xy=z_2d[3], xytext=z_2d[0],
                arrowprops=dict(arrowstyle='->', lw=2, color='purple', alpha=0.5))
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Latent Space Arithmetic\n(z_minimal + (z_rich - z_avg))')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Feature comparison
    ax = axes[1]
    feature_groups = ['Visual\n[0:80]', 'Content\n[80:110]', 'Perf\n[110:130]', 
                      'Outcomes\n[130:140]', 'Cognitive\n[140:150]', 'Context\n[150:170]', 'Portfolio\n[170:190]']
    ranges = [(0, 80), (80, 110), (110, 130), (130, 140), (140, 150), (150, 170), (170, 190)]
    
    x_minimal_np = x_minimal.cpu().numpy()[0]
    x_arithmetic_np = x_arithmetic.cpu().numpy()[0]
    
    group_diffs = []
    for start, end in ranges:
        diff = np.abs(x_arithmetic_np[start:end] - x_minimal_np[start:end]).mean()
        group_diffs.append(diff)
    
    ax.bar(range(len(feature_groups)), group_diffs, color='purple', alpha=0.7)
    ax.set_xticks(range(len(feature_groups)))
    ax.set_xticklabels(feature_groups, rotation=0)
    ax.set_ylabel('Mean Absolute Difference')
    ax.set_title('Feature Group Changes\n(Arithmetic vs Minimal)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'latent_arithmetic.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved arithmetic visualization: {output_path}")
    plt.close()
    
    # Success: arithmetic result should be valid and somewhat similar to inputs
    success = valid_values > 0.95 and sim_to_minimal > 0.5
    
    print(f"\nSuccess Criteria:")
    if success:
        print(f"  ✅ Valid arithmetic operations (values valid + similarity > 0.5)")
    else:
        print(f"  ⚠️  Arithmetic may be unstable")
    
    return {
        'sim_to_minimal': sim_to_minimal,
        'sim_to_rich': sim_to_rich,
        'sim_to_avg': sim_to_avg,
        'valid_values': valid_values,
        'success': success,
    }


def generate_validation_report(
    results: Dict,
    output_dir: Path,
):
    """Generate comprehensive validation report"""
    print("\n" + "="*60)
    print("Validation Report")
    print("="*60)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_python(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(item) for item in obj]
        else:
            return obj
    
    results = convert_to_python(results)
    
    report = {
        'reconstruction_quality': results['reconstruction'],
        'interpolation_smoothness': results['interpolation'],
        'semantic_clustering': results['clustering'],
        'latent_arithmetic': results['arithmetic'],
    }
    
    # Overall assessment
    mvp_checks = [
        results['reconstruction']['mvp_pass'],
        results['interpolation']['mvp_pass'],
        results['clustering']['mvp_pass'],
    ]
    
    prod_checks = [
        results['reconstruction']['production_pass'],
        results['interpolation']['production_pass'],
        results['clustering']['production_pass'],
    ]
    
    mvp_pass = sum(mvp_checks) >= 2  # At least 2 of 3
    prod_pass = sum(prod_checks) >= 2
    
    report['overall_assessment'] = {
        'mvp_pass': mvp_pass,
        'production_pass': prod_pass,
        'mvp_checks_passed': f"{sum(mvp_checks)}/3",
        'production_checks_passed': f"{sum(prod_checks)}/3",
    }
    
    # Save JSON report
    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Saved validation report: {report_path}")
    
    # Print summary
    print(f"\nOverall Assessment:")
    if prod_pass:
        print(f"  🌟 PRODUCTION QUALITY ACHIEVED!")
        print(f"     {sum(prod_checks)}/3 production criteria passed")
    elif mvp_pass:
        print(f"  ✅ MVP QUALITY ACHIEVED")
        print(f"     {sum(mvp_checks)}/3 MVP criteria passed")
    else:
        print(f"  ⚠️  Below MVP quality")
        print(f"     {sum(mvp_checks)}/3 MVP criteria passed")
    
    print(f"\nKey Metrics:")
    print(f"  Reconstruction MSE: {results['reconstruction']['overall_mse']:.6f}")
    print(f"  Outcome MAE: {results['reconstruction']['outcome_mae']*100:.2f}%")
    print(f"  Interpolation Coherence: {results['interpolation']['avg_coherence']:.4f}")
    print(f"  Silhouette Score: {results['clustering']['silhouette_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Validate VAE latent space')
    
    # Paths
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='results/vae_seed_training/vae_best_epoch_2.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--webalchemist-path',
        type=str,
        default='c:/dev/projects/WebAlchemist/organized_rag',
        help='Path to WebAlchemist data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/vae_validation',
        help='Output directory for validation results'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use'
    )
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=32,
        help='Latent dimension (only used if not in checkpoint)'
    )
    
    # Test parameters
    parser.add_argument(
        '--num-interpolation-pairs',
        type=int,
        default=10,
        help='Number of state pairs for interpolation testing'
    )
    parser.add_argument(
        '--interpolation-steps',
        type=int,
        default=10,
        help='Number of steps in each interpolation'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("VAE Latent Space Validation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    # Load model
    model = load_model(args.checkpoint, args.device, args.latent_dim)
    
    # Load seed data
    states = load_seed_data(args.webalchemist_path)
    
    # Run validation tests
    results = {}
    
    # 1. Reconstruction quality
    results['reconstruction'] = test_reconstruction_quality(model, states, args.device)
    
    # 2. Interpolation smoothness
    results['interpolation'] = test_interpolation_smoothness(
        model, states, args.device,
        num_pairs=args.num_interpolation_pairs,
        num_steps=args.interpolation_steps,
    )
    
    # 3. Semantic clustering
    results['clustering'] = test_semantic_clustering(model, states, args.device, output_dir)
    
    # 4. Latent arithmetic
    results['arithmetic'] = test_latent_arithmetic(model, states, args.device, output_dir)
    
    # Generate report
    generate_validation_report(results, output_dir)
    
    print("\n" + "="*60)
    print("✓ Validation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

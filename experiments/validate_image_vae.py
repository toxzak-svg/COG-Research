"""
Quick validation script for Image VAE latent space
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))

from webpage_models.image_vae import ImageWebpageVAE
from webpage_data_collection.aesthetics_loader import create_dataloaders


def compute_latent_representations(model, dataloader, device):
    """Extract all latent representations"""
    model.eval()
    
    latents = []
    scores = []
    
    with torch.no_grad():
        for images, aesthetic_scores, metadata in dataloader:
            images = images.to(device)
            
            # Get latent representation
            output = model(images, return_latent=True)
            mu = output['mu']
            
            latents.append(mu.cpu().numpy())
            scores.append(aesthetic_scores.numpy())
    
    latents = np.vstack(latents)
    scores = np.vstack(scores)
    
    return latents, scores


def validate_latent_space(checkpoint_path, comparison_dataset, rating_dataset, device='cpu'):
    """Validate image VAE latent space quality"""
    
    print("="*60)
    print("Image VAE Latent Space Validation")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("="*60)
    
    # Load checkpoint
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    latent_dim = checkpoint.get('latent_dim', 64)
    print(f"  Latent dimension: {latent_dim}")
    
    # Create model
    model = ImageWebpageVAE(latent_dim=latent_dim, image_size=256)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        comparison_dataset_path=comparison_dataset,
        rating_dataset_path=rating_dataset,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    )
    
    # Compute latent representations
    print("\nComputing latent representations...")
    train_latents, train_scores = compute_latent_representations(model, train_loader, device)
    val_latents, val_scores = compute_latent_representations(model, val_loader, device)
    test_latents, test_scores = compute_latent_representations(model, test_loader, device)
    
    print(f"  Train: {train_latents.shape[0]} samples")
    print(f"  Val: {val_latents.shape[0]} samples")
    print(f"  Test: {test_latents.shape[0]} samples")
    
    # Compute metrics
    print("\n" + "="*60)
    print("LATENT SPACE QUALITY METRICS")
    print("="*60)
    
    # 1. Clustering quality (Silhouette Score)
    # Cluster by aesthetic score quartiles
    train_score_quartiles = np.digitize(train_scores.flatten(), 
                                        bins=np.percentile(train_scores, [25, 50, 75]))
    
    if len(np.unique(train_score_quartiles)) > 1:
        silhouette = silhouette_score(train_latents, train_score_quartiles)
        print(f"\n1. Silhouette Score (aesthetic quartiles): {silhouette:.4f}")
        if silhouette > 0.3:
            print("   ✓ EXCELLENT - Strong semantic clustering")
        elif silhouette > 0.1:
            print("   ✓ GOOD - Moderate semantic clustering")
        elif silhouette > 0:
            print("   ⚠ WEAK - Some structure but needs improvement")
        else:
            print("   ✗ POOR - Minimal semantic structure")
    else:
        print("\n1. Silhouette Score: N/A (insufficient label diversity)")
    
    # 2. Latent space statistics
    print(f"\n2. Latent Space Statistics:")
    print(f"   Mean activation: {np.mean(np.abs(train_latents)):.4f}")
    print(f"   Std activation: {np.std(train_latents):.4f}")
    print(f"   Active dimensions (>0.1): {np.sum(np.std(train_latents, axis=0) > 0.1)}/{latent_dim}")
    
    # 3. Reconstruction quality (from checkpoint)
    print(f"\n3. Reconstruction Loss: {checkpoint.get('val_recon_loss', 'N/A')}")
    
    # 4. Total Correlation (from checkpoint)
    print(f"4. Total Correlation: {checkpoint.get('val_tc_loss', 'N/A')}")
    
    # 5. Outcome prediction (from checkpoint)
    print(f"5. Outcome Prediction Loss: {checkpoint.get('val_outcome_loss', 'N/A')}")
    
    # 6. PCA variance explained
    print(f"\n6. PCA Variance Explained:")
    pca = PCA(n_components=min(10, latent_dim))
    pca.fit(train_latents)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"   First 3 components: {cumvar[2]:.2%}")
    print(f"   First 5 components: {cumvar[4]:.2%}")
    print(f"   First 10 components: {cumvar[min(9, latent_dim-1)]:.2%}")
    
    # Visualization
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    output_dir = Path("results/image_vae_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: PCA projection colored by aesthetic score
    pca_2d = PCA(n_components=2)
    train_pca = pca_2d.fit_transform(train_latents)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(train_pca[:, 0], train_pca[:, 1], 
                         c=train_scores.flatten(), 
                         cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Aesthetic Score')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} var)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} var)')
    plt.title('Image VAE Latent Space (PCA Projection)')
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_pca.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'latent_pca.png'}")
    
    # Plot 2: Latent dimension activations
    plt.figure(figsize=(12, 6))
    dim_stds = np.std(train_latents, axis=0)
    plt.bar(range(latent_dim), dim_stds)
    plt.axhline(y=0.1, color='r', linestyle='--', label='Active threshold (0.1)')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Standard Deviation')
    plt.title(f'Latent Dimension Activity ({np.sum(dim_stds > 0.1)}/{latent_dim} active)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'dimension_activity.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'dimension_activity.png'}")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    results = {
        'checkpoint': str(checkpoint_path),
        'epoch': checkpoint.get('epoch', None),
        'latent_dim': latent_dim,
        'silhouette_score': float(silhouette) if len(np.unique(train_score_quartiles)) > 1 else None,
        'active_dimensions': int(np.sum(dim_stds > 0.1)),
        'mean_activation': float(np.mean(np.abs(train_latents))),
        'std_activation': float(np.std(train_latents)),
        'pca_3comp_variance': float(cumvar[2]),
        'val_loss': checkpoint.get('val_loss', None),
        'val_recon_loss': checkpoint.get('val_recon_loss', None),
        'val_tc_loss': checkpoint.get('val_tc_loss', None),
        'val_outcome_loss': checkpoint.get('val_outcome_loss', None),
    }
    
    # Save results
    with open(output_dir / 'validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_dir / 'validation_results.json'}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--comparison-dataset', type=str, 
                       default='website-aesthetics-datasets/comparison-based-dataset')
    parser.add_argument('--rating-dataset', type=str,
                       default='website-aesthetics-datasets/rating-based-dataset')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    validate_latent_space(
        args.checkpoint,
        args.comparison_dataset,
        args.rating_dataset,
        args.device
    )

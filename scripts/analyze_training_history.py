"""Quick analysis of training history"""
import json
import numpy as np

with open('results/vae_aggressive_tc/training_history.json', 'r') as f:
    history = json.load(f)

train_losses = history['train_losses']
val_losses = history['val_losses']

print(f"Total epochs: {len(train_losses)}")

# Find best validation loss
val_total_losses = [v['total'] for v in val_losses]
best_idx = np.argmin(val_total_losses)
best_epoch = best_idx + 1  # 1-indexed

print(f"\nBest epoch: {best_epoch}")
print(f"Best val loss: {val_total_losses[best_idx]:.4f}")

best = val_losses[best_idx]
print(f"\nBest epoch ({best_epoch}) validation metrics:")
print(f"  Total Loss: {best['total']:.4f}")
print(f"  Reconstruction: {best['reconstruction']:.6f}")
print(f"  Total Correlation (TC): {best['total_correlation']:.6f}")
print(f"  Index-Code MI: {best['index_code_mi']:.6f}")
print(f"  Dimension-wise KL: {best['dimension_wise_kl']:.6f}")
print(f"  Total KL: {best['total_kl']:.6f}")

print(f"\nFinal 5 epochs:")
for i, v in enumerate(val_losses[-5:]):
    epoch_num = len(val_losses) - 5 + i + 1
    print(f"Epoch {epoch_num:3d}: Total={v['total']:7.4f}, Recon={v['reconstruction']:.6f}, TC={v['total_correlation']:.6f}")

print(f"\nFirst 5 epochs:")
for i, v in enumerate(val_losses[:5]):
    epoch_num = i + 1
    print(f"Epoch {epoch_num:3d}: Total={v['total']:7.4f}, Recon={v['reconstruction']:.6f}, TC={v['total_correlation']:.6f}")

# Check for collapse indicators
print(f"\nCollapse indicators:")
tc_values = [v['total_correlation'] for v in val_losses]
recon_values = [v['reconstruction'] for v in val_losses]
print(f"  TC range: [{min(tc_values):.4f}, {max(tc_values):.4f}]")
print(f"  Recon range: [{min(recon_values):.6f}, {max(recon_values):.6f}]")
print(f"  Final TC: {tc_values[-1]:.4f}")
print(f"  Final Recon: {recon_values[-1]:.6f}")

# Check if TC stayed negative (good sign for β-TC-VAE)
negative_tc_count = sum(1 for tc in tc_values if tc < 0)
print(f"  Epochs with negative TC: {negative_tc_count}/{len(tc_values)} ({100*negative_tc_count/len(tc_values):.1f}%)")


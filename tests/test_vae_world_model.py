"""
Smoke test for VAE World-Model implementation

Quick validation that model architecture works correctly.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from webpage_models import WebpageVAE, VAELoss, reparameterize


def test_vae_forward():
    """Test basic forward pass"""
    print("Testing VAE forward pass...")
    
    # Create model
    model = WebpageVAE(
        input_dim=190,
        latent_dim=32,
        outcome_dim=10,
    )
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 190)
    
    # Forward pass
    x_recon, outcomes, mu, logvar = model(x, deterministic=False)
    
    # Check shapes
    assert x_recon.shape == (batch_size, 190), f"Expected (4, 190), got {x_recon.shape}"
    assert outcomes.shape == (batch_size, 10), f"Expected (4, 10), got {outcomes.shape}"
    assert mu.shape == (batch_size, 32), f"Expected (4, 32), got {mu.shape}"
    assert logvar.shape == (batch_size, 32), f"Expected (4, 32), got {logvar.shape}"
    
    print("✓ Forward pass shapes correct")
    
    # Check value ranges
    assert torch.all(x_recon >= 0) and torch.all(x_recon <= 1), "Reconstruction should be in [0, 1]"
    assert torch.all(outcomes >= 0) and torch.all(outcomes <= 1), "Outcomes should be in [0, 1]"
    
    print("✓ Output value ranges correct")


def test_encode_decode():
    """Test encode and decode separately"""
    print("\nTesting encode/decode...")
    
    model = WebpageVAE()
    x = torch.randn(2, 190)
    
    # Encode
    z = model.encode(x, deterministic=True)
    assert z.shape == (2, 32), f"Latent shape incorrect: {z.shape}"
    
    # Decode
    x_recon, outcomes = model.decode(z)
    assert x_recon.shape == (2, 190), f"Reconstruction shape incorrect: {x_recon.shape}"
    assert outcomes.shape == (2, 10), f"Outcomes shape incorrect: {outcomes.shape}"
    
    print("✓ Encode/decode work correctly")


def test_interpolation():
    """Test interpolation between states"""
    print("\nTesting interpolation...")
    
    model = WebpageVAE()
    x1 = torch.randn(1, 190)
    x2 = torch.randn(1, 190)
    
    x_interp, outcomes_interp = model.interpolate(x1, x2, num_steps=5)
    
    assert x_interp.shape == (5, 190), f"Interpolation shape incorrect: {x_interp.shape}"
    assert outcomes_interp.shape == (5, 10), f"Outcome interpolation shape incorrect: {outcomes_interp.shape}"
    
    print("✓ Interpolation works correctly")


def test_sampling():
    """Test sampling from prior"""
    print("\nTesting sampling...")
    
    model = WebpageVAE()
    x_samples, outcomes_samples = model.sample(num_samples=3, device='cpu')
    
    assert x_samples.shape == (3, 190), f"Sample shape incorrect: {x_samples.shape}"
    assert outcomes_samples.shape == (3, 10), f"Sample outcomes shape incorrect: {outcomes_samples.shape}"
    
    print("✓ Sampling works correctly")


def test_reparameterization():
    """Test reparameterization trick"""
    print("\nTesting reparameterization...")
    
    mu = torch.zeros(2, 32)
    logvar = torch.zeros(2, 32)
    
    # Sample multiple times - should be different
    z1 = reparameterize(mu, logvar)
    z2 = reparameterize(mu, logvar)
    
    assert z1.shape == (2, 32), f"Reparameterized shape incorrect: {z1.shape}"
    assert not torch.allclose(z1, z2), "Samples should be different (stochastic)"
    
    # With zero variance, should equal mean
    z_det = reparameterize(mu, torch.ones_like(logvar) * -10)  # Very low variance
    assert torch.allclose(z_det, mu, atol=0.1), "Low variance should approximate mean"
    
    print("✓ Reparameterization works correctly")


def test_loss_computation():
    """Test VAE loss function"""
    print("\nTesting loss computation...")
    
    model = WebpageVAE()
    loss_fn = VAELoss(beta_recon=1.0, beta_kl=0.5, beta_outcome=1.0, beta_consistency=0.2)
    
    # Create dummy data
    x = torch.randn(4, 190)
    outcomes_true = torch.rand(4, 10)
    
    # Forward pass
    x_recon, outcomes_pred, mu, logvar = model(x)
    z = reparameterize(mu, logvar)
    
    # Compute loss
    total_loss, loss_dict = loss_fn(
        model=model,
        x=x,
        x_recon=x_recon,
        outcomes_pred=outcomes_pred,
        outcomes_true=outcomes_true,
        mu=mu,
        logvar=logvar,
        z=z,
    )
    
    # Check loss is scalar
    assert total_loss.dim() == 0, "Loss should be scalar"
    assert total_loss.item() > 0, "Loss should be positive"
    
    # Check loss dict
    expected_keys = ['total', 'reconstruction', 'kl_divergence', 'outcome_prediction', 'consistency']
    assert all(k in loss_dict for k in expected_keys), f"Missing keys in loss_dict: {loss_dict.keys()}"
    
    print("✓ Loss computation works correctly")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Reconstruction: {loss_dict['reconstruction']:.4f}")
    print(f"  KL divergence: {loss_dict['kl_divergence']:.4f}")
    print(f"  Outcome prediction: {loss_dict['outcome_prediction']:.4f}")
    print(f"  Consistency: {loss_dict['consistency']:.4f}")


def test_backward_pass():
    """Test that gradients flow correctly"""
    print("\nTesting backward pass...")
    
    model = WebpageVAE()
    loss_fn = VAELoss()
    
    x = torch.randn(2, 190)
    outcomes_true = torch.rand(2, 10)
    
    # Forward
    x_recon, outcomes_pred, mu, logvar = model(x)
    z = reparameterize(mu, logvar)
    
    # Loss
    total_loss, _ = loss_fn(model, x, x_recon, outcomes_pred, outcomes_true, mu, logvar, z)
    
    # Backward
    total_loss.backward()
    
    # Check gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients computed"
    print("✓ Backward pass works correctly")


def test_model_parameters():
    """Test model parameter count"""
    print("\nTesting model parameters...")
    
    model = WebpageVAE()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Sanity check: Should have reasonable number of parameters (10k - 1M)
    assert 10_000 < total_params < 1_000_000, f"Unexpected parameter count: {total_params}"
    assert total_params == trainable_params, "All parameters should be trainable"
    
    print("✓ Parameter count reasonable")


def main():
    """Run all smoke tests"""
    print("=" * 60)
    print("VAE World-Model Smoke Tests")
    print("=" * 60)
    
    try:
        test_vae_forward()
        test_encode_decode()
        test_interpolation()
        test_sampling()
        test_reparameterization()
        test_loss_computation()
        test_backward_pass()
        test_model_parameters()
        
        print("\n" + "=" * 60)
        print("✅ ALL SMOKE TESTS PASSED!")
        print("=" * 60)
        print("\nVAE implementation ready for training.")
        
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

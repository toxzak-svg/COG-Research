import unittest

try:
    import torch
    from imagination_first_learning.models.vae import VAE, vae_loss
except ModuleNotFoundError:
    torch = None
    VAE = None
    vae_loss = None


@unittest.skipIf(torch is None, "torch is not installed in the active environment")
class TestVAESmoke(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 8
        self.input_dim = 10
        self.latent_dim = 2
        self.model = VAE(input_dim=self.input_dim, latent_dim=self.latent_dim)
        self.x = torch.rand(self.batch_size, self.input_dim)

    def test_forward_shapes(self):
        recon_x, mu, log_var = self.model(self.x)
        self.assertEqual(tuple(recon_x.shape), (self.batch_size, self.input_dim))
        self.assertEqual(tuple(mu.shape), (self.batch_size, self.latent_dim))
        self.assertEqual(tuple(log_var.shape), (self.batch_size, self.latent_dim))

    def test_loss_is_finite_and_backward(self):
        recon_x, mu, log_var = self.model(self.x)
        loss = vae_loss(recon_x, self.x, mu, log_var)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        grad_sum = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                grad_sum += parameter.grad.abs().sum().item()

        self.assertGreater(grad_sum, 0.0)


if __name__ == "__main__":
    unittest.main()

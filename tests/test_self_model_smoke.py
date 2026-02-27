import unittest

try:
    import torch
    from minimal_self_model.models.self_model import SelfModel, self_model_loss
except ModuleNotFoundError:
    torch = None
    SelfModel = None
    self_model_loss = None


@unittest.skipIf(torch is None, "torch is not installed in the active environment")
class TestSelfModelSmoke(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 4
        self.seq_len = 6
        self.input_dim = 5
        self.hidden_dim = 8
        self.output_dim = 5
        self.model = SelfModel(self.input_dim, self.hidden_dim, self.output_dim)
        self.x = torch.rand(self.batch_size, self.seq_len, self.input_dim)
        self.target = torch.rand(self.batch_size, self.seq_len, self.output_dim)

    def test_forward_shape(self):
        y = self.model(self.x)
        self.assertEqual(tuple(y.shape), (self.batch_size, self.seq_len, self.output_dim))

    def test_loss_is_finite_and_backward(self):
        y = self.model(self.x)
        loss = self_model_loss(y, self.target)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(loss.item(), 0.0)

        loss.backward()
        grad_sum = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                grad_sum += parameter.grad.abs().sum().item()

        self.assertGreater(grad_sum, 0.0)


if __name__ == "__main__":
    unittest.main()

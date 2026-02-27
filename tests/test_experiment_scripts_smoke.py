import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from imagination_first_learning.models.vae import VAE
except ModuleNotFoundError:
    torch = None
    VAE = None


REPO_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "torch is not installed in the active environment")
class TestExperimentScriptsSmoke(unittest.TestCase):
    maxDiff = None

    def run_cmd(self, args, timeout=240):
        proc = subprocess.run(
            args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(
                "Command failed with exit code "
                f"{proc.returncode}: {' '.join(str(a) for a in args)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
        return proc

    def write_dummy_vae_checkpoint(self, path: Path, input_dim=5, latent_dim=2):
        torch.manual_seed(0)
        model = VAE(input_dim=input_dim, latent_dim=latent_dim)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": int(input_dim),
                "latent_dim": int(latent_dim),
            },
            "training": {
                "epochs": 0,
                "data_path": "<dummy-for-smoke-test>",
            },
        }
        torch.save(checkpoint, path)

    def test_vae_train_eval_cli_smoke_and_checkpoint_round_trip(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            data_path = tmpdir / "tiny_vae_data.npy"
            ckpt_path = tmpdir / "tiny_vae_model.pth"

            data = rng.uniform(0.0, 1.0, size=(24, 10)).astype(np.float32)
            np.save(data_path, data)

            train_cmd = [
                sys.executable,
                "imagination_first_learning/experiments/train_vae.py",
                "--data-path",
                str(data_path),
                "--output-path",
                str(ckpt_path),
                "--latent-dim",
                "2",
                "--batch-size",
                "8",
                "--epochs",
                "1",
                "--lr",
                "0.001",
                "--val-fraction",
                "0.25",
                "--recon-loss",
                "mse",
                "--beta",
                "1.0",
                "--kl-warmup-epochs",
                "0",
                "--seed",
                "0",
                "--device",
                "cpu",
                "--num-workers",
                "0",
            ]
            train_proc = self.run_cmd(train_cmd)
            self.assertIn("Saved checkpoint to", train_proc.stdout)
            self.assertTrue(ckpt_path.exists())

            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("config", checkpoint)
            self.assertIn("training", checkpoint)
            self.assertEqual(checkpoint["config"]["input_dim"], 10)
            self.assertEqual(checkpoint["config"]["latent_dim"], 2)
            self.assertEqual(checkpoint["training"]["epochs"], 1)
            self.assertEqual(checkpoint["training"]["val_fraction"], 0.25)
            self.assertIsNotNone(checkpoint["training"]["last_train_metrics"])

            eval_cmd = [
                sys.executable,
                "imagination_first_learning/experiments/evaluate_vae.py",
                "--model-path",
                str(ckpt_path),
                "--data-path",
                str(data_path),
                "--batch-size",
                "8",
                "--device",
                "cpu",
            ]
            eval_proc = self.run_cmd(eval_cmd)
            self.assertIn("MSE reconstruction error per sample", eval_proc.stdout)
            self.assertIn("Checkpoint training epochs: 1", eval_proc.stdout)

    def test_self_model_train_eval_cli_smoke_and_checkpoint_round_trip(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            data_path = tmpdir / "tiny_seq_data.npy"
            self_ckpt_path = tmpdir / "tiny_self_model.pth"
            vae_ckpt_path = tmpdir / "tiny_baseline_vae.pth"
            latent_path = tmpdir / "tiny_latent_transition.npz"

            sequences = rng.uniform(0.0, 1.0, size=(12, 6, 5)).astype(np.float32)
            np.save(data_path, sequences)
            self.write_dummy_vae_checkpoint(vae_ckpt_path, input_dim=5, latent_dim=2)

            train_cmd = [
                sys.executable,
                "minimal_self_model/experiments/train_self_model.py",
                "--data-path",
                str(data_path),
                "--output-path",
                str(self_ckpt_path),
                "--vae-model-path",
                str(vae_ckpt_path),
                "--latent-transition-path",
                str(latent_path),
                "--hidden-dim",
                "4",
                "--batch-size",
                "4",
                "--epochs",
                "1",
                "--lr",
                "0.001",
                "--val-fraction",
                "0.25",
                "--seed",
                "0",
                "--device",
                "cpu",
                "--num-workers",
                "0",
            ]
            train_proc = self.run_cmd(train_cmd)
            self.assertIn("Saved self-model checkpoint to", train_proc.stdout)
            self.assertIn("Saved VAE latent transition to", train_proc.stdout)
            self.assertTrue(self_ckpt_path.exists())
            self.assertTrue(latent_path.exists())

            self_ckpt = torch.load(self_ckpt_path, map_location="cpu")
            self.assertIn("model_state_dict", self_ckpt)
            self.assertIn("config", self_ckpt)
            self.assertIn("training", self_ckpt)
            self.assertIn("baseline", self_ckpt)
            self.assertEqual(self_ckpt["config"]["input_dim"], 5)
            self.assertEqual(self_ckpt["config"]["hidden_dim"], 4)
            self.assertEqual(self_ckpt["config"]["output_dim"], 5)
            self.assertEqual(self_ckpt["training"]["epochs"], 1)
            self.assertIsNotNone(self_ckpt["training"]["train_indices"])
            self.assertIsNotNone(self_ckpt["training"]["val_indices"])
            self.assertEqual(self_ckpt["baseline"]["vae_latent_dim"], 2)

            with np.load(latent_path) as latent:
                self.assertIn("A", latent.files)
                self.assertIn("b", latent.files)
                self.assertEqual(tuple(latent["A"].shape), (2, 2))
                self.assertEqual(tuple(latent["b"].shape), (2,))

            eval_cmd = [
                sys.executable,
                "minimal_self_model/experiments/evaluate_self_model.py",
                "--self-model-path",
                str(self_ckpt_path),
                "--vae-model-path",
                str(vae_ckpt_path),
                "--latent-transition-path",
                str(latent_path),
                "--data-path",
                str(data_path),
                "--split",
                "val",
                "--horizons",
                "1",
                "2",
                "3",
                "--batch-size",
                "4",
                "--device",
                "cpu",
            ]
            eval_proc = self.run_cmd(eval_cmd)
            self.assertIn("Self-model one-step MSE per element", eval_proc.stdout)
            self.assertIn("VAE latent-transition one-step MSE per element", eval_proc.stdout)
            self.assertIn("Self-model rollout MSE per element:", eval_proc.stdout)
            self.assertIn("VAE latent-transition rollout MSE per element:", eval_proc.stdout)


if __name__ == "__main__":
    unittest.main()

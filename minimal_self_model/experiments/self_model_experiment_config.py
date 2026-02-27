from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DATA_PATH = PROJECT_ROOT / "minimal_self_model" / "data" / "synthetic_sequential_data.npy"
LEGACY_DATA_PATH = PROJECT_ROOT / "synthetic_sequential_data.npy"
DEFAULT_SELF_MODEL_PATH = PROJECT_ROOT / "self_model.pth"
DEFAULT_VAE_MODEL_PATH = PROJECT_ROOT / "vae_model.pth"
DEFAULT_LATENT_TRANSITION_PATH = PROJECT_ROOT / "vae_latent_transition.npz"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "minimal_self_model" / "experiments" / "configs" / "self_model_experiment.json"
)


def _coerce_project_path(value):
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_experiment_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Self-model config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Expected top-level object in {config_path}, got {type(config).__name__}")
    return config


def get_train_config_defaults(config_path):
    config = load_experiment_config(config_path)
    shared = config.get("shared", {})
    train = config.get("train", {})

    defaults = {}
    if "data_path" in shared:
        defaults["data_path"] = _coerce_project_path(shared["data_path"])
    if "self_model_path" in shared:
        defaults["output_path"] = _coerce_project_path(shared["self_model_path"])
    if "vae_model_path" in shared:
        defaults["vae_model_path"] = _coerce_project_path(shared["vae_model_path"])
    if "latent_transition_path" in shared:
        defaults["latent_transition_path"] = _coerce_project_path(shared["latent_transition_path"])
    for key in ("device", "seed"):
        if key in shared:
            defaults[key] = shared[key]
    for key in ("hidden_dim", "batch_size", "epochs", "lr", "val_fraction", "num_workers"):
        if key in train:
            defaults[key] = train[key]
    return defaults


def get_eval_config_defaults(config_path):
    config = load_experiment_config(config_path)
    shared = config.get("shared", {})
    eval_cfg = config.get("eval", {})

    defaults = {}
    if "data_path" in shared:
        defaults["data_path"] = _coerce_project_path(shared["data_path"])
    if "self_model_path" in shared:
        defaults["self_model_path"] = _coerce_project_path(shared["self_model_path"])
    if "vae_model_path" in shared:
        defaults["vae_model_path"] = _coerce_project_path(shared["vae_model_path"])
    if "latent_transition_path" in shared:
        defaults["latent_transition_path"] = _coerce_project_path(shared["latent_transition_path"])
    if "device" in shared:
        defaults["device"] = shared["device"]
    for key in ("batch_size", "horizons"):
        if key in eval_cfg:
            defaults[key] = eval_cfg[key]
    return defaults

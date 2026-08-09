"""Research utilities for controlled Diffusion Transformer experiments."""

from .config import ExperimentConfig, load_config
from .model import DiT, build_model, resolve_ffn_widths

__all__ = ["DiT", "ExperimentConfig", "build_model", "load_config", "resolve_ffn_widths"]
__version__ = "0.1.0"

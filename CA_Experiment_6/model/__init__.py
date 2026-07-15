"""From-scratch ADA SLM: Llama-style transformer + configs (plan §5.1)."""

from .config import (
    ModelConfig, TrainConfig, ADA_80M, ADA_160M, ADA_300M, ADA_PILOT, ADA_SMOKE,
)
from .model import ADATransformer

__all__ = [
    "ModelConfig", "TrainConfig", "ADA_80M", "ADA_160M", "ADA_300M", "ADA_PILOT", "ADA_SMOKE",
    "ADATransformer",
]

"""Model + training configuration for the from-scratch ADA SLM (plan §5.1)."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    """Decoder-only Llama-style transformer (~80M at the §5.1 defaults)."""
    vocab_size: int = 16000            # own BPE (train_tokenizer.py); +special tokens
    d_model: int = 640
    n_layers: int = 12
    n_heads: int = 10                  # head_dim = d_model / n_heads = 64
    n_kv_heads: int | None = None      # None → MHA (== n_heads); set < n_heads for GQA
    max_seq_len: int = 1024            # raise to 2048 only if compute allows (§5.1)
    # SwiGLU hidden. If None, computed as multiple_of-rounded 8/3 * d_model.
    ffn_hidden: int | None = None
    multiple_of: int = 64
    ffn_mult: float = 8 / 3
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    init_std: float = 0.02

    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        return self.d_model // self.n_heads

    def kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    def hidden_dim(self) -> int:
        if self.ffn_hidden is not None:
            return self.ffn_hidden
        h = int(self.ffn_mult * self.d_model)
        return self.multiple_of * ((h + self.multiple_of - 1) // self.multiple_of)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ── Named configs ────────────────────────────────────────────────────────

# §5.1 target (~80M): 640 / 12 / 10, 16k vocab.
ADA_80M = ModelConfig()

# Scaled backbone (~170M): 896 / 16 / 14 (head_dim 64), 16k vocab. The 80M's
# representation ceiling capped grounded-QA abstention (~0.68/0.68 crossover); the
# capacity bump targets a simultaneous reading+abstention ≥0.70 operating point.
ADA_160M = ModelConfig(
    d_model=896, n_layers=16, n_heads=14, max_seq_len=1024,
)

# §5.4 pilot gate — tiny model to confirm the pipeline learns at all.
ADA_PILOT = ModelConfig(
    vocab_size=16000, d_model=256, n_layers=6, n_heads=8, max_seq_len=512,
)

# CPU smoke config (local .venv sanity test only — not for real training).
ADA_SMOKE = ModelConfig(
    vocab_size=512, d_model=64, n_layers=2, n_heads=4, max_seq_len=128,
)


@dataclass
class TrainConfig:
    """Optimisation + schedule + checkpoint config shared by both stages."""
    stage: str = "pretrain"            # "pretrain" | "sft"
    batch_size: int = 32               # sequences per step (per device)
    grad_accum_steps: int = 8
    seq_len: int = 1024
    max_steps: int = 40000
    warmup_steps: int = 500
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 100
    ckpt_interval: int = 500           # checkpoint to Drive every N steps (§5.3)
    log_interval: int = 20
    dtype: str = "bfloat16"            # "bfloat16" | "float16" | "float32"
    compile: bool = False
    seed: int = 1337
    # SFT-only: fraction of Stage-A LM data mixed in to prevent fluency forgetting.
    sft_replay_frac: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

"""From-scratch decoder-only transformer for the ADA daily-QA SLM (plan §5.1).

Llama-style: RoPE, RMSNorm (pre-norm), SwiGLU MLP, tied embeddings, causal
self-attention with optional GQA. Minimal nanoGPT/llama2.c-style implementation
kept dependency-light for Colab portability. Pure PyTorch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ── RMSNorm ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


# ── Rotary position embedding ────────────────────────────────────────────

def precompute_rope(head_dim: int, max_seq_len: int, theta: float,
                    device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape (max_seq_len, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)          # (T, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q,k: (B, n_heads, T, head_dim); cos,sin: (T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out.to(q.dtype), k_out.to(k.dtype)


# ── Attention (causal, GQA-capable) ──────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.kv_heads()
        self.head_dim = cfg.head_dim()
        self.n_rep = self.n_heads // self.n_kv_heads
        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, attn_mask=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        drop = self.dropout if self.training else 0.0
        if attn_mask is not None:
            # Explicit additive mask (prefix-LM: bidirectional prefix + causal suffix).
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ── SwiGLU MLP ───────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.hidden_dim()
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)   # down
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ── Prefix-LM attention mask ─────────────────────────────────────────────

def build_prefix_lm_mask(prefix_lens: torch.Tensor, T: int, device, dtype) -> torch.Tensor:
    """Additive attention mask (B,1,T,T) for prefix-LM: token i may attend to j iff
    j <= i (causal everywhere) OR both i,j are in the prefix (j < b and i < b) —
    i.e. the prefix (system+persona+context+question) is bidirectional, the answer
    is causal. prefix_lens: (B,) int lengths of each sequence's prefix."""
    i = torch.arange(T, device=device).view(1, T, 1)
    j = torch.arange(T, device=device).view(1, 1, T)
    causal = j <= i                                          # (1,T,T)
    b = prefix_lens.to(device).view(-1, 1, 1)
    prefix = (i < b) & (j < b)                               # (B,T,T)
    allow = causal | prefix
    neg = torch.finfo(dtype).min
    mask = torch.zeros(allow.shape, device=device, dtype=dtype).masked_fill(~allow, neg)
    return mask.unsqueeze(1)                                 # (B,1,T,T)


# ── Transformer block ────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask=attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── Full model ───────────────────────────────────────────────────────────

class ADATransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        # Extractive span head (Phase 1): start/end position logits over tokens —
        # discriminative reading, far more sample-efficient than generative extraction.
        self.span_head = nn.Linear(cfg.d_model, 2)
        # Answerability head: predicts whether the passage answers the question,
        # decoupling the abstain decision from the span-null margin so reading can
        # run at its optimum. Inputs: two pooled views of the bidirectional
        # sequence — the <|bos|> rep AND a mean over real tokens (the backbone was
        # pretrained *causally*, so bos alone is a weak whole-sequence summary) —
        # PLUS 3 span-confidence scalars (null score, best context-span score,
        # their margin). The span signal is a *different* view than the pooled
        # classifier: low span confidence strongly indicates no-answer, which
        # directly lifts abstention recall. Small MLP over the concat.
        self.answerable_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model + 3, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

        cos, sin = precompute_rope(cfg.head_dim(), cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init on residual projections (GPT-2 / nanoGPT convention).
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_embeddings:
            n -= self.lm_head.weight.numel()
        return n

    def _hidden(self, idx: torch.Tensor,
                prefix_lens: torch.Tensor | None = None) -> torch.Tensor:
        """Run the transformer trunk → final normed hidden states (B, T, d_model).
        Shared by the LM head (forward/generate) and the span head."""
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"seq len {T} > max {self.cfg.max_seq_len}"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        attn_mask = None
        if prefix_lens is not None:
            attn_mask = build_prefix_lm_mask(prefix_lens, T, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask=attn_mask)
        return self.norm(x)

    def _span_conf(self, start, end, valid):
        """3 span-confidence scalars per example from the span logits: the null
        (position-0) score, the best context-span score (max start + max end over
        context positions), and their margin. `valid` (B, T) marks the null anchor
        (index 0) + context positions; None → zeros (no span signal). tanh-bounded
        so the raw logit magnitudes stay scale-stable for the MLP."""
        B = start.shape[0]
        if valid is None:
            return start.new_zeros(B, 3)
        valid = valid.to(start.dtype)
        null = start[:, 0] + end[:, 0]                       # (B,)
        ctx = valid.clone()
        ctx[:, 0] = 0.0                                      # context only (drop null)
        neg = torch.finfo(start.dtype).min
        max_s = start.masked_fill(ctx < 0.5, neg).amax(dim=1)
        max_e = end.masked_fill(ctx < 0.5, neg).amax(dim=1)
        best = max_s + max_e
        margin = best - null
        return torch.tanh(0.1 * torch.stack([null, best, margin], dim=-1))

    def read_heads(self, idx: torch.Tensor,
                   prefix_lens: torch.Tensor | None = None,
                   answerable_valid: torch.Tensor | None = None):
        """Extractive-QA heads in one trunk pass: (start_logits, end_logits,
        answerable_logit) — span logits (B, T), answerable logit (B,). Callers
        pass prefix_lens = full length → fully bidirectional reading, and
        answerable_valid (B, T: null anchor + context positions) so the
        answerability head sees span-confidence features."""
        x = self._hidden(idx, prefix_lens)
        sp = self.span_head(x)                          # (B, T, 2)
        start, end = sp[..., 0], sp[..., 1]
        bos = x[:, 0, :]
        if prefix_lens is not None:                     # mean over real tokens
            T = x.size(1)
            pos = torch.arange(T, device=x.device).view(1, T)
            m = (pos < prefix_lens.to(x.device).view(-1, 1)).to(x.dtype).unsqueeze(-1)
            mean = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            mean = x.mean(1)
        span_feats = self._span_conf(start, end, answerable_valid)   # (B, 3)
        ans = self.answerable_head(torch.cat([bos, mean, span_feats], dim=-1)).squeeze(-1)
        return start, end, ans

    def span_logits(self, idx: torch.Tensor,
                    prefix_lens: torch.Tensor | None = None):
        """Span start/end logits only (back-compat wrapper over read_heads)."""
        s, e, _ = self.read_heads(idx, prefix_lens)
        return s, e

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None,
                prefix_lens: torch.Tensor | None = None):
        """idx: (B, T) token ids. targets: (B, T) next-token ids (or None).
        loss_mask: (B, T) 1.0 where the token participates in the loss (SFT
        assistant-span masking); None → all target positions count.
        prefix_lens: (B,) → prefix-LM attention (bidirectional prefix + causal
        answer); None → standard causal attention."""
        x = self._hidden(idx, prefix_lens)
        logits = self.lm_head(x)
        B, T = idx.shape

        loss = None
        if targets is not None:
            if loss_mask is None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1),
                    ignore_index=-100,
                )
            else:
                per_tok = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1),
                    ignore_index=-100, reduction="none",
                ).view(B, T)
                m = loss_mask.to(per_tok.dtype)
                denom = m.sum().clamp_min(1.0)
                loss = (per_tok * m).sum() / denom
        return logits, loss

    def configure_optimizers(self, weight_decay, lr, betas, device_type="cuda"):
        """AdamW with weight decay only on 2D+ params (nanoGPT convention)."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        try:
            return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
        except (RuntimeError, TypeError):
            return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.7, top_k: int | None = 50,
                 eos_id: int | None = None, prefix_len: int | None = None) -> torch.Tensor:
        """Greedy/sampled decode. idx: (B, T0). Crops context to max_seq_len.
        prefix_len: if set, the first prefix_len tokens attend bidirectionally
        (prefix-LM inference); generated tokens are causal."""
        self.eval()
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            plens = None
            if prefix_len is not None:
                # prefix stays bidirectional as generation grows (clamp to window)
                eff = min(prefix_len, idx_cond.shape[1])
                plens = torch.full((B,), eff, dtype=torch.long, device=idx.device)
            logits, _ = self(idx_cond, prefix_lens=plens)
            logits = logits[:, -1, :]
            if temperature <= 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx

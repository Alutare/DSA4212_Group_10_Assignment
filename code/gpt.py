"""Compact GPT (nanoGPT-style, random init only).

Differences from nanoGPT: vocabulary is padded to a multiple of `pad_to` for
tensor-core efficiency but padded logits are sliced off before the loss, so
the loss is exact; per-token losses can be returned (reduction='none');
parameter counts are broken down into embedding / non-embedding.
"""
import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import round_up


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257      # true vocabulary size
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False
    pad_to: int = 64


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias)
        self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.attn_dropout = c.dropout
        self.resid_dropout = nn.Dropout(c.dropout)
        self.n_head, self.n_embd = c.n_head, c.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=c.bias)
        self.c_proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=c.bias)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1 = LayerNorm(c.n_embd, c.bias)
        self.attn = CausalSelfAttention(c)
        self.ln_2 = LayerNorm(c.n_embd, c.bias)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padded_vocab = round_up(config.vocab_size, config.pad_to)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.padded_vocab, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, self.padded_vocab, bias=False)
        self.transformer.wte.weight = self.lm_head.weight      # weight tying
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():                  # GPT-2 residual scaling
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        V = self.config.vocab_size
        if self.padded_vocab != V:
            logits = logits[..., :V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction=reduction)
            if reduction == "none":
                loss = loss.view(B, T)
        return logits, loss

    # ---- bookkeeping -------------------------------------------------------
    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())      # tied weights counted once
        emb = self.transformer.wte.weight.numel()
        pos = self.transformer.wpe.weight.numel()
        return dict(total=total, embedding=emb, position=pos, non_embedding=total - emb - pos)

    def flops_per_token(self):
        """Training FLOPs per token (fwd+bwd): 6N + 12*L*H*Q*T (PaLM/nanoGPT convention).
        N includes the tied embedding once (it is the output matmul), excludes positions."""
        c = self.config
        n = self.param_counts()["total"] - self.transformer.wpe.weight.numel()
        return 6 * n + 12 * c.n_layer * c.n_head * (c.n_embd // c.n_head) * c.block_size

    def configure_optimizers(self, weight_decay, lr, betas, device_type, fused=True):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        nodecay = [p for p in params if p.dim() < 2]
        groups = [dict(params=decay, weight_decay=weight_decay),
                  dict(params=nodecay, weight_decay=0.0)]
        use_fused = fused and device_type == "cuda" and \
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = dict(fused=True) if use_fused else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)

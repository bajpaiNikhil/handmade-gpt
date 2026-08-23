"""
Day 6 — Multiple Heads Running in Parallel (Multi-Head Attention)
-------------------------------------------------------------------
Day 5 (causal_average.py) built ONE self-attention head: given a token,
compute one query, compare it against every key, and pull back one
weighted blend of values. That's one "question" the token gets to ask
about its own past — e.g. "what's the nearest vowel behind me?" A single
head has to cram every useful pattern into that one query/key/value
triple.

Day 6's idea: run SEVERAL heads side by side over the SAME input, each
with its own independently-initialized Q/K/V weights, so each head is
free to specialize in a different question. Concatenate their outputs
back together, then pass the result through one more learned projection
so the model can mix what the different heads found.

This file proves the mechanism on toy tensors, same shapes Day 5 used,
so the two are directly comparable. attention_lm.py wires the result
into a real trained model, same as the Day 4/5 -> detour handoff.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Toy Data (identical shapes to causal_average.py's attention demo)
# ─────────────────────────────────────────────────────────────────────────────
B, T, C = 4, 8, 32   # batch, time/position, channels (== n_embd)
x = torch.randn(B, T, C)
block_size = T
print(f"[data] x shape: {tuple(x.shape)}  (batch={B}, time={T}, channels={C})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Head (unchanged from Day 5 — redefined here so this file is
# self-contained and runnable on its own, matching this repo's convention)
# ─────────────────────────────────────────────────────────────────────────────
class Head(nn.Module):
    """One self-attention head — identical to Day 5's Head."""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Two Heads, Same Input, Different Weights: Do They Actually Diverge?
# ─────────────────────────────────────────────────────────────────────────────
# Before building MultiHeadAttention, confirm the premise: two Head instances
# with the same head_size, run on the SAME x, should produce DIFFERENT
# attention patterns purely because nn.Linear's default init randomizes each
# head's Q/K/V weights independently. If they didn't diverge, running heads
# in parallel would be pointless — four copies of the same question.
head_size = 8
head_a = Head(C, head_size, block_size)
head_b = Head(C, head_size, block_size)

def peek_wei(head, x):
    """Recompute Head.forward's internals to expose wei (Head.forward only
    returns wei @ v — the final blend — not the attention weights that
    produced it). Duplicated here purely for inspection, same spirit as
    causal_average.py exposing intermediate steps a real forward() wouldn't."""
    B, T, C = x.shape
    k = head.key(x)
    q = head.query(x)
    wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
    wei = wei.masked_fill(head.tril[:T, :T] == 0, float('-inf'))
    return F.softmax(wei, dim=-1)

wei_a = peek_wei(head_a, x)
wei_b = peek_wei(head_b, x)
print(f"\n[diverge] head_a wei[0, 3] (token 3's attention over positions 0-3): "
      f"{[round(v, 3) for v in wei_a[0, 3].tolist()[:4]]}")
print(f"[diverge] head_b wei[0, 3] (same token, same input, different weights): "
      f"{[round(v, 3) for v in wei_b[0, 3].tolist()[:4]]}")
print(f"[diverge] rows identical?: {torch.allclose(wei_a[0, 3], wei_b[0, 3])}  "
      f"(expected False — each head learns to look at the past differently)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MultiHeadAttention: Run N Heads in Parallel, Concat, Project
# ─────────────────────────────────────────────────────────────────────────────
# Each head still outputs (B, T, head_size). Running num_heads of them and
# concatenating along the last dim gives (B, T, num_heads * head_size) — set
# num_heads * head_size == n_embd so the block's output shape matches its
# input shape (needed later for residual connections in Day 8, where the
# output gets ADDED back to the input — shapes must match exactly).
#
# The final nn.Linear(n_embd, n_embd) — "proj" — isn't optional plumbing: a
# raw concat just glues four independent answers side by side with no way
# for the model to mix information ACROSS heads. proj is a learned layer
# that lets the model combine what different heads found before passing the
# result onward.
class MultiHeadAttention(nn.Module):
    """Multiple Head instances run in parallel, outputs concatenated and
    projected back to n_embd."""

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads*head_size)
        return self.proj(out)                                  # (B, T, n_embd)


num_heads = 4
head_size = C // num_heads   # 32 // 4 = 8
mha = MultiHeadAttention(num_heads, head_size, C, block_size)
out = mha(x)
print(f"\n[multi-head] num_heads={num_heads}  head_size={head_size}  "
      f"(num_heads * head_size = {num_heads * head_size} == n_embd = {C})")
print(f"[multi-head] output shape: {tuple(out.shape)}  "
      f"(matches input shape {tuple(x.shape)} — drop-in replacement for a single Head)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Parameter Count: 4 Heads of Size 8 vs. 1 Head of Size 32
# ─────────────────────────────────────────────────────────────────────────────
# Does splitting into more, smaller heads cost more parameters than one big
# head? Each Head's Q/K/V are (n_embd, head_size) matrices. Four heads at
# head_size=8 have the same TOTAL Q/K/V width (4*8=32) as one head at
# head_size=32 — same parameter count for the projections themselves.
# MultiHeadAttention's only EXTRA cost is the final proj layer.
single_head_32 = Head(C, C, block_size)                       # Day 5's shape: head_size == n_embd
single_params  = sum(p.numel() for p in single_head_32.parameters())
multi_params   = sum(p.numel() for p in mha.parameters())
proj_params    = sum(p.numel() for p in mha.proj.parameters())

print(f"\n[params] single head (head_size=32) : {single_params:,}")
print(f"[params] 4 heads (head_size=8) + proj: {multi_params:,}  "
      f"(= {multi_params - proj_params:,} in heads + {proj_params:,} in proj)")
print(f"[params] heads-only total matches single-head total?: "
      f"{multi_params - proj_params == single_params}")

print("\n[done] Day 6 complete — multi-head attention proven: heads diverge, shapes match, "
      "params add up.")
print("       next: Day 7 — the FeedForward MLP (attention communicates, MLP computes).")

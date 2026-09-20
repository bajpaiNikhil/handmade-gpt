"""
Day 7 — The FeedForward MLP ("attention communicates, MLP computes")
-------------------------------------------------------------------
Every model through Day 6 is attention-only: each token pulls a weighted
BLEND of other tokens' value vectors. A weighted average is a LINEAR
combination — no matter how many heads you run, the values themselves only
ever get proportionally mixed, never transformed. Attention decides which
tokens to draw from and how much of each; it never asks "given what I now
hold, what should I conclude?"

Today's fix: a small per-position network applied identically at every
token position (same weights everywhere, like a cookie cutter reused on
every piece of dough), expanding each token's representation to 4x its
width, running it through a nonlinear gate, then collapsing back down.
This proves the mechanism on toy tensors, same shapes Day 6's toy file
used, before attention_lm.py wires it into a real trained model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Toy Data (identical shapes to multi_head_attention.py's toy demo)
# ─────────────────────────────────────────────────────────────────────────────
B, T, C = 4, 8, 32   # batch, time/position, channels (== n_embd)
x = torch.randn(B, T, C)
print(f"[data] x shape: {tuple(x.shape)}  (batch={B}, time={T}, channels={C})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FeedForward: Expand, Gate, Collapse — Applied Per-Position
# ─────────────────────────────────────────────────────────────────────────────
# nn.Linear applies its weights to the LAST dimension only, independently at
# every (batch, position) pair — this is exactly the "same cookie cutter
# stamped on every piece of dough" behavior: one shared weight matrix, reused
# identically at all B*T positions, never told which position it's looking at.
#
# expansion=4 follows the "Attention Is All You Need" heuristic: give the
# layer a wider intermediate space (the whiteboard) to spread a thought out
# in before compressing it back to n_embd (the one-sentence summary).
#
# F.gelu is the smooth "dimmer switch" gate: unlike ReLU's hard on/off valve
# (negative in -> exactly zero out, gradient permanently dead), GELU lets a
# small amount of a slightly-negative signal through, so no pathway gets
# stuck fully shut. We hand-type the layer's SHAPE; the exact smoothing curve
# is fine to borrow from PyTorch since the shape is the pedagogical point.
class FeedForward(nn.Module):
    """Per-token feed-forward network: n_embd -> 4*n_embd -> GELU -> n_embd."""

    def __init__(self, n_embd, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expansion * n_embd),
            nn.GELU(),
            nn.Linear(expansion * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)   # (B, T, n_embd) -> (B, T, n_embd)


ffn = FeedForward(C)
out = ffn(x)
print(f"\n[ffn] output shape: {tuple(out.shape)}  (matches input shape {tuple(x.shape)} "
      f"— per-token, drop-in after attention)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Position-Invariance Proof: Same Content, Different Slot
# ─────────────────────────────────────────────────────────────────────────────
# The lecture's claim: FeedForward reacts to WHAT a token holds, not WHERE it
# sits in the sequence. Prove it directly: take one token's content, drop it
# into position 0 of one sequence and position 3 of another, run both through
# the SAME FeedForward instance, and confirm the outputs match exactly. If
# the network secretly depended on position, these would differ.
same_token_content = torch.randn(1, C)          # one token's content, made once

seq_a = torch.zeros(1, T, C)
seq_a[:, 0, :] = same_token_content              # content placed at position 0

seq_b = torch.zeros(1, T, C)
seq_b[:, 3, :] = same_token_content              # SAME content, placed at position 3

out_a = ffn(seq_a)[:, 0, :]                       # FeedForward's answer at position 0
out_b = ffn(seq_b)[:, 3, :]                       # FeedForward's answer at position 3

print(f"\n[position-invariance] same content at position 0 vs position 3")
print(f"[position-invariance] out_a[0, :4]: {[round(v, 4) for v in out_a[0, :4].tolist()]}")
print(f"[position-invariance] out_b[0, :4]: {[round(v, 4) for v in out_b[0, :4].tolist()]}")
print(f"[position-invariance] identical?: {torch.allclose(out_a, out_b, atol=1e-6)}  "
      f"(expected True — FeedForward has no idea which position it's looking at)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Parameter Count: Does the Arithmetic from the Lecture Hold?
# ─────────────────────────────────────────────────────────────────────────────
# Predicted by hand: two matrices, (32->128) and (128->32), plus their biases:
# 32*128 + 128 + 128*32 + 32 = 4096 + 128 + 4096 + 32 = 8,352.
ffn_params = sum(p.numel() for p in ffn.parameters())
predicted  = C * (4 * C) + (4 * C) + (4 * C) * C + C

print(f"\n[params] FeedForward(n_embd=32) actual params: {ffn_params:,}")
print(f"[params] hand-computed prediction            : {predicted:,}")
print(f"[params] match?: {ffn_params == predicted}")

# For scale: how big is this next to Day 6's full multi-head model (8,609)?
multihead_day6_params = 8_609
print(f"[params] FeedForward alone vs Day 6's whole multi-head model (8,609): "
      f"{ffn_params:,} is {ffn_params / multihead_day6_params:.1%} of it")

print("\n[done] FeedForward proven: per-token, position-invariant, params match prediction.")
print("       next: wire this into attention_lm.py — attention communicates, MLP computes.")

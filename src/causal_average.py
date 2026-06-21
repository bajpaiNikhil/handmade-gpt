"""
Day 4 — The Weighted-Average-of-the-Past Trick
-------------------------------------------------
Day 2's bigram model only ever looks at the current token. Day 4 takes the
first step toward fixing that: making each position's representation an
average of everything at or before it, such that the result:
  1. Is always a fixed-size vector, regardless of position (so it can slot
     into the same model shape no matter how far into the sequence we are).
  2. Is computed with one GPU-parallel operation, not a sequential Python
     loop — the same reason we ruled out a recurrent/RNN-style running
     summary (it forces step t to wait for step t-1, every time).

This script proves four ways of computing the SAME causal running average
give identical numbers: an obviously-correct double loop, a vectorized
cumsum, a matrix multiply against a triangular weight matrix, and a softmax
version of that same matmul. We carry the matmul version forward into Day 5
even though cumsum is the more efficient answer to *this* problem, because
the matmul version exposes a (T, T) weight matrix that Day 5 upgrades from
"fixed and uniform" to "learned and data-dependent" — that upgrade IS
self-attention. cumsum has no such object to upgrade.

Day 5 — A Single Self-Attention Head
---------------------------------------
Picks up exactly where Day 4's STEP 5 left off: that softmax version had a
(T, T) wei matrix, but every visible position scored a hardcoded 0 before
softmax (hence uniform weights). Day 5 replaces those hardcoded scores with
REAL ones, computed from each token's content via three learned
projections — Query, Key, Value. Same masking, same softmax, same final
wei @ (...). Only the source of the scores changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Toy Data
# ─────────────────────────────────────────────────────────────────────────────
# Small, made-up numbers so every value can be checked by eye. B = batch,
# T = time/position, C = channels (think: C is the embedding dimension —
# here just 2 numbers per token, nothing to do with the real 65-char vocab).
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(f"[data] x shape: {tuple(x.shape)}  (batch={B}, time={T}, channels={C})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Method 1: Naive Double Loop (obviously correct, deliberately slow)
# ─────────────────────────────────────────────────────────────────────────────
# For every batch, for every position t, average every token at or before t.
# Correct, but it's T sequential .mean() calls per batch row — nothing here
# asks the GPU to do work in parallel.
xbow1 = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]            # (t+1, C) — everything up to and including t
        xbow1[b, t] = xprev.mean(0)     # collapse to (C,) — the running average


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Method 2: Prefix Sums (cumsum) — vectorized, no Python loop
# ─────────────────────────────────────────────────────────────────────────────
# torch.cumsum gives the running TOTAL at every position in one call.
# Dividing by [1, 2, 3, ..., T] turns running totals into running means.
# This is the more efficient way to solve THIS problem — O(T) work, no
# (T, T) matrix ever materialized.
running_sum = torch.cumsum(x, dim=1)                  # (B, T, C)
counts      = torch.arange(1, T + 1).view(1, T, 1)    # (1, T, 1): 1, 2, 3, ... T
xbow2       = running_sum / counts                    # (B, T, C)

print(f"\n[check] loop vs cumsum match  : {torch.allclose(xbow1, xbow2, atol=1e-6)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Method 3: Matrix Multiply Against a Triangular Weight Matrix
# ─────────────────────────────────────────────────────────────────────────────
# Build a (T, T) matrix `wei` where wei[t, s] = "how much should position t's
# output depend on position s's value?" We want:
#   - s > t (future positions) → weight 0 (can't see the future)
#   - s <= t (past + current)  → equal weight, summing to 1 (a plain average)
#
# tril gives the causal mask (1 where s <= t, 0 where s > t). Dividing each
# row by its own sum turns "1 if visible else 0" into "1/(visible count) if
# visible else 0" — a uniform average over exactly the positions that row
# is allowed to see.
tril = torch.tril(torch.ones(T, T))
wei  = tril / tril.sum(1, keepdim=True)
print(f"\n[wei] row 0 (only sees itself)   : {wei[0].tolist()}")
print(f"[wei] row 3 (sees positions 0-3) : {[round(v, 3) for v in wei[3].tolist()]}")

# wei @ x: (T, T) @ (B, T, C) broadcasts over the batch dimension. For each
# batch this is (T, T) @ (T, C) = (T, C) — every position's causal average,
# computed in ONE matrix multiply instead of T sequential steps.
xbow3 = wei @ x
print(f"\n[check] loop vs matmul match  : {torch.allclose(xbow1, xbow3, atol=1e-6)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Method 4: Same Matmul, Built Via Softmax Instead of Division
# ─────────────────────────────────────────────────────────────────────────────
# This produces IDENTICAL numbers to Method 3, but builds `wei` differently —
# and this is the exact scaffold Day 5 reuses for real self-attention.
# Instead of starting from 1s and dividing, start from 0s ("no preference
# yet"), mask out the future with -inf, and let softmax turn that into a
# probability distribution over the visible positions.
#
# Every visible position currently starts at a raw score of 0, so softmax
# makes them all equal — that's why this still comes out to a plain average.
# Day 5's only change: replace these hardcoded 0s with REAL, learned scores
# (how much token t "wants" to attend to token s). Same masking, same
# softmax, same final wei @ x. That one change is self-attention.
wei2 = torch.zeros((T, T))
wei2 = wei2.masked_fill(tril == 0, float('-inf'))   # future positions → -inf
wei2 = F.softmax(wei2, dim=-1)                       # -inf → 0 after softmax; rest → uniform
xbow4 = wei2 @ x

print(f"\n[check] matmul vs softmax-matmul match: {torch.allclose(xbow3, xbow4, atol=1e-6)}")

print("\n[done] Day 4 complete — all four methods agree: loop, cumsum, triangular-matmul, softmax-matmul.")
print("       next: Day 5 — replace the hardcoded uniform wei with real, learned attention scores.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Bigger Toy Data, For the Attention Demo
# ─────────────────────────────────────────────────────────────────────────────
# Day 4's C=2 was big enough to verify the averaging trick by eye, but it's
# too small to see the variance/scaling effect in STEP 8 — head_size can't
# meaningfully exceed C. Bump channels up to 32. T stays 8, so the SAME
# `tril` causal mask built in STEP 4 still applies unchanged below.
C2 = 32
x2 = torch.randn(B, T, C2)
print(f"[data] x2 shape: {tuple(x2.shape)}  (bigger channel count, just for the attention demo)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Query, Key, Value Projections
# ─────────────────────────────────────────────────────────────────────────────
# Three SEPARATE learned linear layers, same input size (C2), same output
# size (head_size). bias=False: pure projections, not shifts — matches the
# original Transformer paper.
#   key   answers "what do I offer, for being searched against"
#   query answers "what am I looking for"
#   value answers "what do I actually contribute, once I'm matched"
# Using the SAME vector for all three roles (Day 4's xbow @ x) would force
# "how do I get found" and "what do I hand over once found" to be identical
# — separate matrices let the model tune each independently.
head_size = 16
key   = nn.Linear(C2, head_size, bias=False)
query = nn.Linear(C2, head_size, bias=False)
value = nn.Linear(C2, head_size, bias=False)

k = key(x2)      # (B, T, head_size)
q = query(x2)    # (B, T, head_size)
v = value(x2)    # (B, T, head_size)
print(f"\n[proj] k/q/v shape: {tuple(k.shape)}  (every token now has its own query, key, and value vector)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Raw Attention Scores + Why They Need Scaling
# ─────────────────────────────────────────────────────────────────────────────
# wei[b, t, s] = q[b, t] . k[b, s] — how much token t wants to attend to
# token s. (B, T, head_size) @ (B, head_size, T) -> (B, T, T).
wei_raw = q @ k.transpose(-2, -1)
print(f"\n[scores] wei_raw shape: {tuple(wei_raw.shape)}")
print(f"[scores] wei_raw variance (unscaled)      : {wei_raw.var():.2f}")

# q.k sums head_size independent, roughly-zero-mean products, so its
# variance grows ~linearly with head_size — purely from summing more terms,
# not stronger signal (verified below: dividing by sqrt(head_size) divides
# the variance by exactly head_size, no more no less — Var(c*X) = c^2*Var(X)
# with c = 1/sqrt(head_size)). Big-magnitude scores push softmax toward
# saturation: nearly all probability mass piles onto one position, attention
# starts behaving like argmax, and gradients to every other position
# vanish. This division is why it's called "SCALED dot-product attention."
wei_scaled = wei_raw * head_size ** -0.5
print(f"[scores] wei_scaled variance (after /sqrt({head_size}))  : {wei_scaled.var():.2f}")

# Make the saturation effect visible directly: softmax one row, unscaled vs
# scaled, and watch how concentrated the resulting probabilities are.
demo_unscaled = wei_raw[0].masked_fill(tril == 0, float('-inf'))
demo_scaled   = wei_scaled[0].masked_fill(tril == 0, float('-inf'))
print(f"\n[softmax] row 7, UNSCALED -> {[round(v, 3) for v in F.softmax(demo_unscaled[7], dim=-1).tolist()]}")
print(f"[softmax] row 7, SCALED   -> {[round(v, 3) for v in F.softmax(demo_scaled[7], dim=-1).tolist()]}")
print("[softmax] unscaled skews toward one dominant spike; scaled spreads attention across more positions.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Causal Mask + Softmax (identical scaffold to STEP 5 above)
# ─────────────────────────────────────────────────────────────────────────────
wei = wei_scaled.masked_fill(tril == 0, float('-inf'))   # future positions -> -inf, same tril as STEP 4
wei = F.softmax(wei, dim=-1)                              # -> real probability distribution per row
print(f"\n[wei] row 3, batch 0 (DATA-DEPENDENT now, not uniform): {[round(v, 3) for v in wei[0, 3].tolist()]}")
print(f"[wei] STEP 4's row 3 for comparison                   : [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Aggregate VALUE Vectors (not the raw input!) Using wei
# ─────────────────────────────────────────────────────────────────────────────
# Day 4 did wei @ x — averaging the raw input. Here we do wei @ v: the
# weights are still decided by Query/Key matching, but what actually gets
# passed forward is the Value projection, not the raw token vector.
out = wei @ v   # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
print(f"\n[output] out shape: {tuple(out.shape)}")
print(f"[output] each position is now a content-aware blend of every position at or before it.")

print("\n[done] Day 5 complete — one self-attention head, working end to end.")
print("       next: Day 6 — multiple heads running in parallel (multi-head attention).")

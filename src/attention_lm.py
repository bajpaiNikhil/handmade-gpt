"""
Detour — A Minimal Self-Attention Language Model on Real Shakespeare
------------------------------------------------------------------------
Days 4-5 (causal_average.py) proved the self-attention mechanism on toy
random tensors — correct math, but disconnected from any real vocabulary
or training. This file is a deliberate detour off the planned roadmap
(which doesn't wire attention into a real trained model until Day 10-11):
a MINIMAL working language model — one attention head, trained on the
real tiny Shakespeare corpus — just to see what changes when "the model
can look back further than one character" stops being a toy idea.

This is NOT Day 10's "Assemble the Block + stack" — that day combines
MULTI-head attention, a FeedForward MLP, residual connections, and
LayerNorm into a real, stacked Transformer block. This is a much smaller
preview: a single head, no MLP, no residuals, no LayerNorm, no stacking.

block_size and batch_size are kept IDENTICAL to bigram.py on purpose, so
the only thing that differs between the two models trained below is the
architecture — making the final loss comparison a fair one.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"[device] running on: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Data, Vocabulary, Split, Batch Sampler  (identical to bigram.py)
# ─────────────────────────────────────────────────────────────────────────────
with open('data/input.txt', 'r') as f:
    text = f.read()

chars      = sorted(set(text))
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}
itos       = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

block_size = 8   # same as bigram.py — kept identical for a fair comparison
batch_size = 4   # same as bigram.py — kept identical for a fair comparison

def get_batch(split):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i     : i + block_size    ] for i in ix])
    y  = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

print(f"[data] vocab_size={vocab_size}  train_tokens={len(train_data):,}  val_tokens={len(val_data):,}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Baseline: Day 2/3's Bigram Model (duplicated here for comparison)
# ─────────────────────────────────────────────────────────────────────────────
# Identical architecture to bigram.py's BigramLanguageModel. Re-defined here
# (rather than imported) so this file can train it side by side with the
# attention model below without re-running bigram.py's entire script.
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — The Attention Model: Token + Position Embeddings, One Head, LM Head
# ─────────────────────────────────────────────────────────────────────────────
# Why a separate POSITION embedding now, when the bigram model never needed
# one: the bigram model's prediction depends only on the current token's
# IDENTITY — "what follows 'F'" is the same question whether 'F' is the 1st
# or 50th character. Attention breaks that. Query/Key dot products only see
# CONTENT — nothing about q[t]·k[s] tells the model whether s is 1 step back
# or 7 steps back, beyond the causal mask's yes/no visibility. Without an
# explicit positional signal, the model has no way to learn position-specific
# patterns (e.g. "capital letters cluster at the start of a line"). So each
# token's embedding becomes token_meaning + position, summed together.
n_embd = 32   # embedding dimension — distinct from vocab_size now (unlike
              # the bigram model, where the embedding table's columns WERE
              # the logits). head_size == n_embd here since there's only
              # one head; splitting n_embd across multiple heads is Day 6.

class Head(nn.Module):
    """One self-attention head — same mechanism proved in causal_average.py."""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # register_buffer (not a plain attribute): tril needs to move to the
        # same device as the rest of the model when .to(device) is called,
        # but it's a fixed mask, not a learnable parameter — register_buffer
        # is exactly "moves with the model, never receives a gradient."
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                    # (B, T, head_size)
        q = self.query(x)                                   # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T), scaled
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)                                    # (B, T, head_size)
        return wei @ v                                       # (B, T, head_size)


class AttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd, n_embd, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                              # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb         # broadcasts (T, n_embd) across the batch dim
        x = self.sa_head(x)           # (B, T, n_embd) — now content-aware, not just per-token
        logits = self.lm_head(x)      # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # CROP to the last block_size tokens before every forward pass.
            # bigram.py never needed this — it only ever used the last
            # token's logits no matter how long idx grew. Here, position
            # embeddings only have block_size rows; feeding more than
            # block_size tokens would index past the end of that table.
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Loss Estimation Helper (Day 3's trick, generalized to take a model)
# ─────────────────────────────────────────────────────────────────────────────
eval_iters    = 200
eval_interval = 1_000
max_iters     = 10_000

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train Both Models Under Identical Conditions
# ─────────────────────────────────────────────────────────────────────────────
def train(model, label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"\n[train] {label}: starting ({max_iters:,} steps)...")
    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model)
            print(f"  [{label}] step {step:5d} / {max_iters}   train loss: {losses['train']:.4f}   val loss: {losses['val']:.4f}")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return estimate_loss(model)


bigram_model = BigramLanguageModel(vocab_size).to(device)
attn_model   = AttentionLanguageModel(vocab_size, n_embd, block_size).to(device)

bigram_params = sum(p.numel() for p in bigram_model.parameters())
attn_params   = sum(p.numel() for p in attn_model.parameters())
print(f"\n[params] bigram model   : {bigram_params:,}")
print(f"[params] attention model: {attn_params:,}")

bigram_final = train(bigram_model, "BIGRAM   ")
attn_final   = train(attn_model,   "ATTENTION")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Compare: Loss + Generated Text, Side by Side
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULT — Bigram (1 token of memory) vs Attention (8 tokens)")
print("=" * 60)
print(f"  bigram    final val loss: {bigram_final['val']:.4f}")
print(f"  attention final val loss: {attn_final['val']:.4f}")
print(f"  difference              : {(bigram_final['val'] - attn_final['val']).item():+.4f}  (positive = attention is better)")

context = torch.zeros((1, 1), dtype=torch.long, device=device)

print("\n[generate] BIGRAM sample:")
print("─" * 60)
print(decode(bigram_model.generate(context, max_new_tokens=300)[0].tolist()))
print("─" * 60)

print("\n[generate] ATTENTION sample:")
print("─" * 60)
print(decode(attn_model.generate(context, max_new_tokens=300)[0].tolist()))
print("─" * 60)

print("\n[done] Detour complete — one real attention head, trained and compared against the Day 2/3 baseline.")
print("       back to the roadmap: Day 6 — multiple heads running in parallel (multi-head attention).")

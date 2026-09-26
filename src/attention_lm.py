"""
Detour + Day 6 — Self-Attention Language Models on Real Shakespeare
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
preview: attention only, no MLP, no residuals, no LayerNorm, no stacking.

Day 6 update: multi_head_attention.py proved MultiHeadAttention on toy
tensors — this file wires it into the same real-data setup the single
head already used, so the loss comparison below is now 3-way: bigram
(1 token of memory) vs single-head (Day 5) vs multi-head (Day 6).

block_size and batch_size are kept IDENTICAL across all three models on
purpose, so architecture is the only thing that differs between them —
making the final loss comparison a fair one.
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
# STEP 3 — Attention Models: Token + Position Embeddings, Attention, LM Head
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
              # the logits).

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


class MultiHeadAttention(nn.Module):
    """Day 6 — several Head instances run in parallel over the same input,
    each with independently-initialized Q/K/V weights so each is free to
    specialize in a different attention pattern. Outputs are concatenated
    back to n_embd width, then passed through one learned projection so the
    model can mix what the different heads found — a plain concat alone
    would just glue their answers side by side with no way to combine them.
    Proved on toy tensors in multi_head_attention.py; this is the same
    class wired into a real trained model."""

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads*head_size)
        return self.proj(out)                                  # (B, T, n_embd)


class SingleHeadAttentionLanguageModel(nn.Module):
    """Day 5 — one attention head, head_size == n_embd."""

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


class MultiHeadAttentionLanguageModel(nn.Module):
    """Day 6 — identical to SingleHeadAttentionLanguageModel except sa_head
    is a MultiHeadAttention (4 heads of head_size=8 instead of 1 head of
    head_size=32). Everything else — embeddings, lm_head, generate — is
    unchanged, since MultiHeadAttention's output shape (B, T, n_embd)
    matches Head's exactly."""

    def __init__(self, vocab_size, n_embd, block_size, num_heads):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(num_heads, n_embd // num_heads, n_embd, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)          # (B, T, n_embd) — blend of 4 independent heads
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


class FeedForward(nn.Module):
    """Day 7 — per-token feed-forward network: n_embd -> 4*n_embd -> GELU ->
    n_embd. Same weights applied identically at every position (proved on
    toy tensors in feedforward.py). This is the "desk-work" step: attention
    only ever produces a LINEAR blend of value vectors, so no matter how
    many heads run, the model still cannot react to what it now holds —
    only mix. FeedForward adds that missing nonlinear, per-token reasoning
    step right after attention's blend."""

    def __init__(self, n_embd, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expansion * n_embd),
            nn.GELU(),
            nn.Linear(expansion * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)   # (B, T, n_embd) -> (B, T, n_embd)


class MultiHeadAttentionFeedForwardLanguageModel(nn.Module):
    """Day 7 — identical to MultiHeadAttentionLanguageModel, with one new
    step: after attention gathers context (communicate), FeedForward
    processes each token's result (compute), before lm_head reads out
    logits. No residual connections or LayerNorm yet — those are Days 8-9,
    and we want a clean before/after loss measurement for the MLP alone."""

    def __init__(self, vocab_size, n_embd, block_size, num_heads):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(num_heads, n_embd // num_heads, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)          # (B, T, n_embd) — COMMUNICATE: gather context
        x = self.ffwd(x)              # (B, T, n_embd) — COMPUTE: react to what was gathered
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


class MultiHeadAttentionFeedForwardResidualLanguageModel(nn.Module):
    """Day 8 — identical to MultiHeadAttentionFeedForwardLanguageModel except
    each sublayer's output is now ADDED back onto its own input (the "express
    lane" from residuals.py) instead of replacing it outright:
    x = x + sa_heads(x)   and   x = x + ffwd(x)
    Proved in residuals.py that this guarantees a gradient path that can
    never be fully crushed no matter how small either sublayer's own
    contribution gets. No LayerNorm or stacking yet (Days 9-10) — this is
    the first NEARLY-complete transformer block."""

    def __init__(self, vocab_size, n_embd, block_size, num_heads):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(num_heads, n_embd // num_heads, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = x + self.sa_heads(x)       # express lane around attention
        x = x + self.ffwd(x)           # express lane around FeedForward
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
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
# STEP 5 — Train All Three Models Under Identical Conditions
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


num_heads = 4   # n_embd(32) // num_heads(4) = head_size(8) — see multi_head_attention.py STEP 4

bigram_model = BigramLanguageModel(vocab_size).to(device)
single_model = SingleHeadAttentionLanguageModel(vocab_size, n_embd, block_size).to(device)
multi_model  = MultiHeadAttentionLanguageModel(vocab_size, n_embd, block_size, num_heads).to(device)
ffwd_model   = MultiHeadAttentionFeedForwardLanguageModel(vocab_size, n_embd, block_size, num_heads).to(device)
resid_model  = MultiHeadAttentionFeedForwardResidualLanguageModel(vocab_size, n_embd, block_size, num_heads).to(device)

bigram_params = sum(p.numel() for p in bigram_model.parameters())
single_params = sum(p.numel() for p in single_model.parameters())
multi_params  = sum(p.numel() for p in multi_model.parameters())
ffwd_params   = sum(p.numel() for p in ffwd_model.parameters())
resid_params  = sum(p.numel() for p in resid_model.parameters())
print(f"\n[params] bigram model            : {bigram_params:,}")
print(f"[params] single-head model       : {single_params:,}")
print(f"[params] multi-head model        : {multi_params:,}")
print(f"[params] multi-head + FFN model  : {ffwd_params:,}  "
      f"({ffwd_params / multi_params:.2f}x multi-head-only — the FeedForward's ~8.3k params, "
      f"as predicted in feedforward.py)")
print(f"[params] multi-head + FFN + residual model: {resid_params:,}  "
      f"(== multi-head+FFN params exactly? {resid_params == ffwd_params} — residuals add ZERO new "
      f"parameters, only a change in wiring)")

bigram_final = train(bigram_model, "BIGRAM     ")
single_final = train(single_model, "SINGLE-HEAD")
multi_final  = train(multi_model,  "MULTI-HEAD ")
ffwd_final   = train(ffwd_model,   "MULTI+FFN  ")
resid_final  = train(resid_model,  "MULTI+FFN+RESID")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Compare: Loss + Generated Text, Five-Way
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULT — Bigram vs Single-Head (D5) vs Multi-Head (D6) vs Multi-Head+FFN (D7) vs +Residual (D8)")
print("=" * 60)
print(f"  bigram              final train/val loss: {bigram_final['train']:.4f} / {bigram_final['val']:.4f}")
print(f"  single-head         final train/val loss: {single_final['train']:.4f} / {single_final['val']:.4f}")
print(f"  multi-head          final train/val loss: {multi_final['train']:.4f} / {multi_final['val']:.4f}")
print(f"  multi-head+FFN      final train/val loss: {ffwd_final['train']:.4f} / {ffwd_final['val']:.4f}")
print(f"  multi-head+FFN+resid final train/val loss: {resid_final['train']:.4f} / {resid_final['val']:.4f}")
print(f"  +residual train/val gap: {(resid_final['val'] - resid_final['train']).item():+.4f}  "
      f"(vs no-residual FFN gap: {(ffwd_final['val'] - ffwd_final['train']).item():+.4f})")
print(f"  +residual vs no-residual (same architecture, just wiring) : "
      f"{(ffwd_final['val'] - resid_final['val']).item():+.4f}  (positive = residual helped)")
print(f"  +residual vs bigram : {(bigram_final['val'] - resid_final['val']).item():+.4f}  (positive = residual model is better)")

context = torch.zeros((1, 1), dtype=torch.long, device=device)

print("\n[generate] BIGRAM sample:")
print("─" * 60)
print(decode(bigram_model.generate(context, max_new_tokens=300)[0].tolist()))
print("─" * 60)

print("\n[generate] SINGLE-HEAD sample:")
print("─" * 60)
print(decode(single_model.generate(context, max_new_tokens=300)[0].tolist()))
print("─" * 60)

print("\n[generate] MULTI-HEAD sample:")
print("─" * 60)
print(decode(multi_model.generate(context, max_new_tokens=300)[0].tolist()))
print("─" * 60)

print("\n[generate] MULTI-HEAD + FFN sample:")
print("─" * 60)
print(decode(ffwd_model.generate(context, max_new_tokens=300)[0].tolist()))
print("─" * 60)

print("\n[generate] MULTI-HEAD + FFN + RESIDUAL sample:")
print("─" * 60)
print(decode(resid_model.generate(context, max_new_tokens=300)[0].tolist()))
print("─" * 60)

print("\n[done] Day 8 complete — residual connections wired around attention and FeedForward,")
print("       trained on real data, compared against Day 7's no-residual version and everything")
print("       before it.")
print("       next: Day 9 — LayerNorm (stabilizing activation magnitudes across the stack).")

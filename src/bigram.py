"""
Day 2 — Bigram Language Model
-------------------------------
We build the simplest possible language model: a bigram.
A bigram model looks at ONE character and predicts the NEXT one.
No memory of anything before that. Just: "given this char, what comes next?"

This will produce terrible output — that's the point.
It proves the full pipeline (data → model → loss → generate) works
before we add any intelligence on Day 4 onward.

Day 3 — Train Loop Cleanup + estimate_loss()
---------------------------------------------
The loss printed during Day 2's training was a single noisy batch (32
examples). estimate_loss() fixes that: it averages loss over many batches,
on BOTH the train split and the never-trained-on val split, under
torch.no_grad(). That gives a clean signal of whether the model is actually
learning, and whether it's overfitting (train improving while val doesn't).
"""

import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Device
# ─────────────────────────────────────────────────────────────────────────────
# MPS = Apple Metal. Lets PyTorch run on the M-series GPU instead of CPU.
# Training on MPS is ~10–20x faster than CPU for tensor operations.
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

print("=" * 60)
print("  DAY 2/3 — BIGRAM LANGUAGE MODEL + LOSS ESTIMATION")
print("=" * 60)
print(f"\n[device] running on: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Data
# ─────────────────────────────────────────────────────────────────────────────
# Same tiny Shakespeare dataset from Day 1.
# ~1.1M characters of plays and sonnets concatenated into one file.
print("\n[data] loading tiny Shakespeare...")

os.makedirs('data', exist_ok=True)
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
if not os.path.exists('data/input.txt'):
    print("[data] not found locally — downloading...")
    urllib.request.urlretrieve(URL, 'data/input.txt')
    print("[data] download complete.")

with open('data/input.txt', 'r') as f:
    text = f.read()

print(f"[data] corpus size : {len(text):,} characters")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Vocabulary
# ─────────────────────────────────────────────────────────────────────────────
# Character-level tokenization: every unique character becomes one token.
# We sort them so the mapping is deterministic across runs.
#
# stoi: char → int  (e.g. 'A' → 1, 'B' → 2, ..., '\n' → 0)
# itos: int  → char (reverse lookup for decoding generated output)
print("\n[vocab] building character vocabulary...")

chars      = sorted(set(text))
print(chars)
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}
itos       = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]           # string → list of ints
decode = lambda l: ''.join([itos[i] for i in l])  # list of ints → string

print(f"[vocab] unique characters : {vocab_size}")
print(f"[vocab] character set     : {repr(''.join(chars))}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Encode & Split
# ─────────────────────────────────────────────────────────────────────────────
# Turn the entire text into one long tensor of integers.
# Then split 90% train / 10% val.
# val_data is never trained on — it tells us if the model ACTUALLY learned
# or just memorised the training set.
print("\n[split] encoding corpus to tensor and splitting...")

data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"[split] train tokens : {len(train_data):,}")
print(f"[split] val tokens   : {len(val_data):,}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Batch Sampler
# ─────────────────────────────────────────────────────────────────────────────
# get_batch picks a random chunk of text and returns:
#   x — the input  (block_size characters)
#   y — the target (same block shifted right by 1)
#
# Every position in a row is a training example:
#   position 0: given [token_0]                         → predict token_1
#   position 1: given [token_0, token_1]                → predict token_2
#   ...and so on.
# So block_size=8 and batch_size=4 → 32 training examples per call.
block_size = 8   # how many characters the model sees at once
batch_size = 4   # how many independent rows we process in parallel

def get_batch(split):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i     : i + block_size    ] for i in ix])
    y  = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — The Bigram Model
# ─────────────────────────────────────────────────────────────────────────────
# A bigram model has exactly ONE learnable thing: a lookup table.
# The table has vocab_size rows and vocab_size columns (65 × 65 = 4,225 numbers).
#
# How it works:
#   - You feed in a token (e.g. token 18 = 'F')
#   - The model looks up row 18 in the table
#   - That row contains 65 numbers (one per possible next character)
#   - The numbers are called LOGITS — higher = the model thinks that char is more likely
#   - We convert logits to probabilities with softmax, then sample
#
# At the start weights are random → all 65 logits roughly equal → loss ≈ 4.17 nats
# (because -ln(1/65) = 4.17; a fair 65-sided die has that much uncertainty)
# Training will push the loss down by making correct next-chars score higher.
print("\n[model] building BigramLanguageModel...")

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # nn.Embedding is just a matrix: (vocab_size, vocab_size)
        # Indexing it with an integer returns that row — this IS the forward pass.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx:     (B, T) — batch of B sequences, each T tokens long
        # targets: (B, T) — the correct next token at each position

        # Lookup: every token in idx fetches its row → shape (B, T, C)
        # C = vocab_size = 65 — one score for each possible next character
        logits = self.token_embedding_table(idx)   # (B, T, C)

        loss = None
        if targets is not None:
            B, T, C = logits.shape

            # cross_entropy expects inputs as (N, C) and targets as (N,).
            # We have (B, T, C) and (B, T), so we merge B and T into one dimension.
            logits  = logits.view(B * T, C)   # (B*T, C) — N = B*T predictions
            targets = targets.view(B * T)     # (B*T,)   — N correct labels
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Autoregressive loop: predict → append → repeat.
        # idx: (B, T) seed context on device
        for _ in range(max_new_tokens):
            # Forward pass — get scores for every position
            logits, _ = self(idx)              # (B, T, C)

            # Bigram only cares about the LAST token (it has no memory of the rest)
            logits = logits[:, -1, :]          # (B, C) — last time step only

            # Turn raw scores into a probability distribution
            probs = F.softmax(logits, dim=-1)  # (B, C) — sums to 1 across C

            # Sample one token from the distribution (not argmax — that produces loops)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Glue the new token onto the growing sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

        return idx   # (B, T + max_new_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Instantiate & Sanity Check
# ─────────────────────────────────────────────────────────────────────────────
model = BigramLanguageModel(vocab_size).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"[model] embedding table : {vocab_size} × {vocab_size}")
print(f"[model] total parameters: {total_params:,}  (this is tiny — a real GPT has billions)")

# Run one forward pass before any training.
# Loss should be ≈ 4.17 — the model has no clue yet, so it spreads probability evenly.
xb, yb = get_batch('train')
_, untrained_loss = model(xb, yb)
print(f"\n[sanity] untrained loss : {untrained_loss.item():.4f}  (random baseline = 4.17)")
print(f"[sanity] if this is wildly off, something is wrong before training even starts.")

# Print row 18 = token 'F' before training.
# These are 65 random numbers — no pattern, no meaning yet.
f_token = stoi['F']   # should be 18
row_before = model.token_embedding_table.weight[f_token].detach().cpu()
print(f"\n[embed] row for 'F' (token {f_token}) BEFORE training:")
print(f"        {[round(v, 3) for v in row_before.tolist()]}")
print(f"        → all roughly similar small random values — model has no idea what follows 'F'")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Loss Estimation Helper  (Day 3)
# ─────────────────────────────────────────────────────────────────────────────
# Problem: a single get_batch() loss is noisy. batch_size=4 * block_size=8
# = 32 examples — a tiny, high-variance sample. One step printing
# "loss: 2.3" could just be a lucky (or unlucky) batch, not real progress.
#
# Fix: average the loss over many batches before reporting it — for BOTH
# splits:
#   train loss → is the model fitting the data it has seen?
#   val loss   → is it generalising to text it has NEVER trained on?
# val_data has existed since Day 1's split, but nothing has read it until
# now. If train loss keeps dropping while val loss stalls or rises, that's
# overfitting — memorising instead of learning patterns. Unlikely for a
# 4,225-parameter table, but the check is free, and it'll matter once the
# model has millions of parameters.
#
# @torch.no_grad() disables autograd for everything inside the function.
# This is a measurement, not a training step — there's no loss.backward()
# or optimizer.step() in here. Without no_grad, PyTorch would still build a
# full backward graph for every eval batch: wasted memory and time for
# gradients that would never be used.
eval_iters    = 200    # batches to average per split — higher = less noisy, slower
eval_interval = 1_000  # how often (in training steps) to run this check
max_iters     = 10_000

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()    # tells the model "we're evaluating, not training". A no-op
                    # for this bigram table — no dropout, no layernorm yet — but
                    # that flips once those land (Day 9+), so the habit starts here.
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()   # switch back before resuming the training loop
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Training Loop
# ─────────────────────────────────────────────────────────────────────────────
# AdamW: an improved version of gradient descent.
# Each step it looks at the loss gradient and nudges the weights in the right direction.
# lr (learning rate) = how big a nudge per step. 1e-3 is a safe default for small models.
print(f"\n[train] starting training ({max_iters:,} steps)...")
print(f"        each step processes {batch_size * block_size} characters")
print(f"        checking train/val loss every {eval_interval:,} steps (avg of {eval_iters} batches)\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(max_iters):
    # Periodic check-in: a clean, low-variance reading instead of one noisy batch.
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"  step {step:5d} / {max_iters}   train loss: {losses['train']:.4f}   val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')        # fresh random batch
    logits, loss = model(xb, yb)       # forward pass → get loss

    optimizer.zero_grad(set_to_none=True)  # clear old gradients (they accumulate otherwise)
    loss.backward()                         # backprop: compute how much each weight contributed to the loss
    optimizer.step()                        # update weights in the direction that lowers loss

final_losses = estimate_loss()
# TODO Day 6+: replace 0.05 with a noise-aware threshold using losses.std() — see Day 3 discussion
gap = (final_losses['val'] - final_losses['train']).item()
print(f"\n[train] done.  final train loss: {final_losses['train']:.4f}   final val loss: {final_losses['val']:.4f}")
print(f"[train] train/val gap: {gap:+.4f}  ({'no real overfitting' if gap < 0.05 else 'val notably worse — overfitting'})")
print(f"[train] the model has learned letter frequencies and common pairs.")
print(f"[train] it still has NO memory beyond one character — that's Day 5's job.")

# Print the same row 18 after training.
# Now the numbers have spread apart — high scores for chars that commonly follow 'F'
# in Shakespeare (like 'i', 'o', 'r', 'a'), low scores for unlikely ones ('Q', 'x', 'z').
row_after = model.token_embedding_table.weight[f_token].detach().cpu()
print(f"\n[embed] row for 'F' (token {f_token}) AFTER training:")
print(f"        {[round(v, 3) for v in row_after.tolist()]}")

# Show the top 5 predicted next characters after 'F'
top5 = row_after.topk(5)
print(f"\n[embed] top 5 chars the model thinks follow 'F':")
for rank, (score, idx) in enumerate(zip(top5.values.tolist(), top5.indices.tolist()), 1):
    print(f"        #{rank}  '{itos[idx]}'  (score {score:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Generate Text
# ─────────────────────────────────────────────────────────────────────────────
# Seed with token 0, which maps to '\n' (newline) — a neutral starting point.
# The model then rolls forward one character at a time.
# Output will look like scrambled Shakespeare: right letter frequencies,
# occasional real words, no coherent sentences. That's expected.
print("\n[generate] sampling 500 characters from the trained model...")
print("[generate] seed token: 0 (newline character '\\n')")
print("[generate] expect: English-shaped gibberish — right frequencies, wrong meaning.\n")

context   = torch.zeros((1, 1), dtype=torch.long, device=device)  # (1, 1) — batch of 1, length 1
generated = model.generate(context, max_new_tokens=500)[0].tolist()

print("─" * 60)
print(decode(generated))
print("─" * 60)

print("\n[done] Day 3 complete.")
print("       next: Day 4 — weighted-average-of-the-past trick")

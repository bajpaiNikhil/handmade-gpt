import os
import urllib.request
import torch

# ── device ───────────────────────────────────────────────────────────────────
# Use Apple MPS (Metal) if available, otherwise fall back to CPU.
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}")

# ── data ─────────────────────────────────────────────────────────────────────
# Tiny Shakespeare: ~1M chars of plays/sonnets concatenated into one text file.
# We use it because it's small (trains in minutes), structured (so bad output
# is obvious), and has a known reference loss from Karpathy's work.
os.makedirs('data', exist_ok=True)
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
if not os.path.exists('data/input.txt'):
    urllib.request.urlretrieve(URL, 'data/input.txt')

with open('data/input.txt', 'r') as f:
    text = f.read()

print(f"corpus: {len(text):,} chars")

# ── vocab ─────────────────────────────────────────────────────────────────────
# We work at the CHARACTER level — every unique char becomes one token.
# sorted(set(text)) gives us all 65 unique chars in consistent order.
# stoi: char → int   ("A" → 0, "B" → 1, …)
# itos: int  → char  (0 → "A", 1 → "B", …)
chars      = sorted(set(text))
vocab_size = len(chars)          # 65 for tiny Shakespeare
print(f"vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]           # "Hi" → [32, 47]
decode = lambda l: ''.join([itos[i] for i in l])  # [32, 47] → "Hi"

# ── encode full dataset ───────────────────────────────────────────────────────
# Turn the entire text into one long tensor of integers.
# Every character is now a number the model can do math on.
data = torch.tensor(encode(text), dtype=torch.long)

# ── train / val split ─────────────────────────────────────────────────────────
# 90% for training, 10% held out to measure if the model is actually learning
# or just memorising. We never train on val_data.
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"train tokens: {len(train_data):,}  |  val tokens: {len(val_data):,}")

# ── hyperparams ───────────────────────────────────────────────────────────────
block_size = 8   # how many characters the model can look back at (context window)
batch_size = 4   # how many independent sequences we process in parallel

# ── batch sampler ─────────────────────────────────────────────────────────────
# This is the core data feeding mechanism. Here's what it produces:
#
# Suppose the data at some random offset looks like this:
#   d[i .. i+8] = [18, 47, 56, 57, 58,  1, 15, 47, 58]
#                  ^--- x (8 chars) ----^  ^--- one extra char
#
#   x = [18, 47, 56, 57, 58,  1, 15, 47]   ← what the model SEES
#   y = [47, 56, 57, 58,  1, 15, 47, 58]   ← what the model must PREDICT
#
# y is just x shifted one position to the right.
# Every position in the row is a separate training example:
#   position 0: given [18]                   → predict 47
#   position 1: given [18, 47]               → predict 56
#   position 2: given [18, 47, 56]           → predict 57
#   ...
#   position 7: given [18, 47, 56, 57, 58, 1, 15, 47] → predict 58
#
# So one row of length 8 gives us 8 training examples for free.
# With batch_size=4 rows, one call to get_batch() = 32 examples.
def get_batch(split):
    d  = train_data if split == 'train' else val_data
    # pick batch_size random start positions (never so close to the end that
    # x+y would run off the edge of the tensor)
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i     : i + block_size    ] for i in ix])  # (4, 8)
    y  = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])  # (4, 8) shifted by 1
    return x.to(device), y.to(device)

# ── smoke test ────────────────────────────────────────────────────────────────
xb, yb = get_batch('train')
print("x shape:", xb.shape)   # (4, 8) — 4 rows, 8 chars each
print("y shape:", yb.shape)   # (4, 8) — same shape, content shifted right by 1
print("first x:", decode(xb[0].tolist()))
print("first y:", decode(yb[0].tolist()))
# Notice: first y is first x shifted by one character.
# The char at y[0][0] is the char that comes AFTER x[0][0] in the original text.

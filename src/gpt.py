import os
import urllib.request
import torch

# ── device ──────────────────────────────────────────────────────────────────
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}")

# ── data ─────────────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
if not os.path.exists('data/input.txt'):
    urllib.request.urlretrieve(URL, 'data/input.txt')

with open('data/input.txt', 'r') as f:
    text = f.read()

print(f"corpus: {len(text):,} chars")

# ── vocab ─────────────────────────────────────────────────────────────────────
chars      = sorted(set(text))
vocab_size = len(chars)
print(f"vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ── encode full dataset ───────────────────────────────────────────────────────
data = torch.tensor(encode(text), dtype=torch.long)

# ── train / val split ─────────────────────────────────────────────────────────
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"train tokens: {len(train_data):,}  |  val tokens: {len(val_data):,}")

# ── hyperparams ───────────────────────────────────────────────────────────────
block_size = 8   # context length (will grow every world)
batch_size = 4

# ── batch sampler ─────────────────────────────────────────────────────────────
def get_batch(split):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i     : i + block_size    ] for i in ix])
    y  = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ── smoke test ────────────────────────────────────────────────────────────────
xb, yb = get_batch('train')
print("x shape:", xb.shape)          # (4, 8)
print("y shape:", yb.shape)          # (4, 8)
print("first x:", decode(xb[0].tolist()))
print("first y:", decode(yb[0].tolist()))

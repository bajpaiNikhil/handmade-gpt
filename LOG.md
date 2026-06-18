# Build Log

Newest entry on top.

---

## Day 2 — Bigram language model

Built `src/bigram.py` as a standalone Day 2 file (separate from the Day 1 data
pipeline in `gpt.py`). Added `BigramLanguageModel`: a single `nn.Embedding(65, 65)`
lookup table — 4,225 parameters total. Implemented `forward()` with
`F.cross_entropy` loss and `generate()` autoregressive loop. Trained for 10,000
steps with AdamW (lr=1e-3) on MPS. Loss dropped from ≈4.17 (random baseline,
-ln(1/65)) to ≈2.6. Added before/after inspection of row 18 (`'F'`) — confirmed
training pushed scores for `'r'`, `'o'`, `'a'` high and crushed unlikely chars to
≈-3. Generated output is English-shaped gibberish: correct letter frequencies,
no coherent words — expected for a one-character memory model.

---

## Day 1 — data pipeline, char vocab, get_batch

Downloaded tiny Shakespeare (1,115,394 chars). Built character-level vocab:
65 unique chars, `stoi`/`itos` dicts, `encode`/`decode` lambdas. Encoded full
corpus to a `torch.long` tensor, 90/10 train/val split. Wrote `get_batch(split)`
— random offset sampler returning `(x, y)` pairs where `y` is `x` shifted by one.
Smoke test: `x`/`y` shapes `(4, 8)` on MPS confirmed clean.

---

## Day 0 — repo scaffolded, torch+MPS verified (mps:0)

Initialized project structure. Confirmed PyTorch MPS backend is available and a
tensor lands on `mps:0` without errors. Ready to start typing.

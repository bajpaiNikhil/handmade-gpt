# Build Log

Newest entry on top.

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

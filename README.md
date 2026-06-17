# handmade-gpt

**Building a language model from scratch on a MacBook, one hand-typed layer at a time.
No `nn.MultiheadAttention`, no magic. Follow the build →**

---

## Progress

A five-world campaign from a blank tensor to an on-device language model.

### W1 — The Skeleton *(data → bigram → first attention head)*
- [ ] Day 1 — Setup + data pipeline
- [ ] Day 2 — Bigram model
- [ ] Day 3 — Train loop + read the loss
- [ ] Day 4 — Weighted-average-of-the-past trick
- [ ] Day 5 — First self-attention head

### W2 — The Block *(multi-head, MLP, residuals, layernorm, scale → working GPT)*
- [ ] Day 6 — Multi-head attention
- [ ] Day 7 — FeedForward MLP
- [ ] Day 8 — Residual connections
- [ ] Day 9 — LayerNorm (pre-norm)
- [ ] Day 10 — Assemble the Block + stack
- [ ] Day 11 — Scale + train

### W3 — The Modern Recipe *(RMSNorm, RoPE, SwiGLU)*
- [ ] Milestone: Llama-shaped architecture

### W4 — Real Language *(hand-written BPE tokenizer, TinyStories)*
- [ ] Milestone: real tokenizer + real dataset

### W5 — On Device *(quantize to GGUF, run on Android)*
- [ ] Milestone: inference on phone

---

**Build log:** [LOG.md](LOG.md)

---

## The rules

- **PyTorch only** — tensors and autograd, nothing else from the framework.
- **Every layer hand-written** — no `nn.MultiheadAttention`, no `nn.Transformer`,
  no `nn.LayerNorm`. If it's a building block of a transformer, I type it myself.
- One session per day. No skipping ahead.

The point is to understand every line, not just to have a model.

---

## Want to follow along / contribute?

Issues and PRs welcome. If you spot a bug in my math or know a cleaner way to
express something without hiding the mechanics — open an issue.

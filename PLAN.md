# handmade-gpt — Full Build Plan

A day-by-day campaign to build a GPT-style language model from scratch on an M4 Pro.
One session per day. Every layer hand-typed.

---

## World 1 — The Skeleton

> Data pipeline → bigram baseline → first self-attention head.

| Day | Goal |
|-----|------|
| 1 | Setup + data: download tiny Shakespeare, write the character-level tokenizer (encode/decode), slice into train/val splits |
| 2 | Bigram model: `nn.Embedding` lookup table, forward pass, cross-entropy loss, first generation |
| 3 | Train loop + read the loss: AdamW, batch loop, track train/val loss, understand what the numbers mean |
| 4 | Weighted-average-of-the-past trick: implement the causal averaging trick manually; derive why we need attention |
| 5 | First self-attention head: `query`, `key`, `value` projections, scaled dot-product, causal mask, dropout |

---

## World 2 — The Block

> Stack multi-head attention + MLP + residuals + normalization → working GPT.

| Day | Goal |
|-----|------|
| 6 | Multi-head attention: split embedding dim across H heads, concat, project out |
| 7 | FeedForward MLP: two linear layers, ReLU/GELU, the 4× expansion heuristic |
| 8 | Residual connections: add skip paths around attention and MLP, understand why they matter |
| 9 | LayerNorm (pre-norm): hand-write `LayerNorm`, place it before each sub-layer |
| 10 | Assemble the Block + stack: `Block = LayerNorm → Attention → LayerNorm → MLP`, stack N blocks |
| 11 | Scale + train: push to ~10M params, tune LR/batch/context length, generate coherent Shakespeare |

---

## World 3 — The Modern Recipe

Swap in the upgrades that separate GPT-2 from Llama:

- **RMSNorm** instead of LayerNorm (simpler, faster)
- **Rotary Position Embeddings (RoPE)** instead of learned absolute positions
- **SwiGLU** activation instead of GELU in the MLP
- **Grouped-Query Attention (GQA)** — optional stretch goal

---

## World 4 — Real Language

- Hand-write a **BPE tokenizer** (merge-rules loop, vocab builder, encode/decode)
- Switch dataset to **TinyStories** (~2M tokens of simple English)
- Retrain from scratch with the new tokenizer + modern architecture

---

## World 5 — On Device

- Quantize weights to **8-bit / 4-bit** (hand-written or via `llama.cpp` GGUF export)
- Run inference **on Android** via an NNAPI / CPU backend
- Stretch: CoreML export for on-device iOS inference

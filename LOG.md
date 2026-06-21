# Build Log

Newest entry on top.

---

## Detour — one real attention head vs. the Day 2/3 bigram, on real Shakespeare

New file `src/attention_lm.py`. Days 4-5 only proved the attention mechanism
on toy `torch.randn` tensors — this wires a single real head into a trainable
model (token + positional embeddings, one `Head`, `lm_head`) and trains it on
the real tiny-Shakespeare corpus, side by side with a freshly-trained Day 2/3
bigram model under identical `block_size=8`, `batch_size=4`. Result: attention
final val loss 2.4552 vs bigram's 2.5402 — modest on the final number, but
attention reached near-final loss by step ~2000-3000 while bigram needed
9000+. Checked the train/val gap for both (-0.009 and +0.006) to confirm the
win is generalization, not the attention model's extra parameters (7,553 vs
4,225) just memorizing. Generated text is still gibberish for both — expected
with no MLP, residuals, LayerNorm, or multiple heads yet. Not part of the
official roadmap (Day 10-11 do the real assembly) — purely a sanity check
taken as a deliberate detour off-roadmap.

---

## Day 5 — a single self-attention head

Extended `causal_average.py` (not a new file — this upgrades Day 4's own
`wei` matrix rather than introducing a separate concept). Replaced the
hardcoded uniform scores with real ones via Query/Key/Value:
`wei[t,s] = q[t]·k[s]`, scaled by `1/sqrt(head_size)`, causally masked,
softmaxed, then aggregated as `wei @ v` (Value, not the raw input). Confirmed
the scaling math on real numbers: unscaled wei variance 2.08, scaled 0.13 —
exactly variance/head_size(16), matching Var(c·X)=c²·Var(X). Watched softmax
saturation directly: an unscaled row put 49% of its mass on one position;
scaled spread it 0.09-0.20 across the row. Day 4's uniform row-3 weights
`[0.25, 0.25, 0.25, 0.25]` became content-dependent `[0.21, 0.15, 0.52, 0.12]`.

---

## Day 4 — the weighted-average-of-the-past trick

New file `src/causal_average.py`. Proved four ways of computing a causal
running average agree exactly: a double-loop (correct, slow), `cumsum`
(vectorized, O(T), no (T,T) matrix), a triangular-matrix matmul
(`wei = tril / tril.sum(1)`, `wei @ x`), and a softmax-over-masked-zeros
version of that same matmul. Ruled out an RNN-style recurrent running summary
first — same shape as Kadane's algorithm, but sequential, so it can't be
parallelized on a GPU. Kept the matmul version over the more efficient
`cumsum` specifically because it exposes a `(T, T)` weight matrix that Day 5
upgrades from fixed to learned.

---

## Day 3 — train loop cleanup + estimate_loss()

Added `estimate_loss()` to `bigram.py`: averages loss over `eval_iters=200`
batches per split, under `torch.no_grad()`, replacing the single noisy
training-batch loss Day 2 printed. Put `val_data` to use for the first time
since the Day 1 split — confirmed empirically that a single-batch reading
(4.9046) and a 200-batch average (4.6666) of the *same* untrained model
disagree meaningfully. Extracted `max_iters`/`eval_interval` as named
constants (the latter is load-bearing: `step == max_iters - 1` forces a final
clean eval). Final run: train loss 2.5453, val loss 2.5313, gap +0.0140 — no
overfitting, as expected for a 4,225-parameter table.

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

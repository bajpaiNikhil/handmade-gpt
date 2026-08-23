# Concepts index — the running glossary

Every session appends to this list. Claude adds new terms at session end
(concise: term → one- or two-line meaning → where it was first used). Kept in
no particular order; it grows by date.

## Seeded — Days 0–6 (session 1 target)

- **Character-level tokenizer / vocabulary** — mapping the corpus's 65 unique
  chars to ints (stoi/itos); the model predicts the next *char* index.
- **Embedding (nn.Embedding)** — a lookup table: token index → dense vector.
  "A linear map whose basis is one-hot."
- **Autograd** — PyTorch's automatic chain rule: records the computational
  graph in a forward pass, then `loss.backward()` fills every parameter's
  `.grad`.
- **Cross-entropy loss** — `-log p(y_true)`; minimized when the model puts mass
  on the true token. `−ln(1/65)=4.17` is the uniform/random baseline.
- **Val loss / estimate_loss** — loss averaged over many held-out batches under
  `torch.no_grad()`; the "how far from truth" number we trust over single-batch
  noise.
- **nats vs bits** — cross-entropy in natural log units (×1/ln 2 → bits).
- **Perplexity** — `exp(val_loss)`; ≈ the average number of "plausible next
  chars" the model entertains.
- **Causal mask (tril)** — a (T, T) lower-triangular 0/1 mask that forbids
  position t from attending to s > t (no future peeking).
- **Query / Key / Value** — per-token projections that answer, respectively,
  "what am I looking for", "what do I offer when searched", "what do I hand
  over once matched".
- **Scaled dot-product attention** — `softmax(q·kᵀ/√d) @ v`; the `/√d` keeps
  score variance ~1 so softmax doesn't saturate.
- **Attention weights `wei`** — the (B, T, T) probability rows telling how much
  each position blends each past position's value.
- **Multi-head attention** — H independent heads in parallel (head_size =
  n_embd/H), concatenated then projected (`proj`) so answers can mix across
  heads; same Q/K/V parameter budget as one wide head.
- **Dropout** — a training-time regularizer that randomly zeroes a fraction of
  activations/weights; a no-op in eval mode.

## Appended by later sessions

### Day 6 redo (Mon Aug 10)

- **Softmax saturation** — when raw scores are large in magnitude, softmax puts
  almost all mass on one position; every other position's gradient collapses
  toward zero, so backprop can no longer teach the head where to look. Measured:
  unscaled, a 4-way row went 0.9143/0.0847/0.0009/0.0000 with the last
  position's gradient at −5e-06 vs −0.039 scaled (~8,000×).
- **Parameter parity (head_size = n_embd / H)** — since `H · head_size = n_embd`
  by construction, total Q/K/V parameters are the same at any head count
  (3,072 for both 1×32 and 4×8 at n_embd=32). The head count is a free knob:
  it chooses how many independent questions to spend a fixed budget on.
- **`proj` (output projection)** — the `nn.Linear(n_embd, n_embd)` after the
  concat. `torch.cat` is pure memory layout and does no math, so `proj` is the
  only place information mixes *across* heads. Costs `32·32+32 = 1,056` — the
  entire price of multi-head over single-head.
- **Shape preservation / residual-ready** — a multi-head layer maps
  `(B,T,n_embd) → (B,T,n_embd)`, which is what makes Day 8's `x = x + attn(x)`
  legal at all (elementwise addition needs matching shapes).
- **`nn.ModuleList`** — registers submodules so `.parameters()` and `.to(device)`
  see them. A plain Python list of layers registers nothing: the optimizer never
  updates them and they never move to the device.
- **Symmetry breaking** — heads with identical weights receive identical
  gradients and stay identical forever. Random initialization is the only thing
  that ever makes heads different; training then widens the gap.
- **Measurement noise floor** — two `estimate_loss` calls on the *same* weights
  differ by ~0.01–0.03 nats (sampling different batches). Real model differences
  smaller than that are unresolvable in a single unseeded run.

(Space for Days 7–11, W3–W5 terms — Claude appends as sessions complete.)
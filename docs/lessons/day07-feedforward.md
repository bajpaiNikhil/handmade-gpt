# Day 7 — The FeedForward MLP ("attention communicates, MLP computes")

## Session card

- **Goal:** understand and build the per-token FeedForward layer; wire it into
  the language model and measure what it adds.
- **Files:** new `src/feedforward.py` (toy proof), then extend the wiring to a
  new model inside `src/attention_lm.py` (or a sibling file) and retrain.
- **Numbers already logged (baseline to beat):** multi-head 2.3217 (8,609 params).
- **Prerequisites:** Days 1–6. Uses the trained-state of the day-6 comparison.

---

## Part 1 — LECTURE (teach first)

### 1.0 Where we are

Every model so far is *attention only*. Tokens now gather context from up to 8
positions behind them (multi-head 2.3217). The output is still gibberish, and
there is one structural reason that is holding us back: **attention alone is
linear mixing of vectors, per token.**

### 1.1 The division of labor — "communicate, then compute"

The standard mental model: information must (1) *move between* tokens, then
(2) be *processed* where it lands.

- **Attention** does the moving. It takes a weighted average of other tokens'
  value vectors. But a weighted average is a **linear** combination — no matter
  how many heads, at the end of attention every token has produced linear
  blends. There is no non-linear function of the *contents* beyond the softmax
  weighting itself.

WRITE THIS DOWN: **attention exchanges information; it does not compute a
nonlinear function of a single token's information.**

- **The FeedForward MLP** does the computing. Applied **identically to every
  token position** (same weights, per-position), it is the layer that can
  transform each token's aggregated representation into a richer, non-linear
  one — pattern recognition, memorized facts, feature combination.

Concretely: two linear layers with a non-linearity between them:
`n_embd -> 4·n_embd -> GELU -> n_embd`.

### 1.2 Why per-token (weight sharing across positions)

One shared MLP applied independently to each of the `T` positions. This is
parametric efficiency: whatever the MLP "knows" (facts, transformations) is
position-independent and can be applied anywhere, and it is trained on `B·T`
examples at once. It is the same idea as a `1x1` conv / a pointwise
transformation.

### 1.3 Why 4× — the expansion heuristic

The original "Attention Is All You Need" paper picked `d_ff = 4 · d_model`
(2048 vs 512). The intermediate is deliberately *wider* than the residual
stream: it gives the layer a high-dimensional working space to "spread thoughts
out" in before collapsing back to `n_embd`. This is a heuristic that survived
into nearly every modern model — Llama, GPT-2/3, etc. all run ≈4× (the modern
recipe swaps the activation, not the width; that is World 3).

### 1.4 Where the model's "knowledge" lives

A useful lens (Geva et al., "Transformer Feed-Forward Layers Are Key-Value
Memories"): think of the first linear layer's rows as **keys** (patterns) and
the second linear layer's rows as **values** (answers). The activation picks
which keys the input matches; the second layer reads out the corresponding
answers. Under this view the MLP is an associative memory, and it is why the
MLP holds roughly **two-thirds of a transformer's parameters** — measure it on
our own model once it exists.

WRITE THIS DOWN: **FFN ≈ per-token nonlinear processing; in the "memories"
view it is where facts live. It holds ~2/3 of parameters.**

### 1.5 Why GELU (and not ReLU)

ReLU works; GELU is a smooth approximation that keeps a small gradient for
negative inputs (no dead neurons) and behaved better empirically in modern LLMs.
`F.gelu` is one line. We hand-type the layer *shape*; using `F.gelu` as the
pointwise function is fine (we are not hiding a transformer building block —
the interest of the project is the block architecture).

### 1.6 The training-side risk: overfitting shows up here first

The bigram had 4,225 params; our attention models ~7.5–8.6k. Adding an MLP
adds `2 · n_embd · 4n_embd ≈ 8·n_embd²` params — at n_embd=32 that is ~8,700,
a ~2× jump in one day. That means the value of the MLP must be weighed against
the train/val gap growing. This is the first day the loss-vs-overfit tradeoff
is visible; we will watch train and val *together* (Day 3´s `estimate_loss`
exists for exactly this).

---

## Part 2 — QUESTIONS (ask, wait, validate, correct)

**Q1.** Why is attention *alone* insufficient — what specific mathematical
property of "weighted average" is the hole the MLP fills?

**Q2.** The MLP is applied at every position with the same weights. What does
that buy us, and what would it mean if each position had its *own* MLP?

**Q3.** Where do ~2/3 of a transformer's parameters live, and under the
"key-value memory" lens what does each of the two linear layers represent?

**Q4.** Roughly how many parameters does an MLP with `n_embd=32` and a 4×
expansion add? Show the arithmetic.

**Answer key:**
- **A1.** A weighted average is linear in the vectors being blended — it cannot
  express nonlinear transformations of a token's own aggregated content. The
  model needs a per-token nonlinear function to actually *use* the gathered
  context. (The softmax weights are nonlinear, but the aggregation `Σ wei·v` is
  still a linear combination.)
- **A2.** Weight sharing across positions: the same computation applies at any
  position, trained on B·T examples, position-independent ("a fact works no
  matter where it appears"), ~T times fewer parameters than per-position
  weights.
- **A3.** In the MLPs (×number of layers). First linear = keys (patterns the
  input can match), second linear = values (answers read out); the activation
  between them selects which keys match.
- **A4.** Two matrices: `(32→128)` and `(128→32)`, plus their biases:
  `32·128 + 128 + 128·32 + 32 = 4096 + 128 + 4096 + 32 = 8,352`. Roughly
  `8·n_embd²` — about equal to all of our attention models so far.

---

## Part 3 — BUILD (step by step, each explained)

We hand-type a new file `src/feedforward.py`. Reuse the repo's toy-first
convention.

**STEP 1 — toy setup:** `B, T, n_embd = 4, 8, 32`; `x = torch.randn(...)`.
- *What/Why:* same shapes as Day 6 so multi-head and FFN experiments compose.
  Demonstrate the MLP is *per-token*: input and output both `(B, T, n_embd)`.

**STEP 2 — the `FeedForward` class:**
```python
class FeedForward(nn.Module):
    def __init__(self, n_embd, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expansion * n_embd),
            nn.GELU(),
            nn.Linear(expansion * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)
```
- *What:* two linears with GELU between; expansion factor 4.
- *Why expansion as an argument:* Day 11 will tune it; knobs are config, not
  constants.
- *Why GELU:* section 1.5.

**STEP 3 — the "position invariance" proof:**
- *What/Why:* run the network and verify output shape `(B, T, n_embd)`, then
  run **the same input repeated at two different positions** (e.g. `x[:, 0]`
  vs `x[:, 3]`, flip input order) and confirm the network treats them by
  content, not by slot. This makes "applied identically per position" tangible.

**STEP 4 — parameter check:** print `sum(p.numel() for p in ffn.parameters())`
and confirm it matches the Q4 arithmetic (8,352 for n_embd=32). Also print the
fraction of a full model that now lives in the FFN once one is attached.

**STEP 5 — wire it in:** in `src/attention_lm.py` (or a sibling file), add a
`BlocklessModel`: token+pos embeddings → `MultiHeadAttention` → `FeedForward`
→ `lm_head`. Keep `block_size=8`, `batch_size=4`, 10,000 steps so the
comparison stays fair against 2.3217.
- *Why this order:* gather (attention) then process (MLP) — the communicate-
  compute loop, first appearance. Note we are *not* adding residuals/LN yet;
  Days 8–9 come right after and we want clean before/after measurements.
- *Expected:* a loss clearly below 2.3217 if the MLP earns its keep, OR a loss
  that stalls if 4× at n_embd=32 is overkill at this scale — either outcome is
  a finding to record, not a failure.

**STEP 6 — train, compare, sample:** reuse the `train()` + `estimate_loss()`
helpers; print train/val side by side (watch the gap for overfitting); generate
a sample and compare its plausibility vs the Day 6 multi-head sample.

---

## Part 4 — VERIFY

- Shapes: FFN output `(B, T, n_embd)` == input shape.
- Params: FFN = 8,352; fraction-of-model now ~2/3+.
- Loss: record new val loss vs 2.3217 baseline and the train/val gap.
- Write the numbers into the LOG entry.

---

## Part 5 — QUIZ (answers for Claude)

- **Q1.** "Communicate vs compute": in one sentence each, what does attention
  do for a token and what does the MLP do for it? (Move info / transform info.)
- **Q2.** Why width 4× before collapsing back? (High-dim working space; paper
  heuristic that persisted; not a law, a knob.)
- **Q3.** Why is the MLP the same across positions, and what would change if it
  weren't? (Efficiency + position-independence; per-position would blow up
  params and lose generalization.)
- **Q4.** What risk first appears when we add the FFN, and what number tells us
  it is happening? (Overfitting; growing train/val gap in `estimate_loss`.)
- **Q5.** Where is the MLP's "memory" in the key-value lens? (First layer rows
  = keys; second layer rows = values.)

---

## Part 6 — LOG entry, notes, look-ahead

- **LOG:** "Day 7 — FeedForward MLP" entry in the user's voice: what was built,
  the position-invariance proof, param count, new val loss vs 2.3217, train/val
  gap, sample comparison, and the "communicate vs compute" mental model in one
  line.
- **Notes:** fill `docs/notes/day07-feedforward.md`; my-words summary written by
  the user in conversation; Claude completes + corrects; user approves.
- **Look-ahead:** Day 8 — residual connections: `x = x + sublayer(x)`. This is
  what lets us stack these blocks without gradients dying, and Day 8's
  experiment (gradient norms with vs without) will make that real.
- **Stretch:** skim Geva et al. on FFN as key-value memories; or measure, on
  our own model, the FFN's share of parameters and confirm the ~2/3 claim
  yourself.
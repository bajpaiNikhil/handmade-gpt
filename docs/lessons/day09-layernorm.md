# Day 9 — LayerNorm, hand-written, pre-norm

## Session card

- **Goal:** derive LayerNorm's formula from the definitions of mean and
  variance, hand-type it (no `nn.LayerNorm`), verify it matches `nn.LayerNorm`,
  and measure what it does to a stacked model's training stability.
- **Files:** new `src/layernorm.py` (hand-written LN proof + stability
  experiment), then wire LN into the block from Day 8.
- **Numbers to relate:** Day 8 val loss; expect the LN version to be at least
  as good and noticeably *stabler* in the loss curve / gradient norms.
- **Prerequisites:** Days 1–8. Re-derives mean/variance from zero.

---

## Part 1 — LECTURE (teach first)

### 1.0 Where we are

From Day 10 we stack blocks, and stacking creates a new failure mode: layer
after layer, the distribution of activations **drifts** — components grow,
shrink, shift — and a model trained with drifted statistics is unstable
(exploding/vanishing at the *forward* pass level this time, not just the
gradient level). Residuals fixed the gradient path; normalization fixes the
**activation magnitude drift**.

### 1.1 Mean and variance, re-derived (from zero)

For a list of numbers `x₁ ... x_d`:

- Mean: `μ = (1/d) Σᵢ xᵢ` — the center.
- Variance: `σ² = (1/d) Σᵢ (xᵢ − μ)²` — average squared distance from center.
- Standard deviation: `σ = sqrt(σ²)`.

These are the only two quantities in this entire day. LayerNorm takes a vector,
subtracts its mean, divides by its std, rescales by `γ`, shifts by `β`.

### 1.2 The LayerNorm formula (hand-written)

For an input `x` of shape `(..., n_embd)` normalized over the **last**
(n_embd) dimension:

```
μ   = x.mean(-1, keepdim=True)
σ²  = x.var(-1, unbiased=False, keepdim=True)      # biased: divide by d, not d-1
x̂   = (x − μ) / sqrt(σ² + ε)
y   = γ * x̂ + β
```

- **Normalizing over the last dim (n_embd), not the batch.** This is the
  defining difference from BatchNorm. Each token's vector is normalized in
  isolation using only its own components — because in autoregressive
  generation a batch is (B, T) and there is no "natural" across-batch
  statistic; per-token/per-position normalization works identically whether
  batch is 1 (generation) or 64 (training). Batch size can change freely; the
  layer stays well-defined.
- **`ε`** (e.g. `1e-5`) prevents divide-by-zero and keeps the gradient through
  `1/sqrt(σ²+ε)` finite when variance is tiny.
- **`γ, β`**: *learned* affine parameters, init `γ=1, β=0`. They let the layer
  be the identity at init and — critically — let the net **recover any scale it
  actually wants**, because forcing zero-mean unit-variance forever would
  destroy representational range. Normalization is a *reparameterization*, not
  a discarding of information.
- **bias = False note:** with LN we can drop biases in the linear layers
  feeding into it (LN absorbs them by mean subtraction) — modern models do
  this. We will note it, not chase it.

### 1.3 Why eps appears in sqrt(&middot;+ε) and why unbiased=False

The `torch.var` default divides by `N−1` (unbiased, sample variance). For LN
the normalization is over the *whole* vector (a population, not a sample), so
we use `unbiased=False` to divide by `N` — and this choice must match what
`nn.LayerNorm` does (it does). The `1e-5` is shrunk later in some modern models
(e.g. `1e-6`), it is a stability knob, not a theory knob.

### 1.4 Pre-norm vs post-norm (derivation-level argument)

- **Post-norm (original paper):** normalize *after* the sublayer:
  `LN(x + sublayer(x))`. The identity path itself passes through normalization
  — the clean gradient highway from Day 8 is now blocked by LN's Jacobian
  (which is a projection-like matrix, not identity-preserving).
- **Pre-norm (GPT-2 onward):** normalize *before* the sublayer:
  `x + sublayer(LN(x))`. The identity path `x` stays **untouched**; only the
  sublayer's input is normalized. Gradient-wise this is strictly friendlier,
  and empirically it trains stably without warmup even at depth (Xiong et al.,
  "On Layer Normalization in the Transformer Architecture").

WRITE THIS DOWN: **pre-norm = normalize, then compute, then add back onto the
bare residual path. Post-norm normalizes the path itself and weakens the
highway.**

### 1.5 The honesty note (this is what "understand deeply" includes)

Ask "why does LayerNorm *actually* work" and parts of the answer are still
debated. Solid, verified parts: it removes mean-shift across the residual
stream, keeps pre-activations in a well-scaled regime regardless of upstream
drift, and (pre-norm) decouples gradient flow from the sublayers. Less settled:
whether the variance-whitening or the mean-removal is the load-bearing half, and
why it helps so much more than a simple scaling would. This is a *living* area —
the honest position is "we will measure it, and we will not pretend the theory
is closed."

### 1.6 What we measure today

Two models trained under identical conditions: the Day-8 block (attention+MLP+
residuals, no LN) vs the same block with **pre-norm LN** around each sublayer.
Metrics: val loss, train/val gap, and — the stability signal — the **spread of
gradient norms across layers** over training steps. Pre-norm should produce
tighter, stabler gradients and more monotonic loss descent.

---

## Part 2 — QUESTIONS (ask, wait, validate, correct)

**Q1.** Over which dimension does LayerNorm normalize in our model, and why
*not* over the batch? (n_embd, the last dim; because it must work with batch=1
at generation and different batch sizes freely.)

**Q2.** If we initialized `γ=1, β=0`, what does that make the layer at init,
and why does the *learnable* `γ/β` rescue representational range? (Identity at
init; lets the net re-seal any scale/shift it wants instead of being pinned to
zero-mean unit-variance.)

**Q3.** Pre-norm vs post-norm: which one keeps the Day-8 identity gradient
highway clean, and why? (Pre-norm: it normalizes only the sublayer's input;
the `+x` path is untouched.)

**Q4.** We compute variance with `unbiased=False`. What does that mean, and why
does it match `nn.LayerNorm`? (Population variance: divide by `d`, not `d−1`;
LN normalizes the whole vector.)

**Answer key** matches the lecture content above (1.2–1.4).

---

## Part 3 — BUILD (step by step)

**STEP 1 — `src/layernorm.py`, the formula first:**
```python
class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta  = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * xhat + self.beta
```
- *What/Why per line:* mean then biased variance (population), stabilize with
  eps, affine transform with learnable γ/β. All shapes `(..., n_embd)`
  broadcastable over last dim.

**STEP 2 — verify against the reference:**
- *What:* `torch.allclose(ours(x), nn.LayerNorm(n_embd)(x), atol=1e-5)` on a
  few random tensors, including *different batch sizes* (e.g. B=2 and B=64 with
  the same single weight vector — proving batch-invariance is built in).
- *Why:* honest check the hand-written layer is not subtly wrong (shape, dim,
  unbiased flag, eps placement). If it doesn't match, debug — don't move on.

**STEP 3 — the "what did it do" inspection:**
- Run inputs with wildly different scales/shifts (multiply by 1, 10, −5; add
  offsets) and show the LN output has mean≈0, std≈1 *before* γ/β, then show γ/β
  recovering scale. Print `out.mean()`, `out.std()`.

**STEP 4 — wire pre-norm LN into the block:**
- Upgrade Day 8's model to: `x = x + sa(LN(x));  x = x + ffn(LN(x))` — the real
  begin-block shape. Keep the Day-8 (no-LN) model in the same script under the
  same training conditions.

**STEP 5 — the stability experiment:**
- Train both. At a few checkpoints, record the *spread* of gradient norms, e.g.
  `[p.grad.norm() for p in model.parameters() if p.grad is not None]` and print
  max/min (or std of the list). Pre-norm should show less spread + a more
  monotonic val-loss descent.

**STEP 6 — train, compare, sample:** LOG entry numbers + sample.

---

## Part 4 — VERIFY

- allclose vs `nn.LayerNorm` (multiple batch sizes).
- mean≈0 / std≈1 invariant to input scale/shift.
- val loss (pre-norm block) vs Day-8 (no LN).
- gradient-norm spread: no-LN vs LN.
- sample plausibility.

---

## Part 5 — QUIZ (answers for Claude)

- **Q1.** Write the LayerNorm formula. Expect four parts: mean, biased
  variance, `(x−μ)/sqrt(σ²+ε)`, `γ·x̂+β`.
- **Q2.** Why per-vector (last-dim) not per-batch? (Generation with B=1;
  batch-varying well-defined behavior; consistent semantics train/inference.)
- **Q3.** Why is the bias in a linear layer "optional" before LN? (LN's mean
  subtraction absorbs constant shifts.)
- **Q4.** Pre-norm's placement vs post-norm: one sentence each on the gradient
  consequence.
- **Q5.** Name one honest open question about why LN works. (Any of: variance
  vs mean removal, or why it helps more than plain scaling — accept a
  well-phrased uncertainty.)

---

## Part 6 — LOG entry, notes, look-ahead

- **LOG:** "Day 9 — LayerNorm (hand-written, pre-norm)": allclose proof, the
  invariance demonstration, val loss vs Day 8, gradient-norm spread, sample.
- **Notes:** `docs/notes/day09-layernorm.md` (user my-words summary + Claude
  completion + approval).
- **Look-ahead:** Day 10 — assemble the complete `Block` (`LN → SA → +`, `LN →
  FFN → +`), stack N of them, add dropout, and train the first *real* GPT. The
  piece list is now complete: embeddings, multi-head, MLP, residuals, LN.
- **Stretch:** read the original LayerNorm paper (Ba, Kiros, Hinton 2016) for
  the story of why it beats BatchNorm in recurrent settings — directly relevant
  to why transformers use layer (not batch) normalization.
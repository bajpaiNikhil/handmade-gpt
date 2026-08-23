# Day 8 — Residual connections ("why deep stacks don't die")

## Session card

- **Goal:** understand the identity shortcut deeply — mathematically and by
  hand-derivation — then add residuals around attention/MLP, and *watch* the
  gradient behave differently than without them.
- **Files:** new `src/residuals.py` (tiny hand-derived net + wiring into the
  LM), reusing trained-state of Day 7.
- **Numbers to relate:** Day 7's loss after adding the MLP; Day 8 adds the
  residual paths and, shortly after, stacking becomes possible.
- **Prerequisites:** Days 1–7. The central tool is the chain rule — we will
  derive rather than trust.

---

## Part 1 — LECTURE (teach first)

### 1.0 Where we are

From Day 10 on we want to **stack** blocks — because one "communicate +
compute" round is a shallow pass, and language needs many rounds (Day 10 will
show why stacking is where induction heads etc. emerge). But a naive stack had
a classic problem that stalled deep networks for 20 years.

### 1.1 The vanishing-gradient problem (derived, not folklore)

Consider a stack of `L` layers, each a function `f`. The final loss depends on
the first layer through the chain rule:

```
dL/dw₁ = dL/dh_L · dh_L/dh_{L-1} · ... · dh₁/dw₁
```

Each factor `dh_{i}/dh_{i-1}` is a Jacobian matrix. If the typical eigenvalue
magnitude of these Jacobians is `< 1`, the product → 0 exponentially with depth
(vanishing); if `> 1`, it explodes. Either way, deep stacks can't be trained
without help. ReLU + careful init (and warmup, Day 11) recovers *some* of it —
transformers add the second half of the fix structurally: **skip the layers in
the gradient path.**

### 1.2 The residual (identity) shortcut — derived

Define `h_{i} = h_{i-1} + sublayer(h_{i-1})`. Then:

```
dh_{i}/dh_{i-1} = I + d(sublayer)/dh_{i-1}
```

`I` is the identity — *always* there, regardless of what the sublayer does.
Backprop through `L` residual layers therefore carries the gradient via a
products-of-identities path `≈ I` even when the sublayer Jacobians are small.
The gradient has a **highway**: it never has to pass through a sequence of
shrinking matrices to reach the early layers.

WRITE THIS DOWN: **`x + f(x)` means the gradient can flow at full strength
through the identity term `I`, even if `f`'s Jacobian vanishes.**

### 1.3 Do the tiny derivation by hand (the build's STEP 1)

Use a 3-layer toy net `y = f₃(f₂(f₁(x)))` vs `y = ((x + f₁(x)) + f₂(...)) + f₃(...)`.
Differentiate both by hand/on paper and write down dL/dx for each. The plain
stack is a chain of up-to-3 small Jacobians; the residual stack has a term that
is the identity and needs no product. That contrast is the whole day.

### 1.4 The residual stream framing (why residuals are not just "something that works")

Reading the circuit-engineering liter datum (Elhage et al., "A Mathematical
Framework for Transformer Circuits"): think of the `n_embd` vector at each
position as a **bus / "residual stream"** shared by all layers. Each sublayer
*reads* the stream, writes a small delta, and *adds* it back in. Consequences:
- Any layer's output is a sum of contributions from all earlier layers — the
  stream is additive, so later layers can "inspect" the additions on top.
- The LM head (and LayerNorm, Day 9) reads this accumulated stream once, at the
  end — it does not have to read each layer separately.
- Because each sublayer writes only a *small* delta onto a steadily-sized
  vector, the model can compose many stages without the representation
  exploding in amplitude.

This is also *why* pre-norm placement (Day 9) makes sense: we want the idle
identity path to stay untouched by normalization.

### 1.5 History, so the intuition is anchored

Pre-ResNet (2015) nets were stuck around ~30 layers before training stalled;
the ResNet identity shortcut pushed ImageNet nets to 152+ layers overnight. The
transformer inherited the same trick. If residuals feel like "a small change",
that's misleading — historically they were *the* change that made depth train
at all.

### 1.6 What we add today (and what we deliberately don't)

- Residual around the attention block: `x = x + MultiHeadAttention(x)`.
- Residual around the MLP: `x = x + FeedForward(x)`.
- NOT yet: LayerNorm, stacking, dropout. Days 9–10 arrive separately so each
  mechanism gets a clean measurement.

---

## Part 2 — QUESTIONS (ask, wait, validate, correct)

**Q1.** Write `h_i = h_{i-1} + f(h_{i-1})`. What does `dh_i/dh_{i-1}` equal, and
which term never vanishes no matter what `f` does?

**Q2.** Why does a product of small Jacobians (plain stack) destroy gradients,
and how does the identity term break that product?

**Q3.** In the "residual stream" framing, what is the stream and what does each
sublayer do to it? Why does an additive stream let deep models compose?

**Q4.** Design question: if we put LayerNorm *inside* the residual path (i.e.
normalize the stream before reading it) vs on the stream itself, which one
keeps the identity gradient path clean? (Preview of Day 9 — discussing is fine,
just plant the question.)

**Answer key:**
- **A1.** `I + df/dh_{i-1}`. The `I` term is always present regardless of `f`.
- **A2.** The chain product of layer Jacobians decays exponentially when their
  typical singular values < 1. The identity term contributes a *summand* that
  survives any number of products, so the gradient no longer requires all
  factors to cooperate.
- **A3.** The stream is the accumulated `n_embd` vector at each position; each
  sublayer reads it, computes a delta, adds it. Additivity means outputs are
  composed by summation, so later layers see everyone's additions and no single
  layer must dominate the representation.
- **A4.** Place the normalization *before* the sublayer and keep the identity
  path bare (`x + LN(attn(x))` style) — that is pre-norm. Keep the two-layer
  discussion honest: this is exactly Day 9's argument.

---

## Part 3 — BUILD (step by step)

**STEP 1 — the hand-derivation** (paper, not code): the 3-layer stack vs
residual stack; write `dL/dx` for both. This is the whole concept; do it slow.

**STEP 2 — `src/residuals.py`, toy proof:**
```python
def no_residual(x):
    h = f1(x); h = f2(h); h = f3(h); return h
def with_residual(x):
    h = x + f1(x); h = h + f2(h); h = h + f3(h); return h
```
- *What:* two tiny nets, same `f`s, different wiring.
- *Why:* show identical params, identical expressivity in theory, different
  gradient behavior — isolate the residual effect.

**STEP 3 — gradient-norm experiment (the day's money print):**
- *What:* for both nets, `loss.backward()` and print `grad.norm()` per weight
  tensor (e.g. by iterating `model.parameters()`, skipping `None` grads).
- *Why:* the earlier layers' gradient norms are where vanishing shows up. With
  residuals, the first layer's gradient norm should be dramatically healthier
  than without.
- *How to read:* seeds + small nets; the exact ratio depends on the scale, but
  the *pattern* (early-layer norms much larger with residuals) is the finding.
  Guard: re-run the same comparison several times / with the same seed, so you
  report a pattern and not one lucky run.

**STEP 4 — wiring into the LM:** add `x = x + self.sa_heads(x)` and
`x = x + self.ffn(x)` into the Day-7 model (new class, keep the non-residual
version alongside, both trained under identical conditions as always). You now
have the first **nearly-complete block** — the only missing piece after this
day is LayerNorm (Day 9) and stacking (Day 10).
- *Why both versions:* the before/after is the measurement; keep it in one
  script like the 3-way Day-6 comparison.
- *Expected:* residual version trains at least as well as the non-residual one
  and *definitely* does not degrade early layers' gradients (verified in STEP
  3's style);

**STEP 5 — train, compare, sample:** val loss vs Day 7, train/val gap, sample
plausibility. Record in LOG.

---

## Part 4 — VERIFY

- Hand-derivation written out (user's own words on paper).
- Gradient-norm table: no-residual vs residual, per layer index.
- Val loss with residuals vs without, ± gap.
- Sample: any visible improvement over Day 7?

---

## Part 5 — QUIZ (answers for Claude)

- **Q1.** Give the one-line reason residuals let gradients travel far.
  (Identity term `I` in the Jacobian survives every product in the chain.)
- **Q2.** What did residuals enable historically that non-residual deep nets
  couldn't do? (Depth: ~30 → 150+ layers; transformer inherited it.)
- **Q3.** In the residual stream view, what is added to the stream and who
  reads the stream at the end? (Each sublayer adds a delta; LN + head read the
  stream once at the end — Day 9-10.)
- **Q4.** Why does the pre-norm placement (normalize before the sublayer) keep
  the gradient highway clean? (The identity path `x` is not normalized.)
- **Q5.** Name the single number that empirically shows residuals protecting
  early layers. (First/lower-layer gradient norm.)

---

## Part 6 — LOG entry, notes, look-ahead

- **LOG:** "Day 8 — residual connections": the hand-derivation, gradient-norm
  table, val loss before/after, sample, the residual-stream mental model.
- **Notes:** `docs/notes/day08-residuals.md`, user my-words summary, Claude
  completes, user approves.
- **Look-ahead:** Day 9 — LayerNorm. Residuals fixed the gradient path; the
  next problem is that activation magnitudes across a deep stack drift and
  destabilize training. Normalization is the stabilization half of pre-norm
  blocks.
- **Stretch:** read the ResNet abstract/comments for the historical framing, or
  ELHAGE-section on residual stream for circuit framing. Revisit our gradient
  norms at depth after Day 10 stacks and notice the same pattern at real scale.
# Notes — Day 8: Residual connections

> Claude fills at session end; user reviews and approves.

## My-words summary (user)

> Attention gathers info, residual adds it back without losing the original.

## Corrected / completed (Claude)

That's the residual half exactly right — it's the "express lane" idea:
`x = x + sublayer(x)` carries an unmodified copy of the input across in
parallel with whatever the sublayer (attention or FeedForward) computes, so
nothing already there gets overwritten, only added to. The reason this
matters goes one level deeper than just "don't lose the original": it's
about the *gradient* (the backward-flowing correction signal used during
training), not just the forward-pass value. For `h_i = h_{i-1} + f(h_{i-1})`,
the sensitivity of `h_i` to `h_{i-1}` splits into two added pieces: the
identity (always exactly 1, no matter what `f` does) plus `f`'s own
sensitivity (which can shrink toward zero as stacks get deep). Because a
plain stack (no express lane) only ever *multiplies* these per-layer
sensitivities together, one small factor is enough to crush the whole
product — while the residual stack's guaranteed "+1" per layer means the
product can never be fully crushed. Measured directly today: a plain
3-layer toy chain's earliest-layer gradient was destroyed by depth 10
(underflowed to exact 0.0 in float precision — the layer would never learn,
ever), while the residual version's gradient stayed healthy and even grew
with depth. Residuals add **zero new parameters** — confirmed exactly
(16,961 params, identical with or without residuals) — the entire
improvement comes from wiring, not capacity.

## WRITE THIS DOWN captures

- `x + f(x)` lets the gradient flow at full strength through the identity term
  `I` even if f's Jacobian vanishes.
- Residual stream = additive bus; sublayers read, compute a delta, add it back.

## Misconceptions corrected today

- **Q2 (why residuals fix vanishing gradients):** initial answer described
  the *forward*-pass benefit correctly ("the layer above is aware of the
  changes") but didn't distinguish that from the *backward*-pass mechanism
  the lesson is actually about. Corrected: the express lane turns each
  layer's backward step from pure multiplication into addition — one part
  keeps shrinking under multiplication same as before, but a second,
  unmultiplied part (the identity) is simply added in at full strength every
  time, so the total can't collapse to zero even if the multiplied part does.
- **Q3 (residual stream):** correctly identified the whiteboard/residual
  stream as a shared, never-erased vector, but conflated it with the
  "correction value" (gradient) — the stream itself carries the token's
  *forward* representation; the gradient is a separate, backward-flowing
  quantity through the same structure. Also hadn't yet connected "nothing
  gets erased" to *why* that matters for depth: if a layer overwrote instead
  of added, every earlier layer's contribution would be destroyed by the
  very next layer's turn.

## My quiz answers

_(pending — quiz not yet run this session)_

## Numbers logged (LOG.md)

- **Hand-derivation:** done — worked by hand with a=0.3 per layer, 3 layers.
  Plain stack: 0.3×0.3×0.3 = 0.027. Residual stack: (1+0.3)³ = 2.197.
- **`residuals.py` (real nn.Linear + tanh, weights scaled to ~0.3 sensitivity,
  3 seeds):** grad(f1) plain ≈ 0.0002–0.0003, grad(f1) residual ≈ 1.10–1.39
  — a ~4,500–5,700x gap, consistent across all 3 seeds, direction matching
  the hand math (harsher in practice because tanh's own sensitivity
  compounds with the 0.3 weight scale).
- **Side experiment (depth 3 vs 10 vs 20, same 0.3-scaled layers):**
  ```
  depth  grad(f1) plain    grad(f1) residual
    3    0.0003083         1.391601
   10    0.0000000000      1.792191   (plain underflowed to exact 0.0)
   20    0.0000000000      2.731302
  ```
  Plain stack's earliest layer is completely dead (zero gradient, never
  learns) by depth 10; residual's gradient stays healthy and *grows* with
  depth, matching `(1+0.3)^L` growing unbounded as L increases.
- **Params:** multi-head+FFN = 16,961, multi-head+FFN+residual = 16,961 —
  identical, confirming residuals add zero new parameters.
- **Val loss, two independent unseeded runs:**
  ```
                              run 1      run 2
  multi-head+FFN val loss    2.3089     2.2263
  multi-head+FFN+resid val   2.2244     2.2080
  residual vs no-residual   +0.0845    +0.0183
  residual train/val gap    +0.0373    +0.0182
  ```
  Direction (residual beats no-residual) held in both runs; exact margin
  swung with the usual `estimate_loss` sampling noise (±0.01–0.03, per Day 6's
  measured noise floor) — one run even showed a slightly negative no-residual
  gap (-0.0319), which is noise landing negative, not a real finding.
- Generated text: still gibberish for all five models — expected, no
  LayerNorm or stacking yet.

## Follow-up: pushing `residuals.py` from 3 to 10 layers

`residuals.py` was updated (default is now `NUM_LAYERS = 10`, not 3) to add a
stage-by-stage output-norm trace, answering "what happens to the actual
output, not just the gradient, as depth increases":

```
layer   output norm (plain)   output norm (residual)
  1           0.723647              4.150535
  2           0.106643              4.242605
  3           0.018456              4.364934
  ...
  9           0.000000              4.559890
 10           0.000000              4.736363
final: plain mean=-0.000000 std=0.000000 | resid mean=-0.226156 std=1.175013
```

The plain chain's output collapses to exact zero by layer 9-10 — not just a
vanishing gradient, the actual *representation* is destroyed; the network
stops depending on its input at all. The residual chain's output never
collapses — but its norm keeps *growing* every layer (4.15 → 4.74). That
growth is what saves the gradient (confirmed: `grad(f1)` for the plain
10-layer chain is now exactly `0.0` across all 3 seeds, vs `1.55`-`1.79` with
residuals) — but unbounded growth is itself a new failure mode at real depth,
and it's the direct, measured motivation for Day 9: residuals fix vanishing
gradients by letting activations grow without bound; LayerNorm/RMSNorm is
the fix for *that* growth.

## One line for tomorrow

Residuals fixed the gradient path — depth can now train. But nothing yet
stops each layer's added contribution from drifting the residual stream's
overall scale up or down as more layers write to it; Day 9's LayerNorm is
the stabilization half of the pre-norm block.

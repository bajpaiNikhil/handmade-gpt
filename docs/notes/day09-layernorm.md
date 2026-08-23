# Notes — Day 9: LayerNorm (hand-written, pre-norm)

> Claude fills at session end; user reviews and approves.

## My-words summary (user)

> Write the LayerNorm formula, explain the last-dim normalization, and say why
> pre-norm placement keeps the residual highway clean.

## Corrected / completed (Claude)

## WRITE THIS DOWN captures

- Pre-norm = normalize, compute, add back onto the bare residual path.
- Normalizing over n_embd (not batch) works at generation time (batch=1).
- γ/β = identity init, rescues representational range.

## Misconceptions corrected today

## My quiz answers

## Numbers logged (LOG.md)

- allclose vs nn.LayerNorm (which batch sizes?).
- mean≈0 / std≈1 invariants observed.
- Val loss pre-norm block vs Day-8 no-LN; gradient-norm spread.

## One line for tomorrow
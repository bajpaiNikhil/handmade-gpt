# Notes — Day 8: Residual connections

> Claude fills at session end; user reviews and approves.

## My-words summary (user)

> Why does `x + f(x)` fix deep stacks? Where does the identity term live in the
> chain rule, and what is the residual stream?

## Corrected / completed (Claude)

## WRITE THIS DOWN captures

- `x + f(x)` lets the gradient flow at full strength through the identity term
  `I` even if f's Jacobian vanishes.
- Residual stream = additive bus; sublayers read, compute a delta, add it back.

## Misconceptions corrected today

## My quiz answers

## Numbers logged (LOG.md)

- Hand-derivation written (yes/no).
- Gradient-norm table: no-residual vs residual per layer.
- Val loss before vs after residuals; train/val gap.

## One line for tomorrow
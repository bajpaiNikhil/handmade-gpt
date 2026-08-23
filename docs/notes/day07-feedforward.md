# Notes — Day 7: The FeedForward (MLP)

> Claude fills at session end; user reviews and approves.

## My-words summary (user)

> What is the FeedForward layer, why does attention alone not suffice, and how
> does the "communicate vs compute" split work?

## Corrected / completed (Claude)

## WRITE THIS DOWN captures

- Attention exchanges info; it does not compute a nonlinear function per token.
- FFN ≈ per-token nonlinear processing; in the "memories" view, where facts
  live; holds ~2/3 of a transformer's params.

## Misconceptions corrected today

## My quiz answers

## Numbers logged (LOG.md)

- FFN params (n_embd=32, 4×): 8,352
- Val loss before (2.3217) vs after MLP; train/val gap.

## One line for tomorrow
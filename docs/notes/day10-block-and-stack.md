# Notes — Day 10: Assemble the Block + stack

> Claude fills at session end; user reviews and approves.

## My-words summary (user)

> Draw the full Block dataflow in one line, say where dropout lands and why,
> and what stacking enables that a single block cannot.

## Corrected / completed (Claude)

## WRITE THIS DOWN captures

- One block = one communicate+compute round; stacking is how later blocks build
  on earlier rounds (e.g. induction heads at two layers).
- Final LN stabilizes the accumulated stream for the plain readout.

## Misconceptions corrected today

## My quiz answers

## Numbers logged (LOG.md)

- Config used (n_embd / n_head / n_layer / block_size / dropout / batch).
- Total params; val loss vs Day 9; sample (300 chars, structure read).

## One line for tomorrow
# Notes — Day 11: Scale & ship (W2 BOSS)

> Claude fills at session end; user reviews and approves.

## My-words summary (user)

> What does perplexity mean, which scaling knobs moved the loss, and did the
> final model "scan"?

## Corrected / completed (Claude)

## WRITE THIS DOWN captures

- Perplexity = exp(val_loss); ~1.5 nats ≈ 4.5 plausible next chars.
- Warmup prevents early instability; cosine decay lowers late noise; AdamW
  weight decay is decoupled from the adaptive step.

## Misconceptions corrected today

## My quiz answers

## Numbers logged (LOG.md)

- Sweep table (config → val loss → steps → wall time).
- Final val loss + perplexity vs 4.17 start and 1.48 floor.
- Final config + params + checkpoint path (`checkpoints/gpt_w2_final.pt`).
- Two best samples.

## BOSS verdict + Article 1 seeds

- Did it scan? Why/why not, honestly.
- 3 bullet hooks for Sunday's article (from the brief in day11 lesson).

## One line for the map
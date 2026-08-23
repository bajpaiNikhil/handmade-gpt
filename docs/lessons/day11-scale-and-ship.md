# Day 11 — Scale & ship (BOSS) + Article 1 brief

## Session card

- **Goal:** complete the campaign's W2 boss: sweep the scale knobs on the
  Day-10 GPT, tune LR warmup/decay, push toward the Shakespeare loss floor
  (~1.48 nats) as far as an M4 Pro reaches, generate text that *scans*.
- **Files:** the Day-10 model/config, swept; new training config + final
  checkpoints log; LOG entry; **Article 1 brief** for Sunday.
- **Baseline:** Day 10 val loss; tracker floor shown at 1.48.
- **BOSS criteria (campaign map):** "stack blocks, scale, train on MPS —
  output: Shakespeare-flavored text that actually scans." Meeting it = W2 done.

---

## Part 1 — LECTURE (teach first)

### 1.0 Where we are — architecture done, scale now

The architecture is now the classic decoder-only transformer, every layer
hand-typed. The gap between today's loss and a *readable* Shakespeare text is
mostly *capacity and compute*, not plumbing. Day 11 is about squeezing an M4
Pro for everything it has.

### 1.1 The loss floor and what the numbers mean

- Random baseline (uniform over 65 chars): `−ln(1/65) = 4.17 nats` — where we
  started.
- A well-trained small Shakespeare char-level model lands near **~1.45–1.50**.
  That is the entropy of the corpus under a good model — the practical floor
  for char-level Shakespeare. (The tracker uses 1.48.)
- **Perplexity = exp(val_loss)** — "average number of plausible next chars".
  exp(4.17)≈65 chars to choose among; exp(1.5)≈4.5; exp(1.4)≈4. That cluster
  around 4–5 is why Shakespeare-char models output readable-but-not-deterministic
  text: every char has several genuinely plausible continuations.

WRITE THIS DOWN: **perplexity = exp(val_loss). At ~1.5 nats the model weights
≈4.5 plausible next characters — the natural noise floor of the task, not a
defect.**

### 1.2 The scaling-knob map (what moves which number)

| Knob | Levers | What it trades |
|------|--------|----------------|
| Capacity | `n_embd`, `n_layer`, `n_head` | params → capacity to memorize/generalize; train/val gap cost |
| Context | `block_size` | how far back patterns reach; compute per step grows ~T·(a little more than linear) |
| Compute | steps, `batch_size` | time on MPS; smoothes gradient noise too |
| Optimizer | lr, warmup, weight decay | how well capacity *learns* at all at depth |
| Regularization | dropout | overfit shield (Day 10); too high slows learning |

Scaling-law intuition (Chinchilla-shaped): loss ≈ improves with params *and*
data/tokens *and* compute, all three roughly power-law. We are tiny; the honest
goal is "bend the curve", not "match GPT-2".

### 1.3 LR warmup + cosine decay + weight decay (AdamW) — why at depth

- **Warmup:** early steps have wild statistics; a full LR immediately can blow
  up the loss valley (esp. with pre-norm + deep stacks). Warm up LR linearly
  over a few hundred steps, then cosine-decay to near 0 — GPT-3-style schedule
  sanity. At our small scale: `lr=3e-4`-ish, ~200–500 warmup steps on long
  runs.
- **Weight decay (AdamW, not Adam+L2):** AdamW decays weights inside the update
  independent of the adaptive per-parameter rate (L2-in-Adam couples them).
  `weight_decay≈0.01–0.1` gently pulls weights to 0 — implicit regularization,
  standard for transformers.
- **Why now and not Day 2?** The bigram barely needed it (one table, no depth).
  Deep stacks have the failure modes warmup/decay are *for*: variance explosion
  early, silent overfit late.

### 1.4 The honest gap to GPT-2

GPT-2 "124M" is the classic reference; our hands-built model will sit at
roughly **0.5–5M params** — 25–250× smaller — on ~1 MB of text vs GPT-2's
very-large web corpus, with `block_size` 64–256 vs 1024. So: we are not
competing with GPT-2. We are verifying that OUR hand-typed layers, at whatever
scale an M4 Pro affords in ~1 hour, move the loss toward the floor and make
text that scans. That is the boss.

### 1.5 The "scans" bar, defined honestly

"Scans" = text where words are mostly real, word order is sentence-ish, line
starts look like stage directions / speaker colon patterns, and punctuation
lands plausibly. Not grammatical prose — word-shaped, structure-shaped text.
That is exactly what a ~2M-param char model on Shakespeare produces around
1.6–1.7 nats.

---

## Part 2 — QUESTIONS (ask, wait, validate, correct)

**Q1.** Perplexity of a val loss of 1.5 is what? What does that number mean in
plain words? (exp(1.5)≈4.5 — model weighs ~4.5 plausible next chars.)
**Q2.** We have ~1 MB of text and a few M params. Of the scaling knobs, which
two are the *first* suspects for "loss stalls high"? (capacity/context:
n_embd+n_layer, block_size — data/compute after.)
**Q3.** Why warmup at the start rather than a flat full LR, in one sentence?
(First-step statistics are wild; full LR early destabilizes.)
**Q4.** AdamW vs Adam+L2 on the weights — what is the actual difference?
(Decay applied independently of the adaptive step; proper decoupling.)
**Q5.** What is our boss criterion, exactly, and how will we *measure* "scans"?
(Shakespeare-flavored sample that scans; judge jointly, plus val loss for the
number.)

**Answer key** maps to 1.1–1.5.

---

## Part 3 — BUILD (step by step)

**STEP 1 — baseline sweep (cheap first):** keep the Day-10 model; run short
runs (2–5k steps) sweeping in order: n_embd {64,96,128,192}, n_layer {2,4},
block_size {64,128,256}, batch. Log loss per config compactly (a tiny table;
remember MPS wall-time per step for planning).
- *Why this order:* capacity changes loss the most at our scale; context is
  next; batch/steps are "compute", last.

**STEP 2 — add the optimized trainer:** a `get_lr(step)` implementing
warmup→cosine decay (hand-written — a 10-line function, no scheduler import
needed; it's ours), `optimizer = torch.optim.AdamW(params, lr, weight_decay=0.1)`,
and manually `param_group['lr'] = get_lr(step)` per step. Explain each line
(this replaces the flat `lr=1e-3` of every prior day).

**STEP 3 — final training run:** pick the best config from STEP 1; train to
diminishing returns within ~30–60 min wall time (10–20k steps at the chosen
size). Print eval every 1k steps; watch for the train/val gap (the overfit
dial: if val > train by a lot, this is the dropout/weight-decay zone).

**STEP 4 — checkpoint:** `torch.save(model.state_dict(), 'checkpoints/gpt_w2_final.pt')`,
log final val loss, params, config, wall time.

**STEP 5 — generate several seeded samples:** same seed, different
`max_new_tokens` (e.g. 300 and 1000); judge "scans" jointly; pick 2 winners.

**STEP 6 — wrap W2:** final LOG entry + mark BOSS done in the tracker
(`docs/gpt-build-tracker.html` — browser toggle for Day 11).

---

## Part 4 — VERIFY

- Sweep table recorded (config → val loss → steps → wall-time).
- Final val loss + perplexity vs the 4.17 start and ~1.48 floor: where do we
  land?
- Train/val gap + the number that decided the final config.
- The two best samples reproduced in the LOG entry.

---

## Part 5 — QUIZ (answers for Claude)

- **Q1.** Give the name and formula linking val loss to "number of plausible
  next characters". (Perplexity; exp(val_loss).)
- **Q2.** Order the knobs by expected loss impact at our scale, and say why.
  (Capacity/context first; compute/optimizer after; regularization last.)
- **Q3.** Warmup+cosine schedule: what is each doing? (Warmup avoids early
  instability; decay lowers the noise of late optimization.)
- **Q4.** What two levers exist for an overfitting train/val gap, and which
  affects learning speed? (dropout/weight-decay; dropout slightly slows too.)
- **Q5.** Where did we land vs the two landmarks (4.17 start, 1.48 floor), and
  is that "scans"? (Report honestly; boss-criterion judgement with reasoning.)

---

## Part 6 — LOG entry, notes, ARTICLE 1 brief

- **LOG:** "Day 11 — Scale & train (W2 BOSS complete)" — sweep table, final
  config+loss+perplexity, samples, wall-time, one proud-honest summary line.
- **Notes:** `docs/notes/day11-scale-and-ship.md`.

### Article 1 brief (for Sunday's writing day)

Save-point piece — the big one that "gets cited".

- **Title seed:** "I built a GPT from scratch on a MacBook — every layer by
  hand" (campaign map).
- **Story spine:** start at `−ln(1/65)=4.17` (random) → walk the *why* of each
  layer exactly as built: lookup table → causal averaging → one head → multi
  head → MLP → residual → LayerNorm → stacked block → the loss falling toward
  1.5. Every number real, from LOG.md.
- **Recipe (per campaign map):** lookup table → attention → working transformer,
  with the loss dropping below 4.17, and the val-loss table bigram vs
  single-head vs multi-head vs block vs stacked.
- **Differentiator:** your hand-typed rule (no `nn.MultiheadAttention`, no
  `nn.Transformer`) is the hook; every layer is ~5 lines you can paste.
- **Ending:** the scans sample + perplexity ~4.5 framing; tease World 3
  (RMSNorm→RoPE→SwiGLU) and on-device (your Android angle).
- **Assets to gather:** the loss table, 2 best samples, param counts, one
  "architecture sketch" image. Use the `medium-article-writer` skill for the
  outline and the `article-rater` skill before publishing.
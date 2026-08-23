# Notes — Day 6: Multi-head attention (redo)

Session date: Mon Aug 10. Style: teach the existing file, no retyping.
Files walked: `src/multi_head_attention.py`. Files run:
`src/multi_head_attention.py`, `src/attention_lm.py`.

## My-words summary (user, from memory)

> **PENDING** — not captured this session. Quiz was deferred to the Day 7
> session; this section gets filled from the user's answers there.

## Corrected / completed (Claude)

The day in one paragraph: a single attention head gives each token exactly one
(query, key, value) triple — one learned notion of "similar", one question it
may ask about its past. Language needs many such questions at the same position
(previous-token, word-start, structural position, co-reference, induction), and
they cannot share one set of Q/K/V matrices without fighting. Multi-head runs H
heads in parallel over the same input, each `head_size = n_embd / H` wide, then
concatenates and projects. Because `H · (n_embd/H) = n_embd`, the total Q/K/V
parameter budget is *identical* to one full-width head — the extra questions are
free. The only added cost is `proj`, and `proj` is not plumbing: `torch.cat` is
pure memory layout that does no math, so before `proj` nothing has ever combined
head 0's finding with head 3's. `proj` is the sole place cross-head mixing can
happen. It also keeps output shape == input shape, which is what makes Day 8's
`x = x + attn(x)` legal.

## WRITE THIS DOWN captures

- Attention is a content-aware weighted average of the past. The weights are
  computed from the content (via Q and K) — they are not fixed.
- One head = one learned notion of "similar". Language needs many notions of
  "similar" at the same position, simultaneously. Multi-head supplies them in
  parallel.
- Multi-head = H independent questions in parallel at the same parameter budget
  as one wide head; concat reassembles them to full width, proj lets the model
  mix across what the different heads found.
- Softmax acts on score *gaps*, exponentially. `1/√head_size` holds score
  variance ≈ 1 so gaps stay moderate, so no position's gradient is crushed to
  zero — the head stays trainable.

## Derivations done from the ground up

**Parameter parity (worked by hand, then verified by PyTorch).**
One `Head` holds three `nn.Linear(n_embd, head_size, bias=False)`, so its
parameter count is `3 · n_embd · head_size`.

```
1 head,  head_size 32 : 3 · (32 · 32) = 3,072
4 heads, head_size  8 : 4 · 3 · (32 · 8) = 3,072
```

Equal because the 4 and the 8 multiply back to 32. Not a numeric coincidence —
forced by the design rule `head_size = n_embd / H`, i.e. `H · head_size = n_embd`
always. **The head count is a free knob: you are not buying more parameters, you
are choosing how many independent questions to spend the same parameters on.**

**Why `/√head_size`, measured not memorized.** `q · k` sums `head_size`
independent products, and variances add, so `Var(q·k) ≈ head_size`. Dividing by
`√head_size` divides the variance by `head_size` (since `Var(c·X) = c²·Var(X)`),
pinning it near 1 at any head width. Live demo at `head_size=64`:

```
raw scores    : [ 4.65,  2.27, -2.24, -7.32]   variance 27.85
scaled scores : [ 0.58,  0.28, -0.28, -0.91]   variance  0.44     (= 27.85/64)

softmax(raw)    : [0.9143, 0.0847, 0.0009, 0.0000]
softmax(scaled) : [0.4184, 0.3108, 0.1770, 0.0938]

d(prob[0])/d(scores) [raw   ]: [0.078, -0.077, -0.00086, -0.000005]
d(prob[0])/d(scores) [scaled]: [0.243, -0.130, -0.0741,  -0.0392  ]
```

The constant does *not* change which score wins — ordering is identical in both
rows. It changes how violently softmax converts a lead into a monopoly.
Unscaled, position 0 takes 91% and position 3 is erased to 0.0000; its gradient
is −5e-06 vs −0.039 scaled, a factor of ~8,000. A gradient that small means
backprop can never teach the head that position 3 was worth attending to. The
head freezes into its initialization and degenerates into "copy the single best
match". **Saturation is where gradients go to die.**

## File details worth remembering

- `Head` is **unchanged** from Day 5. Multi-head introduces no new attention
  math — `MultiHeadAttention` is a *container* over unmodified heads. The
  innovation is organizational, not mechanical.
- `self.tril[:T, :T]` slices the mask to the incoming `T`. Without the slice,
  generation (which starts at T=1) would crash against a fixed 8×8 mask.
- `k.shape[-1] ** -0.5` reads head width off the tensor at runtime rather than
  hardcoding `0.125`. Load-bearing in this very file: the same `Head` class is
  used at `head_size=32` (baseline, line 144) and `head_size=8` (inside
  `MultiHeadAttention`, line 127). A hardcoded constant would silently
  mis-scale one of them.
- `nn.ModuleList`, not a plain list — this is a correctness trap, not style.
  PyTorch finds parameters by walking registered submodules; a plain
  `[Head(), Head()]` registers nothing, so `.parameters()` misses the heads
  (optimizer never updates them) and `.to(device)` misses them (CPU/MPS
  mismatch at the first forward). The heads would stay randomly initialized
  forever.
- `torch.cat(..., dim=-1)` — heads are independent along *channels*. Every head
  saw all positions of all sequences; what differs is what it extracted.
- STEP 3 verifies heads diverge *before* STEP 4 builds anything on that
  assumption. Habit worth stealing: test the premise, then build on it.
- Why heads diverge at all: identical weights would receive identical gradients
  every step (same input, same output, same loss contribution) and stay
  identical **forever**. Symmetry never breaks on its own — random init is the
  only thing that ever makes heads different, and the optimizer then widens the
  gap.

## Numbers verified today

**Toy file (`multi_head_attention.py`, seeded 1337) — exact match to LOG.md:**

```
head_a wei[0,3] : [0.415, 0.241, 0.137, 0.208]
head_b wei[0,3] : [0.366, 0.227, 0.296, 0.11 ]
rows identical? : False                        <- the premise of the whole day
output shape    : (4, 8, 32) == input shape    <- residual-ready
single head (head_size=32)   : 3,072
4 heads (head_size=8) + proj : 4,128 = 3,072 in heads + 1,056 in proj
heads-only == single-head?   : True
```

Both rows sum to ~1.0 and have exactly 4 non-zero entries (positions 4–7
masked) — softmax and causality both holding. Head A puts its second-largest
weight on position 3 (itself, 0.208); head B on position 2 (0.296) while nearly
ignoring position 3 (0.11). Same input, same architecture, different questions —
and nothing is trained yet, this is pure random init.

`1,056 = 32·32 + 32` — proj's weight matrix plus bias. Q/K/V use `bias=False`,
which is why 3,072 comes out clean.

**Real training (`attention_lm.py`) — three independent unseeded runs:**

```
                   LOG.md     claude run   my run     spread
bigram             2.5390     2.5440       2.5439     0.005
single-head        2.4496     2.4242       2.4415     0.025
multi-head         2.3217     2.3327       2.3179     0.015
multi vs single   +0.1278    +0.0914      +0.1235     mean ~0.114
params: 4,225 / 7,553 / 8,609 — identical in every run (deterministic)
```

Ordering **bigram > single-head > multi-head held in all three runs**. Exact
values drift because `attention_lm.py` sets no seed (only the toy files do), so
init and every `get_batch` draw differ. Defensible claim: *multi-head beats
single-head by roughly 0.09–0.13 nats (~0.11 mean) for 14% more parameters* —
not "by 0.128". A single unseeded run at `batch_size=4` cannot resolve finer
than that.

**The noise floor, measured accidentally.** Same model, same weights, two calls
to `estimate_loss`:

```
[BIGRAM] step 9999 val loss: 2.5318   vs   RESULT bigram final: 2.5439   (0.012 apart)
(claude's run: 2.5779 vs 2.5440 — 0.034 apart)
```

`estimate_loss` carries ±0.01–0.03 of sampling noise, which accounts for most of
the run-to-run spread above. **The models are more stable than the measurements
are.** Do not chase a 0.02 "improvement" in later days.

**What the curves show that the final line hides:** by step 2000 both attention
models (2.57 / 2.49) had already beaten the bigram's *step-10,000* score
(2.53–2.58). The bigram spends its whole run crawling to where attention arrived
in a fifth of the time. Also, multi-head's val loss is non-monotonic (2.3901 at
step 6000, 2.3578 at 7000, 2.3373 at 9000) — that is `estimate_loss` sampling
noise, not regression; real regression is a sustained rise, not a single blip.

## Generated text

All three still gibberish — **expected, not a bug**. No MLP, no residuals, no
LayerNorm, one layer only (Days 7–10). Multi-head alone was never going to fix
it. Levels are visibly different though: bigram produces consonant pileups no
English word allows (`JjPok'ANChey`); single-head produces word-shaped chunks in
one long run-on; multi-head leaks real function words (`the and,` `but a ware`
`I by`) and some line structure. Honest caveat: at ~2.3 nats, eyeballing samples
is weak evidence — a story could be told either way. **The loss number is the
measurement; samples are only a sanity check that nothing is catastrophically
broken.**

## Misconceptions corrected today

- **PENDING** — the "which piece was foggy before?" question was not answered
  this session. To revisit at the start of Day 7.
- Observed during questioning rather than self-reported: Q2 (what `proj` solves)
  and Q3 (identical-weight heads) were answered correctly and unaided. Q1
  (parameter parity) needed the arithmetic broken into steps — `32×8=256`, then
  `×3`, then `×4` — and there was a slip at `256×3` (758 → 768) worth noting
  only because the whole point was two totals colliding exactly. Q4 (the
  scaling) was the genuine gap: the sticking point was that a *constant* divisor
  seems like it should not matter since it does not change which score is
  largest. Resolved by measuring — it changes the *gaps*, and softmax
  exponentiates gaps.

## Environment gotcha (cost ~15 min)

`attention_lm.py` died with `FileNotFoundError: 'data/input.txt'`. Cause: PyCharm
ran `gpt.py` with **working directory = `src/`**, so the relative path resolved
to `src/data/input.txt` while the shell (running from project root) looked at
`data/input.txt`. Same script, same relative path, two `cwd`s, two files. Fixed
by moving `src/data/` → `data/`. **Set PyCharm's Run Configuration working
directory to the project root**, or every future day repeats this.

Secondary lesson: the first background run reported *exit code 0* despite
failing, because it was piped through `tee` and a shell pipeline reports the
last command's status. Use `set -o pipefail`, or do not pipe the command whose
exit status matters.

## My quiz answers

> **DEFERRED** — quiz skipped this session by request; the 5 retrieval
> questions open the Day 7 session instead (better spacing anyway). Questions
> held over:
> 1. What does `softmax(q·kᵀ/√d)` compute, and what does `wei @ v` mean?
> 2. What quantity does `1/√head_size` hold roughly constant, and why care?
> 3. Same parameter budget as one wide head — so why is multi-head more
>    expressive?
> 4. Why can multi-head's output drop straight into a residual connection?
> 5. What does multi-head *not* fix, and why is that fine?

## One line for tomorrow

Attention moves information *between* tokens; nothing so far lets a token do any
nonlinear computation *on* what it gathered — every path from input to logits is
a chain of linear maps. That is Day 7's FeedForward MLP: **attention
communicates, MLP computes.**

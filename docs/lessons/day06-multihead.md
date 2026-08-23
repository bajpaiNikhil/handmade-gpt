# Day 6 — Multi-head attention (the redo)

## Session card

- **Goal:** understand multi-head attention deeply enough to re-explain it to
  someone else, and re-derive every design choice in `src/multi_head_attention.py`.
- **Style:** teach existing file only — no retyping. We rebuild understanding,
  not the file.
- **Files:** `src/multi_head_attention.py` (walk it), `src/attention_lm.py` (run it).
- **Numbers already logged (match these):** bigram 2.5390, single-head 2.4496,
  multi-head 2.3217. Params: 4,225 / 7,553 / 8,609.
- **Prerequisites:** Days 1–5. This session re-derives what multi-head stands
  on from the bottom up.

## Why we are redoing this (say this to the user up front)

The Day 6 file was built and it trained, but the understanding didn't anchor —
that's normal, and it is exactly why `LOG.md` + notes exist. Today we go slow.
Every concept below is taught from first principles before any file is touched.
By the end the file should read like prose, not a mystery.

---

## Part 1 — LECTURE (teach first, no questions yet)

Speak these sections as board-work. Pause at WRITE THIS DOWN moments so the
student can capture the sentence.

### 1.0 Where we are on the map

Days 1–5 produced a working single-head attention model. Its val loss (2.4496)
beats the bigram (2.5390) — attention's "look back 8 characters" genuinely
helps. But the model can still only ask **one question about its past per
token**. Today's question: what if a token could ask several questions at once?

### 1.1 Matrix multiply is just a batch of weighted sums (ground up)

A matrix multiply `A @ B` with shapes `(m, n) @ (n, p) -> (m, p)` computes, for
each output cell, a dot product: one row of `A` dotted with one column of `B`.
A dot product is a **weighted sum**: `a·b = Σᵢ aᵢ bᵢ`, where `bᵢ` weights how
much each component of `a` counts.

Why this matters for everything today: attention `wei @ v` is exactly this — a
set of weighted sums, where the weights (`wei`) say *how much* each past token
contributes and `v` says *what* each past token contributes.

### 1.2 Dot product as similarity

For two vectors, `a·b = |a||b|cos(θ)`: the larger the alignment, the larger the
dot product. So a dot product is a cheap, differentiable **similarity score**.
This is the entire semantic of Q/K matching: high dot → "these two tokens are a
good pair for this question".

### 1.3 A single self-attention head, re-derived (Day 5 recap)

Each token produces three vectors from its embedding:

- **Query (q)** — "what am I looking for?"
- **Key (k)** — "what do I offer to be found by?"
- **Value (v)** — "what do I hand over once I'm matched?"

Three separate `nn.Linear` layers produce these, **not one shared matrix**. The
reason is semantic: *how I get found* and *what I hand over* are different
questions, and forcing them to share weights would fuse them.

The mechanism:

```
wei[t,s] = q[t] · k[s] / sqrt(head_size)     affinity of t for s
wei       = mask future positions -> -inf
wei       = softmax over past positions      -> probability distribution
out[t]    = Σ_s wei[t,s] · v[s]              weighted blend of values
```

WRITE THIS DOWN: **attention is a content-aware weighted average of the past.
The weights are computed from the content (via Q/K), not fixed.**

#### 1.3.1 Why `/sqrt(head_size)` (derived, not memorized)

`q` and `k` have `head_size` components, each roughly zero-mean. A sum of
`head_size` independent zero-mean products has variance that grows like
`head_size` (variances add). With big head_size, the raw dot products get
large, and softmax saturates: nearly all probability mass lands on one
position, gradients to every other position vanish. The scaling
`* head_size ** -0.5` divides the scores by `sqrt(head_size)`, which divides
the *variance* by `head_size` (because `Var(c·X) = c²·Var(X)`) — restoring
softmax to a working regime. This is why it is called *scaled* dot-product
attention. Day 5 verified this empirically: unscaled variance 2.08, scaled
0.13, and the softmax row went from a peaked 0.49-on-one-position to spread
0.09–0.20.

#### 1.3.2 Why the mask

`tril` keeps `inf` out of the future. The reason is autoregression: at
generation time token `t` simply does not exist yet, so a training-time model
that could read it would be cheating — it would use answer-on-the-slate. Also,
weighted sums that could see the future aren't composable: Day 10 stacks these
blocks, and you cannot stack a layer that peeks.

### 1.4 The problem with one head — the capacity argument

One head gives every token exactly **one (query, key, value) triple** — one
question it gets to ask about its past. But natural text needs many
simultaneous relationships for what to predict next:

- which position precedes me (the previous-token pattern);
- what token introduced "the" earlier (co-reference);
- whether the next char is after a capital letter;
- what rhymes / what follows a blank at line start;
- in a stack: "A B ... A → expect B" (induction).

One Q/K/V triple cannot encode all of these at once — each is a different
similarity measure, and they'd fight in the same matrices. Mechanistic
interpretability work on real transformers shows different **heads specialize**
into different roles; that is the empirical payoff of multi-head.

WRITE THIS DOWN: **one head = one learned notion of "similar". Language needs
many notions of "similar" at the same position. Multi-head supplies them in
parallel.**

### 1.5 The multi-head design response

Run `H` heads side by side over the same input, each with its own
independently-initialized Q/K/V weights, each with `head_size = n_embd / H`.
Each head outputs `(B, T, head_size)`. Concatenate along the last dim →
`(B, T, n_embd)`. Then pass through one more learned linear layer, `proj`.

Two design points to dwell on:

**Why head_size = n_embd / H (the FLOP-parity argument, derived):**
A single full-width head has 3 Q/K/V matrices of shape `(n_embd, n_embd)`.
With `H` heads of width `n_embd/H`, the total width across heads is
`H · (n_embd/H) = n_embd` — so the total Q/K/V parameter count is identical.
Multi-head buys *several smaller, independent questions* for the **same cost**
as one big question. Measured (from the file): single head of size 32 = 3,072
params; 4 heads of size 8 = 3,072 params. The only extra cost is `proj`.

**Why heads must be independent (diverge):** if two heads shared weights they'd
ask identical questions, so 4 heads would just duplicate one head's answer 4
times — no extra information. Independence comes from random initialization
then separate optimization. The file verifies the premise *before* building
anything (STEP 3).

**Why `proj` exists (not just plumbing):** a raw concat glues four answers side
by side in four separate subspaces — nothing ever mixes *across* heads. `proj`
is a learned `(n_embd, n_embd)` transform that lets the model combine what
different heads found before handing the result onward. It also gives heads the
room to "talk": later (Day 8) this `proj` output is what gets added back via a
residual connection.

WRITE THIS DOWN: **multi-head = H independent questions in parallel, same
parameter budget as one wide head; concat reassembles, proj lets the model mix
across what different heads found.**

### 1.6 The reading of the loss numbers (preview of VERIFY)

On identical training conditions the three-way comparison should come out
ordered bigram > single-head > multi-head, with multi-head winning by
~0.13 nats over single-head for ~14% more parameters. That is the whole day:
multi-head's advantage is architecture, not parameter count.

---

## Part 2 — QUESTIONS (ask, wait, validate, correct)

Ask all four. Do not reveal any until the user has attempted it. Mark right /
refine wrong. Suggested grading notes in the answer key.

**Q1.** Why do 4 heads of head_size 8 cost the same Q/K/V parameters as one
head of head_size 32? Do the arithmetic.

**Q2.** What problem does `proj` solve that a plain concatenation leaves
unsolved?

**Q3.** Suppose all heads in a multi-head layer had *identical* weights.
Would multi-head still beat single-head? Why or why not?

**Q4.** What goes wrong if we delete the `* k.shape[-1] ** -0.5` scaling in a
head, and roughly *why* does it go wrong?

**Answer key (for Claude):**
- **A1.** Each head has 3 matrices of `(n_embd, head_size)`. Four heads at
  size 8: `4 · 3 · (32·8) = 3,072`. One head at size 32: `3 · (32·32) =
  3,072`. Identity because `H · (n_embd/H) = n_embd`. The head count and
  head size are inversely coupled by design — total projection width is fixed.
- **A2.** Concat only places each head's answer in its own subspace; nothing
  can combine information across heads. `proj` is a learned affine/linear
  transform that mixes the subspaces back together into the shared `n_embd`
  space the rest of the model (and later, residual connections) expects.
- **A3.** No. Identical weights = identical questions. The concat would be one
  head's answer repeated `H` times; `proj` could at best learn to re-derive a
  single head. The *independence* is the whole benefit — different questions,
  different patterns. This is why the file checks divergence first (STEP 3).
- **A4.** Scores grow unboundedly with head_size, softmax saturates (≈argmax),
  attention concentrates on one token, and gradients to all other positions
  vanish — training degenerates into "copy nearest/exact match", losing the
  gradient signal that shapes attention. The scale factor keeps score variance
  ~constant (≈1) as head_size grows.

---

## Part 3 — BUILD / WALK: `src/multi_head_attention.py`, function by function

For each STEP: state **what** the step does, **why** we do it, **how** it
links to Part 1. Read code comments together as we go — the file's comments are
already a lecture.

**STEP 1 — toy data `(B, T, C) = (4, 8, 32)`**
- *What:* random input, batch 4, time 8, channels 32 (= n_embd).
- *Why:* toy shapes keep every number small enough to inspect by eye, exactly
  like Day 5's `causal_average.py`. C=32 because head_size must divide n_embd
  for `num_heads * head_size == n_embd`; 32 divides by 4 evenly.
- *How:* mirrors the real embedding dimension used in `attention_lm.py`
  (n_embd = 32), so what we prove on toys transfers to the trained model.

**STEP 2 — `Head`, identical to Day 5**
- *What:* one self-attention head, redefined here so the file runs standalone
  (matches the repo's self-contained convention, as `bigram.py` duplicates the
  bigram model inside `attention_lm.py`).
- *Why:* nothing new to learn — multi-head is built from unchanged heads. The
  point is that multi-head is a *container* over existing machinery.
- *How:* same Q/K/V, scale, mask, softmax as section 1.3.

**STEP 3 — two heads, same input, different weights: do they diverge?**
- *What:* instantiate two `Head`s on the same `x` and peek at their `wei` rows
  for token 3 via a `peek_wei` helper (which re-walks `Head.forward` internals
  because `forward` only returns `wei @ v`, hiding the weights — the same
  inspect-the-intermediates spirit as Day 4).
- *Why:* prove the premise of multi-head *before* building it. If the two rows
  were equal (Allclose True), parallel heads would be pointless.
- *How:* this is the empirical check behind Part 1 Q3. Expect
  `torch.allclose(...) = False` and visibly different probability rows.
- *Why a helper and not a change to `Head.forward`:* keep `Head` production-
  clean; the helper is inspection-only, the same philosophy as
  `causal_average.py` exposing intermediates a real forward wouldn't.

**STEP 4 — `MultiHeadAttention`: cat + proj**
- *What:* `nn.ModuleList` of `num_heads` `Head`s; forward runs all heads,
  `torch.cat` on dim=-1, then `proj`.
- *Why the two pieces:* concat to get back to full `n_embd` width (and to make
  this a drop-in replacement for a single `Head` — same output shape); proj to
  mix across heads (section 1.5).
- *How:* output shape `(B, T, n_embd)` equals input shape — this exact shape
  property is what Day 8's residual connections (`x = x + attn(x)`) will rely
  on. Worth saying out loud now.
- *Why ModuleList:* it registers each head's parameters with the module so
  `.to(device)` and `.parameters()` see every head without a plain list losing
  them.

**STEP 5 — parameter count check**
- *What:* compute single-head-32 vs multi-head params and decompose the
  multi-head total into heads-only + proj.
- *Why:* explicitly the arithmetic of section 1.5 — the design intent was
  FLOP-parity; we verify the math on the real objects.
- *How:* expect heads-only total == single-head total (3,072), proj = 1,056
  (the `(32·32) + 32` including its bias).

When the walk is done, run it:

```
source .venv/bin/activate
python3 src/multi_head_attention.py
```

---

## Part 4 — VERIFY by running

1. Run `python3 src/multi_head_attention.py` from the project root. Read every
   printed line together; confirm the rows diverge, shapes match, params add up.
2. Then run the real-data comparison:
   ```
   python3 src/attention_lm.py
   ```
   It trains three models (bigram, single-head, multi-head) for 10,000 steps
   each on MPS — expect a total runtime on the order of several minutes.
3. Compare printed final val losses to LOG.md: expect ≈ 2.5390 / 2.4496 /
   2.3217. Small stochastic differences (seeded only inside the toy files, not
   here) are fine; the *ordering* must hold. If ordering breaks, stop and debug
   together — never hand-wave it.
4. Read the three generated samples and name what each model's failure mode
   looks like (bigram ≈ letter-frequency gibberish; single/multi-head still
   gibberish because there is no MLP/residual/LN yet — that is Days 7–9, not a
   bug).

---

## Part 5 — QUIZ (retrieval) with answers for Claude

Grades: right / partial / wrong, one-line reason each.

- **Q1.** In your own words: what does `wei = softmax(q · kᵀ / √d)` compute, and
  what does the output `wei @ v` *mean*? (Expect: a probability distribution
  over visible past positions, then a weighted blend of their values.)
- **Q2.** Why does scaling by `1/√head_size` matter and what quantity does it
  keep approximately constant? (Variance of raw scores ≈ 1.)
- **Q3.** Multi-head keeps the Q/K/V parameter budget identical to a single
  full-width head. What makes it strictly more expressive anyway? (Independent
  questions / specialization, not more parameters.)
- **Q4.** Why can a multi-head layer's output be dropped straight into a
  residual connection later? (Output shape `(B, T, n_embd)` == input shape.)
- **Q5.** What would multi-head NOT fix on its own, and why is that fine?
  (Still no per-token nonlinear computation, no layer stacking stability — that
  is exactly MLP/residual/LN, the next three days.)

---

## Part 6 — LOG entry draft, notes, look-ahead

**LOG entry:** draft a newest-on-top entry titled "Day 6 (redo) — multi-head
attention understood and verified on existing files", summarizing: the
re-taught mental model (one head = one question; H independent questions in
parallel at equal parameter cost; concat + proj), the re-verified toy numbers
(diverge False, params 3,072 == 3,072, proj 1,056), the matched 3-way val
losses, and one honest note about which bit was confusing before and cleared
today. Echo the user's established style (see existing entries).

**Notes draft:** fill `docs/notes/day06-multihead.md` using the template; the
my-words summary should be *written by the user* in conversation (claude asks
"put today's core idea in your own words" and the user's answer goes in), then
Claude fills in the corrected version + quiz answers + concepts. User reviews
and approves the final file.

**Look-ahead:** next is Day 7 — the FeedForward MLP. Attention gives tokens
information from their neighbors; the MLP is what lets each token *do*
something nonlinear with it ("attention communicates, MLP computes").

**Stretch (optional seminar):** about multi-head in real models — read a few
head-specialization examples from Anthropic's "In-context learning and
induction heads"; note here that induction heads (the "A B ... A → B" pattern)
need *two* attention layers, which is why stacking (Day 10) is where heads get
really interesting.
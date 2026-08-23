# Day 10 — Assemble the Block + stack (the first real GPT)

## Session card

- **Goal:** assemble the complete `Block` (`LN → MultiHeadAttention → residual`,
  `LN → FeedForward → residual`), add dropout, stack N blocks, and train the
  first genuinely complete GPT. It generates.
- **Files:** new `src/gpt_block.py` — the real assembly; config knobs
  (`n_layer`, `n_head`, `n_embd`, `block_size`, `dropout`) as constructor args.
- **Numbers to relate:** best val loss so far (Day 9 block); Day 10 stacks and
  should land materially lower on real Shakespeare.
- **Prerequisites:** Days 1–9. All four mechanism days (6–9) plug together
  today, in exactly the "communicate → compute → repeat" loop.

---

## Part 1 — LECTURE (teach first)

### 1.0 Where we are — the full recipe

Everything needed for a GPT now exists as proven pieces:
- token + position embeddings (Day 1 / detour)
- multi-head self-attention (Day 6)
- FeedForward MLP (Day 7)
- residual connections (Day 8)
- LayerNorm, pre-norm (Day 9)

Today they go into a box — `Block` — and N copies of that box get stacked,
wrapped with a final LayerNorm + a readout linear. That is the classical
decoder-only transformer.

### 1.1 The Block, spelled out

```
x = embedding(token) + position(t)                      -- (B, T, n_embd)
for each of n_layer blocks:
    x = x + MultiHeadAttention( LayerNorm(x) )          -- communicate
    x = x + FeedForward( LayerNorm(x) )                 -- compute
x = LayerNorm(x)                                        -- final stabilize for readout
logits = Linear(x)                                      -- (B, T, vocab)
```

- **Both sublayers use pre-norm** (Day 9): normalize *before*, add back onto
  the bare residual path.
- **Why the final LN?** The residual stream accumulates arbitrary magnitudes
  from many layers; the LM head is a plain linear readout, so it benefits from a
  well-scaled, normalized read (GPT-2 does exactly this).
- **Why attention (and often the first MLP) w/o bias** before LN: LN absorbs
  shifts (Day 9·1.2); keep `bias=False` where the sublayer feeds into a LN.

### 1.2 Why stack at all — depth is where attention gets clever

One block can look back `block_size` positions with a handful of heads. Two
blocks do something new: layer-2 attention can attend to *what layer-1 already
blended*. The canonical fruit of this is the **induction head**: in a pattern
`A B ... A`, a two-layer stack learns "A preceded B, and I am again after A, so
predict B" — i.e. copying/chunking, the substrate for in-context learning.
Strong evidence in real transformers (Olsson et al.) that induction heads
emerge specifically at *two* attention layers. Day 10's stacked model is where
this becomes possible at all.

WRITE THIS DOWN: **one block = one communicate+compute round. Stacking is what
lets later blocks build on earlier rounds — the depth at which attention gets
"clever" (e.g. induction).**

### 1.3 Dropout — the new piece, and its two placements

- **On the attention weights** (`p` on `wei` after softmax): randomly drop some
  attention edges each forward. Prevents one token's pattern from cementing and
  improves generalization (a cheap regularizer; the standard `Dropout(wei)` in
  `Head.forward` between softmax and `@ v`).
- **On the block output** (after each sublayer, before adding back to the
  residual): randomly snips part of the sublayer's write to the stream.
- Small models use `p ≈ 0.2`; GPT-2 used 0.1. It is a *training-mode-only*
  regularizer (off at eval/generation via `model.eval()` — we already call
  `.eval()` in `estimate_loss`).
- Why not more aggressive dropout: it both regularizes AND slows learning; at
  our tiny scale 0.0–0.2 is the sane range. It's a knob Day 11 tunes.

### 1.4 Configuration as a class — "knobs, not constants"

`GPT(vocab, n_embd, n_head, n_layer, block_size, dropout)` and `Block` takes
its subconfig. Day 11's whole job is sweeping these; building them in as
constructor args today is what makes that sweep possible.

### 1.5 What "it generates" means today

With a full stack, output should finally leave "gibberish that has correct
letter frequencies" behind and start to show word- and phrase-like structure
and shape resembling Shakespearean stage directions. Not *good* text — that is
Day 11's scale-up — but recognizably structured. The bar for "working GPT" per
the campaign is set there.

---

## Part 2 — QUESTIONS (ask, wait, validate, correct)

**Q1.** Write the block dataflow in one line, including where LayerNorm goes
and where the residual add lands. (Expect: `x + attn(LN(x))`, `x + ffn(LN(x))`.)
**Q2.** Why two dropout placements, and which one is *off* at evaluation time?
(On wei + on sublayer output; both—`model.eval()` disables them.)
**Q3.** Why does the final LayerNorm exist if every sublayer already has one?
(Stabilize the accumulated stream for the plain linear readout.)
**Q4.** What specifically becomes possible at *two* stacked attention layers
that one layer cannot do? (Induction: "A B ... A → predict B"; composition of
round-1 blends.)
**Q5.** Why is `block_size` cropping in `generate()` still required, and where
does it bite? (Position table has `block_size` rows; feed only the last
`block_size` tokens — the detour file already does this.)

**Answer key** maps to 1.1–1.5.

---

## Part 3 — BUILD (step by step, each explained)

**STEP 1 — `Head` + dropout:** one added line after softmax:
`wei = self.dropout(wei)`. Register `self.dropout = nn.Dropout(p)`.
- *Why:* the attention-edge regularizer (1.3).

**STEP 2 — `MultiHeadAttention` + dropout?** Keep heads as-is (each head's wei
is dropped); add dropout on the *proj output* before it returns (the sublayer
write gets stochastically thinned too). Explain the placement.

**STEP 3 — `FeedForward`:** add an optional dropout after the second Linear
(again "thin the write"), `expanded` state from Day 7 unchanged.

**STEP 4 — `Block`:**
```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa  = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffn = FeedForward(n_embd, dropout=dropout)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```
- *Why* `n_embd // n_head` recomputed here — the head-size-parity argument of
  Day 6 shown as a constraint, not magic.

**STEP 5 — `GPTLanguageModel`:**
- token embedding `(vocab, n_embd)`, position embedding `(block_size, n_embd)`,
  `nn.Sequential([Block(...)] * n_layer)`, final `LayerNorm(n_embd)`,
  `lm_head = nn.Linear(n_embd, vocab)`; forward = embeddings → blocks → LN →
  logits; `generate()` with the crop (Day 6 detour pattern); `estimate_loss`
  reused verbatim.

**STEP 6 — param + dtype sanity:** print total params and a quick forward-shape
check on toy `(B, T)`; then train a small config first
(`n_embd=64, n_head=4, n_layer=2, block_size=64, dropout=0.1, batch 16`) on
real Shakespeare and generate. Adjust only if OOM/slow on MPS.

**STEP 7 — compare + sample:** val loss vs Day 9's single block; sample 300
chars; read the structure out loud ("does it smell like a stage direction?")
and record everything.

---

## Part 4 — VERIFY

- Shapes: `(B, T, n_embd)` in → `(B, T, vocab)` logits out.
- Params printed; total and per-component (embeddings/attn/ffn/LN/head).
- Val loss down vs Day 9; train/val gap sane.
- Sample: word-like fragments, line/paragraph structure, any repeated "name:"
  speaker pattern.

---

## Part 5 — QUIZ (answers for Claude)

- **Q1.** Full forward dataflow of one block, one line. (See answer key Q1.)
- **Q2.** Two reasons the final LN exists. (Stabilize stream; feed scaled
  inputs to plain readout.)
- **Q3.** What does stacking buy over a single wide block of the same total
  params? (Composition/induction-type behaviors; recursing on blends.)
- **Q4.** Why is dropout dropped at eval and by what mechanism? (`nn.Dropout`
  is a no-op in eval mode; `estimate_loss` calls `model.eval()`.)
- **Q5.** If `n_head=4` and `n_embd=96`, what is `head_size` and why must it be
  an integer divisor? (24; because concat must reproduce `n_embd` exactly.)

---

## Part 6 — LOG entry, notes, look-ahead

- **LOG:** "Day 10 — Assemble the Block + stack" (first real GPT): config,
  params, val loss vs Day 9, stable gradient check optional, sample + its
  structure, one wrap-up line ("the machine exists: a complete decoder-only
  transformer, every layer hand-typed").
- **Notes:** `docs/notes/day10-block-and-stack.md`.
- **Look-ahead:** Day 11 — scale. The architecture is done; the loss is still
  high and the text rough. Day 11 turns the knobs (bigger `n_embd`, more
  layers/tokens/context, LR warmup + weight decay) to push toward the
  Shakespeare **floor (~1.48 nats)** as far as an M4 Pro can, then writes
  Article 1.
- **Stretch:** Olsson et al., "In-context learning and induction heads" — read
  the two-layer intuition section; check whether our own 2-layer model shows
  signs of it.
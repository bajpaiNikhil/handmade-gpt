# Lessons — index + schedule

This folder holds the lesson plans for the whole campaign. Each lesson is
written so a teaching session can run straight from it: lecture text, question
bank with answer key, build/walk steps, verification numbers, quiz, notes
guidance, and article briefs at save-points.

Session order and calendar live in the **Week schedule** below. When the user
says "start today's session", map today's date to a session here, open the
file, and begin the lecture.

---

## Week of Mon Aug 10 — finish World 2: The Block (Day 6 → Day 11 BOSS)

| Day | Session | File to open | What gets built |
|-----|---------|--------------|-----------------|
| Mon | Setup + Day 6 redo | `day06-multihead.md` | teaching env (this set of files) + a full re-teach of multi-head over the existing `src/multi_head_attention.py` |
| Tue | Day 7 | `day07-feedforward.md` | `src/feedforward.py`, wire MLP into `attention_lm.py` |
| Wed | Day 8 | `day08-residuals.md` | `src/residuals.py`, residual connections + gradient-norm experiment |
| Thu | Day 9 | `day09-layernorm.md` | `src/layernorm.py` (hand-written LN), stability experiment |
| Fri | Day 10 | `day10-block-and-stack.md` | `src/gpt_block.py` — complete GPT block + stack, trains + generates |
| Sat | Day 11 | `day11-scale-and-ship.md` | scaling knobs, final train, BOSS done, Article 1 brief |
| Sun | — | (article brief in day11) | **Article 1** writing day (`medium-article-writer` skill) |

Notes: `docs/notes/` mirrors the same numbering (one template per session, plus
`concepts-index.md`). `LOG.md` gets a newest-on-top entry each teaching day.

---

## Reading order / review block

These are not blockers — they deepen Days 1–6 "from start to finish". Slot them
into a quiet session or the week after:

| File | Covers |
|------|--------|
| `00-foundations-math.md` | linear algebra, probability, information theory from zero (why 4.17 = −ln(1/65), why ~1.48 is the Shakespeare floor) |
| `01-backprop-and-autograd.md` | chain rule → computational graph → why `loss.backward()` works, micrograd-style |
| `02-w1-retrospective.md` | deep-review of Days 1–6 code as built: gpt.py, bigram.py, causal_average.py |

---

## World map after W2

| World | Sessions | Save point |
|-------|----------|------------|
| W3 — Modern recipe | `w3-rmsnorm.md`, `w3-rope.md`, `w3-swiglu.md`, `w3-gqa.md` | Article 2 |
| W4 — Real language | `w4-bpe-tokenizer.md`, `w4-tinystories.md` | Article 3 |
| W5 — On device | `w5-quantization.md`, `w5-android.md` | Final article |

These lesson files are written when the user reaches that world (same boot-and-go
pattern); the W2 files above are the ones currently seeded.

---

## Session rules (short form)

- Teach first → ask → validate/correct → build (every step's what/why) → verify
  by running → notes draft (user approves) → quiz → LOG entry draft → look-ahead.
- No `nn.MultiheadAttention` / `nn.Transformer` / `nn.LayerNorm`. Hand-typed,
  PyTorch only, everything measured.
- No hard time cap; finish a concept before advancing; Sunday absorbs leftovers.
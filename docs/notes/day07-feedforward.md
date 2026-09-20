# Notes — Day 7: The FeedForward (MLP)

> Claude fills at session end; user reviews and approves.

## My-words summary (user)

> Attention gathers info from other words, feed forward actually processes it.

## Corrected / completed (Claude)

That's the core of it, expanded: attention's "gathering" is a weighted
**average** of other tokens' value vectors — mathematically linear, so no
matter how many heads run in parallel, the values themselves only ever get
proportionally blended. There is no step where the model reacts to or draws
a conclusion from what it just gathered. FeedForward is that missing step:
a small network — expand to 4x width, apply a smooth nonlinear gate (GELU),
collapse back down — applied with the **same weights at every token
position** (one shared "cookie cutter," not one per position). This buys
efficiency (trained on every position of every example at once) and
transfer (a pattern learned at position 3 works identically at position 7).
Wired in as `attention → FeedForward` ("communicate, then compute"), it was
the single largest loss improvement since single-head → multi-head, and it
roughly doubled the model's total parameter count in one step — which is
exactly why it's also the first day the train/val gap became worth
watching for overfitting.

Under the "key-value memory" lens (Geva et al.): the first linear layer's
rows act as pattern-matchers (keys), the second layer's rows as the
answers read out once a pattern matches (values) — the librarian analogy.
This is where most of a transformer's learned "facts" live, and typically
accounts for ~2/3 of total parameters in real models.

## WRITE THIS DOWN captures

- Attention exchanges info; it does not compute a nonlinear function per token.
- FFN ≈ per-token nonlinear processing; in the "memories" view, where facts
  live; holds ~2/3 of a transformer's params.

## Misconceptions corrected today

- **Q1 (why attention alone is insufficient):** initial answer correctly
  identified "no nonlinearity introduced" but conflated it with a second,
  unrelated idea — "we're just not aware of the facts we can extract from
  the training set." Corrected: the missing piece isn't insufficient
  *data*, it's that a weighted average is mathematically incapable of
  producing a reaction/judgment no matter how much data or training it
  gets. That "facts live somewhere" idea belongs to a later point (the
  key-value memory lens), not to why attention alone is structurally
  insufficient.
- **Q4 (overfitting risk):** the mechanism (more params → more storage
  capacity → higher memorization risk) was correctly identified, but the
  requested real-world anchor (student memorizing practice-test answers
  vs. understanding the material, visible as a gap between practice-test
  and real-exam performance) wasn't stated — added afterward to anchor the
  idea to something concrete for retention.

## My quiz answers

- **Q1** (attention vs FFN roles) — Right.
- **Q2** (why 4x expansion) — Partial: had the "whiteboard/ideation" analogy,
  missing that 4x is a heuristic (paper's choice, copied since), not a rule.
- **Q3** (why shared weights per position) — Right: more practice reps +
  transfer across positions.
- **Q4** (overfitting risk) — Partial: had the mechanism (more params → more
  memorization risk), missing the concrete tell: the train/val loss gap
  widening (+0.0127 → +0.0367/+0.0428 across the two runs).
- **Q5** (key-value memory lens) — Partial: had the two-step process, missing
  which specific layer is the pattern-matcher (first) vs. the answer-lookup
  (second).

## Numbers logged (LOG.md)

- FFN toy params (n_embd=32, 4x expansion): predicted 8,352 by hand,
  matched exactly by PyTorch. Position-invariance proof: identical output
  (`torch.allclose`, atol=1e-6) for the same content placed at position 0
  vs. position 3.
- Full model params: multi-head-only 8,609 → multi-head+FFN 16,961 (1.97x
  — the FFN alone is 97% the size of the entire Day 6 model).
- Two independent unseeded runs of `attention_lm.py` (terminal, then
  PyCharm after the working-directory fix):

  ```
                          run 1 (terminal)   run 2 (PyCharm)
  multi-head val loss     2.3246             2.3881
  multi-head+FFN val loss 2.2506             2.2499
  FFN vs multi-head gain  +0.0740            +0.1382
  FFN train/val gap       +0.0367            +0.0030
  multi-head-only gap     +0.0127            +0.0428
  ```

  Exact gap numbers swing run to run (no fixed seed in training, same
  noise-floor lesson as Day 6) — but the FFN improvement itself held in
  both runs, which is the defensible claim: **FeedForward measurably
  lowers loss beyond multi-head attention alone**, not "by exactly X."
- Generated text: still gibberish for all four models (bigram, single-head,
  multi-head, multi-head+FFN) — expected, no residuals or LayerNorm yet.

## Environment gotcha (recurrence of Day 6's)

Hit the *exact same* `FileNotFoundError: 'data/input.txt'` from Day 6's
notes, for the exact same reason: PyCharm's auto-generated "temporary" run
configuration for `attention_lm` had `WORKING_DIRECTORY = $PROJECT_DIR$/src`
instead of the project root. Confirmed via git that `src/attention_lm.py`
itself hadn't changed since June 26 — the bug was 100% environment, not
code. Root cause this drifted silently: `.idea/workspace.xml` (where run
configs live) is untracked by git (`.idea/.gitignore` excludes it), and
PyCharm's temporary configs get regenerated from defaults over time — so
Day 6's fix never persisted anywhere durable. Re-fixed by setting
`WORKING_DIRECTORY` back to `$PROJECT_DIR$`. **Still outstanding:** the
run configuration is still marked `temporary="true"` — until it's saved as
a permanent configuration in PyCharm, this can silently reset again.

## One line for tomorrow

Attention communicates, FeedForward computes — but neither layer's output
is added back to its input yet, so nothing stops gradients from vanishing
as we stack more of these. Day 8: residual connections, `x = x +
sublayer(x)`.

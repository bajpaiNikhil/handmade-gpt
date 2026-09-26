"""
Day 8 — Residual Connections ("why deep stacks don't die")
-------------------------------------------------------------------
STEP 1 (done by hand on paper first): a 3-layer chain where each layer
shrinks its input by a factor of 0.3. Plain stack: 0.3*0.3*0.3 = 0.027 —
a nudge at the start arrives 97% destroyed. Same three layers wired with
an "express lane" (x + f(x)) around each: (1+0.3)^3 = 2.197 — the nudge
arrives AMPLIFIED, because every stage's sensitivity has a guaranteed "1 +"
term that survives no matter how small the layer's own contribution gets.

This file reproduces that same 0.3-per-layer shrink with REAL nn.Linear
layers and REAL autograd, to confirm the hand math holds once PyTorch is
doing the differentiation instead of us. NUM_LAYERS is now 10 (not 3) to see
what depth does to both the gradient AND the actual forward-pass output.
"""

import torch
import torch.nn as nn

torch.manual_seed(1337)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — A Chain of Tiny Layers, Each Scaled to ~0.3 Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
# Same layers (same weights) are reused for BOTH the plain chain and the
# residual chain below — the ONLY thing that differs between no_residual and
# with_residual is the wiring, isolating the effect we're measuring.
dim = 16
NUM_LAYERS = 10   # was 3 (matching the paper derivation) — now pushed to 10

def make_layer(seed_offset):
    torch.manual_seed(1337 + seed_offset)
    layer = nn.Linear(dim, dim)
    with torch.no_grad():
        layer.weight.mul_(0.3)   # scales this layer's local sensitivity to
        layer.bias.zero_()        # roughly 0.3 — matching the paper's knob
    return layer

layers = [make_layer(i) for i in range(NUM_LAYERS)]

def no_residual(x, trace=None):
    """Plain chain: h -> f(h) -> f(h) -> ... No express lane."""
    h = x
    for layer in layers:
        h = torch.tanh(layer(h))
        if trace is not None:
            trace.append(h.norm(dim=-1).mean().item())
    return h

def with_residual(x, trace=None):
    """Same layers, express lane around each: h -> h + f(h)."""
    h = x
    for layer in layers:
        h = h + torch.tanh(layer(h))
        if trace is not None:
            trace.append(h.norm(dim=-1).mean().item())
    return h

print(f"[setup] dim={dim}  NUM_LAYERS={NUM_LAYERS}  each layer's weights scaled "
      f"to ~0.3 sensitivity (matches the paper's a=0.3 per layer)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — What Happens to the OUTPUT Itself, Stage by Stage
# ─────────────────────────────────────────────────────────────────────────────
# Track the average output vector norm after EACH layer, for both chains, on
# the same input. This shows the forward-pass consequence of depth directly —
# not just the gradient story from STEP 4 below, but what the values themselves
# do as they pass through 10 stages with vs without the express lane.
torch.manual_seed(7)
x0 = torch.randn(4, dim)
print(f"\n[input] avg input vector norm: {x0.norm(dim=-1).mean().item():.4f}")

trace_plain = []
y_plain = no_residual(x0, trace=trace_plain)
trace_res = []
y_res = with_residual(x0, trace=trace_res)

print(f"\n{'layer':>6} | {'output norm (plain)':>20} | {'output norm (residual)':>22}")
print("-" * 55)
for i, (np_, nr) in enumerate(zip(trace_plain, trace_res), start=1):
    print(f"{i:>6} | {np_:>20.6f} | {nr:>22.6f}")

print(f"\n[final output] plain  : mean={y_plain.mean().item():+.6f}  std={y_plain.std().item():.6f}")
print(f"[final output] resid  : mean={y_res.mean().item():+.6f}  std={y_res.std().item():.6f}")
print("\n[reading] the plain chain's output norm should COLLAPSE toward ~0 as depth")
print("          increases — each tanh+0.3-scaled layer squashes it a little more,")
print("          and by layer 10 the output has nearly stopped depending on the")
print("          input at all (information destroyed, not just gradients). The")
print("          residual chain's norm should instead keep GROWING, stage by")
print("          stage, since the original signal is never overwritten, only")
print("          added to — this unbounded growth is exactly the NEW problem")
print("          Day 9's normalization (LayerNorm/RMSNorm) exists to fix.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Gradient-Norm Experiment: Does the Earliest Layer Still Learn?
# ─────────────────────────────────────────────────────────────────────────────
# The earliest layer (layers[0]) is the one a vanishing gradient would hit
# hardest, since its correction signal has to survive the longest journey
# backward. Compare its gradient norm (and the raw input's gradient norm)
# with vs without the express lane. Repeated across 3 different random
# inputs/seeds so we're reporting a PATTERN, not one lucky run.
def run_comparison(seed):
    torch.manual_seed(seed)
    x = torch.randn(4, dim, requires_grad=True)

    # --- plain stack ---
    for layer in layers:
        for p in layer.parameters():
            p.grad = None
    y_plain = no_residual(x)
    loss_plain = y_plain.pow(2).mean()
    loss_plain.backward()
    grad_x_plain  = x.grad.norm().item()
    grad_f1_plain = layers[0].weight.grad.norm().item()

    # --- residual stack (fresh grads, same x, same layer weights) ---
    x.grad = None
    for layer in layers:
        for p in layer.parameters():
            p.grad = None
    y_res = with_residual(x)
    loss_res = y_res.pow(2).mean()
    loss_res.backward()
    grad_x_res  = x.grad.norm().item()
    grad_f1_res = layers[0].weight.grad.norm().item()

    return grad_x_plain, grad_f1_plain, grad_x_res, grad_f1_res


print(f"\n{'seed':>6} | {'grad(x) plain':>14} | {'grad(x) resid':>14} | "
      f"{'grad(f1) plain':>15} | {'grad(f1) resid':>15}")
print("-" * 78)
for seed in [1, 2, 3]:
    gx_p, gf1_p, gx_r, gf1_r = run_comparison(seed)
    print(f"{seed:>6} | {gx_p:>14.10f} | {gx_r:>14.6f} | {gf1_p:>15.10f} | {gf1_r:>15.6f}")

print(f"\n[expected] grad(f1) — the EARLIEST layer's gradient — should be")
print(f"           dramatically smaller without the express lane at depth {NUM_LAYERS},")
print( "           consistently across all 3 seeds — likely underflowing to 0.0.")
print("\n[done] Residual toy proof complete — same layers, same weights, only the")
print("       wiring differs, and both the output trace AND the earliest layer's")
print("       gradient tell the same story.")
print("       next: wire x = x + sa_heads(x) and x = x + ffwd(x) into attention_lm.py.")

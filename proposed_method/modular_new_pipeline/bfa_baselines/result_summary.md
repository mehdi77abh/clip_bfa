# Baseline results — Part 1 summary

Generated from `results/**/{eval.json,attack_meta.json}` (seed 42).

**Configuration — push-98**: every baseline is asked for +98% ASR with a **200-bit
budget**; the reported `flips→98` is how many bit flips were needed (blank = never
reached 98%). ASR / CACC / clean→tgt are **held-out** (1000-image eval split,
`evaluate.py`). Target classes: Caltech101=97 ("wild cat"), ImageNet1k=291 ("lion").

**Scope (Part 1)**: TrojViT + ProFlip × {ViT-B/16, ViT-L/14, SigLIP-B/16} ×
{Caltech101, ImageNet1k} = 15 runs. Flip-S and all Food101 runs are **pending**
(Food101 loader bug — to be fixed).

---

## Main results (held-out eval)

Status key: **⚠ collapsed** = attack won by destroying the model (clean→tgt high,
CACC crushed) — *not* a stealthy backdoor; **✓ 98%** = reached 98% while keeping
CACC reasonable; **weak** = stealthy but ASR < 98%.

### Caltech101 (target 97 "wild cat")

| Method | Fitness | Model | ASR ↑ | CACC ↑ | clean→tgt ↓ | Flips | Trigger ASR | flips→98 | Status |
|---|---|---|---|---|---|---|---|---|---|
| trojvit | — | ViT-B/16 | 0.610 | 0.805 | 0.001 | 196 | 0.617 | — | weak, stealthy |
| trojvit | — | ViT-L/14 | 0.491 | 0.912 | 0.004 | 199 | 0.513 | — | weak, stealthy |
| trojvit | — | SigLIP-B/16 | 0.567 | 0.922 | 0.004 | 200 | 0.474 | — | weak, stealthy |
| proflip | mse_sni | ViT-B/16 | 0.947 | 0.072 | 0.924 | 72 | 0.0 | 72 | ⚠ collapsed |
| proflip | cos | ViT-B/16 | 0.974 | 0.049 | 0.956 | 129 | 0.0 | 129 | ⚠ collapsed |
| proflip | mse_sni | ViT-L/14 | 0.986 | 0.219 | 0.766 | 65 | 0.0 | 65 | ⚠ collapsed |
| proflip | cos | ViT-L/14 | 0.921 | 0.431 | 0.543 | 123 | 0.0 | — | ⚠ collapsed |

### ImageNet1k (target 291 "lion")

| Method | Fitness | Model | ASR ↑ | CACC ↑ | clean→tgt ↓ | Flips | Trigger ASR | flips→98 | Status |
|---|---|---|---|---|---|---|---|---|---|
| trojvit | — | ViT-B/16 | 0.767 | 0.672 | 0.000 | 196 | 0.836 | — | weak, stealthy |
| trojvit | — | ViT-L/14 | 0.197 | 0.704 | 0.000 | 199 | 0.247 | — | weak, stealthy |
| trojvit | — | SigLIP-B/16 | 0.257 | 0.763 | 0.000 | 200 | 0.391 | — | weak, stealthy |
| proflip | mse_sni | ViT-B/16 | 0.982 | 0.010 | 0.949 | 72 | 0.0 | 72 | ⚠ collapsed |
| proflip | cos | ViT-B/16 | 0.000 | 0.520 | 0.000 | 201 | 0.0 | — | ✗ failed (ASR 0) |
| proflip | mse_sni | ViT-L/14 | 0.976 | 0.499 | 0.173 | 43 | 2.73 | 43 | ✓ 98% |
| proflip | cos | ViT-L/14 | 0.984 | 0.512 | 0.039 | 77 | 2.73 | 77 | ✓ 98% |
| proflip | mse_sni | SigLIP-B/16 | 0.961 | 0.697 | 0.011 | 200 | 0.0 | — | ok, <98% |

*Flips = bits actually applied (≤ 200 budget; ProFlip multi-bit element updates can
land at 201). `Trigger ASR` = attack success of the trigger alone, before any flip.*

---

## ASR@K — quick-trace search curve (optimistic, see caveats)

ASR on the search-time quick subset at K bit flips (step function over multi-bit
updates). This is the `flips→98` source; it **overstates** the held-out ASR.

| Method | Fitness | Model | Dataset | @10 | @40 | @100 | @200 |
|---|---|---|---|---|---|---|---|
| trojvit | — | ViT-B/16 | Caltech101 | 0.625 | 0.691 | 0.727 | 0.754 |
| trojvit | — | ViT-B/16 | ImageNet1k | 0.824 | 0.824 | 0.824 | 0.813 |
| trojvit | — | ViT-L/14 | Caltech101 | 0.645 | 0.680 | 0.715 | 0.731 |
| trojvit | — | ViT-L/14 | ImageNet1k | 0.262 | 0.340 | 0.395 | 0.379 |
| trojvit | — | SigLIP-B/16 | Caltech101 | 0.512 | 0.563 | 0.609 | 0.656 |
| trojvit | — | SigLIP-B/16 | ImageNet1k | 0.426 | 0.457 | 0.453 | 0.441 |
| proflip | mse_sni | ViT-B/16 | Caltech101 | 0.031 | 0.512 | 0.981 | 0.981 |
| proflip | cos | ViT-B/16 | Caltech101 | 0.000 | 0.008 | 0.926 | 0.981 |
| proflip | mse_sni | ViT-B/16 | ImageNet1k | 0.000 | 0.828 | 0.988 | 0.988 |
| proflip | cos | ViT-B/16 | ImageNet1k | 0.000 | 0.000 | 0.000 | 0.000 |
| proflip | mse_sni | ViT-L/14 | Caltech101 | 0.000 | 0.270 | 0.981 | 0.981 |
| proflip | cos | ViT-L/14 | Caltech101 | 0.000 | 0.125 | 0.793 | 0.945 |
| proflip | mse_sni | ViT-L/14 | ImageNet1k | 0.164 | 0.973 | 0.988 | 0.988 |
| proflip | cos | ViT-L/14 | ImageNet1k | 0.031 | 0.441 | 0.981 | 0.981 |
| proflip | mse_sni | SigLIP-B/16 | ImageNet1k | 0.000 | 0.000 | 0.008 | 0.953 |

---

## Key findings

1. **Neither baseline achieves high ASR *and* stealth.** They split into two failure
   modes:
   - **TrojViT** — stealthy (CACC 0.67–0.92, clean→tgt ≈ 0) but **ASR < 98%**
     (0.20–0.77). `flips→98 = never` on all 6 runs. Its ASR is bounded by trigger
     quality and its bulk distillation can't be realized in 200 flips (it tuned
     6 144–8 192 proj entries but only ~200 bits ≈ <1% could be applied).
   - **ProFlip** — often reaches 98% ASR, but on ViT-B/16 and ViT-L/14-Caltech it does
     so by **collapsing the model** (clean→tgt 0.77–0.96, CACC 0.01–0.22): it predicts
     the target class for *everything*, clean images included. That is not a backdoor.
   - Only **ProFlip on ViT-L/14 & SigLIP-B/16 × ImageNet1k** is a genuine high-ASR +
   reasonable-CACC result (ASR 0.96–0.98, CACC 0.50–0.70, clean→tgt ≤ 0.17).

2. **Trigger strength is wildly model-dependent** (this is the `Trigger ASR` column):
   TrojViT's trigger alone gets 0.836 on ViT-B/16-ImageNet but only 0.247 on
   ViT-L/14-ImageNet. ProFlip's trigger is ~0 everywhere (it relies entirely on bit
   flips). So `flips→98` is not purely "flip efficiency" — it depends heavily on how
   strong each method's trigger is. **Report `Trigger ASR` next to `flips→98`.**

3. **ViT-L/14's TrojViT trigger under-converges.** `L_target` only fell 0.871→0.777
   (cos 0.13→0.22) over 250 iters — the trigger optimiser is not reaching high target
   alignment on ViT-L/14 the way it does on ViT-B/16. TrojViT's ViT-L ASR is therefore
   trigger-limited, not a search failure.

---

## Validity caveats (read before using these numbers)

- **⚠ ProFlip model collapse (most important).** ProFlip runs with `clean→tgt > 0.5`
  are *not stealthy* — the model was driven to predict the target for all inputs. This
  is almost certainly caused by **deviation #6** (the released ProFlip loss adds
  `+2·CE(clean, true labels)` to preserve clean accuracy; our port dropped that clean
  term). As-is, these runs overstate ProFlip (high ASR but zero stealth) and are **not
  comparable** to a stealthy backdoor. → Either restore the clean term, or present them
  explicitly as "ProFlip without clean preservation (non-stealthy)".

- **Quick-trace overfitting.** `flips→98` and the ASR@K table are measured on the
  search-time quick subset, which the search *optimizes* — they **overstate** held-out
  performance. Example: TrojViT ViT-L/14-ImageNet quick peaks at 0.41 but held-out
  eval is 0.197. **Anchor the paper on the held-out `ASR` column**, and treat
  `flips→98` as a search-efficiency curve.

- **One outright failure**: ProFlip `cos` on ViT-B/16-ImageNet reached ASR 0.0 with
  201 flips (the CBS pool/search found nothing that helped). Honest result, but note it.

- **Clean accuracy baseline**: CACC here is the *attacked* model's clean accuracy. For
  reference, clean CLIP zero-shot is ≈0.80 (ViT-B/16-Caltech), ≈0.68 (ViT-B/16-ImageNet),
  ≈0.75 (ViT-L/14-ImageNet). TrojViT barely degrades CACC (good stealth); ProFlip often
  destroys it.

---

## What's still needed (Part 2)

- **Flip-S** — not yet run (0 results).
- **Food101** — all runs blocked by a loader bug (to be fixed next).
- Your method's `trigger ASR` + held-out ASR, for the flip-efficiency comparison.
- Consider restoring ProFlip's clean-preservation term (deviation #6) so its rows are a
  fair, stealthy comparison.

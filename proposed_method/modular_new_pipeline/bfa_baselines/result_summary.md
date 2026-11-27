# Baseline results — summary (23 runs, seed 42)

Push-98 config (target +98% ASR, 200-bit budget). ASR/CACC/clean→tgt are
**held-out** (1000-image eval split). Target classes: Caltech101=97, ImageNet1k=291,
Food101=76. Grid: {trojvit, proflip, flips} × {ViT-B/16, ViT-L/14, SigLIP-B/16} ×
{Caltech101, Food101, ImageNet1k}.

## The headline

**No baseline achieves a stealthy high-ASR backdoor.** Of 23 runs:
- **13 collapse** (CACC < 0.3 — the model is driven to predict the target for
  everything, or is simply destroyed). These are *not* valid backdoors.
- **10 are stealthy** (CACC ≥ 0.5) but **all have ASR < 0.90** (max 0.85).
- **0** runs reach ≥98% ASR while preserving clean accuracy.

So the only "high-ASR" results come from *collapsing* the model (Flip-S reaches
ASR 1.0 with 1–2 flips but CACC ≈ 0). A reviewer will reject those as
non-backdoors unless the collapse is fixed.

## Full results

Status: **COLLAPSED** = CACC < 0.3 (not stealthy); **stealthy** = CACC ≥ 0.5.
`trig%` = trigger-only ASR. `c→tgt` = clean→target rate (lower = stealthier).

| Method | Model | Dataset | ASR | CACC | c→tgt | Flips | trig% | Status |
|---|---|---|---|---|---|---|---|---|
| flips | ViT-B/16 | Caltech101 | 1.000 | 0.514 | 0.450 | 1 | 0.4 | COLLAPSED |
| flips | ViT-B/16 | Food101 | 1.000 | 0.007 | 1.000 | 1 | 0.4 | COLLAPSED |
| flips | ViT-B/16 | ImageNet1k | 1.000 | 0.000 | 0.997 | 2 | 0.0 | COLLAPSED |
| flips | ViT-L/14 | Caltech101 | 1.000 | 0.005 | 1.000 | 1 | 0.4 | COLLAPSED |
| flips | ViT-L/14 | ImageNet1k | 1.000 | 0.000 | 1.000 | 2 | 0.0 | COLLAPSED |
| proflip | SigLIP-B/16 | Caltech101 | 0.000 | 0.916 | 0.003 | 6 | 0.0 | FAILED (no attack) |
| proflip | SigLIP-B/16 | Food101 | 0.006 | 0.005 | 0.006 | 0 | 0.4 | COLLAPSED |
| proflip | SigLIP-B/16 | ImageNet1k | 0.000 | 0.758 | 0.000 | 0 | 0.0 | FAILED (no attack) |
| proflip | ViT-B/16 | Caltech101 | 0.951 | 0.075 | 0.926 | 169 | 0.0 | COLLAPSED |
| proflip | ViT-B/16 | Food101 | 0.978 | 0.007 | 0.849 | 58 | 0.8 | COLLAPSED |
| proflip | ViT-B/16 | ImageNet1k | 0.000 | 0.690 | 0.000 | 0 | 0.0 | FAILED (no attack) |
| proflip | ViT-L/14 | Caltech101 | 0.000 | 0.918 | 0.009 | 0 | 0.0 | FAILED (no attack) |
| proflip | ViT-L/14 | Food101 | 0.006 | 0.007 | 0.006 | 0 | 0.4 | COLLAPSED |
| proflip | ViT-L/14 | ImageNet1k | 0.960 | 0.153 | 0.782 | 110 | 0.0 | COLLAPSED |
| trojvit | SigLIP-B/16 | Caltech101 | 0.719 | 0.922 | 0.003 | 197 | 68.2 | stealthy, <98% |
| trojvit | SigLIP-B/16 | Food101 | 0.958 | 0.004 | 0.006 | 199 | 96.3 | COLLAPSED |
| trojvit | SigLIP-B/16 | ImageNet1k | 0.286 | 0.770 | 0.000 | 200 | 37.0 | stealthy, low ASR |
| trojvit | ViT-B/16 | Caltech101 | 0.843 | 0.921 | 0.002 | 198 | 90.9 | stealthy, <98% |
| trojvit | ViT-B/16 | Food101 | 0.876 | 0.003 | 0.007 | 196 | 89.3 | COLLAPSED |
| trojvit | ViT-B/16 | ImageNet1k | 0.851 | 0.661 | 0.000 | 199 | 94.0 | stealthy, <98% |
| trojvit | ViT-L/14 | Caltech101 | 0.703 | 0.949 | 0.012 | 198 | 79.7 | stealthy, <98% |
| trojvit | ViT-L/14 | Food101 | 0.698 | 0.005 | 0.006 | 196 | 68.4 | COLLAPSED |
| trojvit | ViT-L/14 | ImageNet1k | 0.334 | 0.751 | 0.000 | 198 | 52.1 | stealthy, low ASR |

## Per-method reading

**TrojViT** — the best-behaved baseline. Caltech101 + ImageNet1k are **stealthy**
(CACC 0.66–0.95, c→tgt ≈ 0) but ASR only 0.29–0.85 (**never 98%**). It is
**trigger-dominated** (trigger-only ASR 37–96%): the bit flips add little, and
the 200-bit budget is far below its native TBN (880–1250). **Food101 collapses**
(CACC ≈ 0) — the proj distillation wrecks the model on Food101; needs
investigation. `flips_to_98 = not reached` everywhere.

**ProFlip** — unreliable (⚠️ **these runs predate the clean-CE fix**): 4 runs fail
outright (ASR = 0 — CBS finds nothing), 5 collapse (CACC ≈ 0). No clean stealthy
success. The collapse is exactly what the clean-preservation CE (edit #1) fixes,
so these rows should be **re-run**.

**Flip-S** — **all 5 runs collapse** (ASR 1.0 with 1–2 flips, CACC ≈ 0). The
ASR-gated acceptance (mismatch **F4**, not yet fixed in the default) accepts any
flip that raises ASR even if it destroys clean accuracy, so it drives the model
to "always predict target." Not a backdoor — must fix F4 (loss-based acceptance +
clean preservation) and re-run.

## Critical issues before using these numbers

1. **Flip-S F4 collapse** — the ASR-gated acceptance must be replaced with the
   paper's loss-based acceptance (+ clean preservation), or every Flip-S row is a
   non-stealthy collapse.
2. **ProFlip runs are pre-clean-CE** — re-run with `--clean-ce-weight 1.0` (and
   the paper-faithful acceptance `--cbs-accept-always`) to remove the collapse.
3. **TrojViT Food101 collapse** (CACC ≈ 0) — investigate why the proj
   distillation destroys Food101 specifically.
4. **`trigger_asr_pre` units are inconsistent** — TrojViT stores a fraction,
   ProFlip a percentage. Normalize before reporting (table above already
   converts to %).
5. **Budget caveat (T1)**: TrojViT's `flips_to_98 = null` is budget-limited (200
   vs native 880–1250), not a method ceiling.

## Bottom line

As-is, the table's story is "no baseline reaches 98% stealthy; several collapse."
That is a *favourable* headline for the proposed method, **but** it is only
defensible once the collapse causes are fixed — otherwise a reviewer will say the
baselines were unfairly constrained (Flip-S's broken acceptance, ProFlip's missing
clean-CE). Fix #1–#2, re-run, then the comparison is clean: TrojViT = stealthy but
<98%, ProFlip/Flip-S = either <98% or (if they can reach 98% without collapsing)
a real `flips_to_98` number.

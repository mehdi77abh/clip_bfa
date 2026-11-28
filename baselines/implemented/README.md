# Corrected TrojViT → CLIP implementation

Self-contained corrected port of **TrojViT (CVPR 2023)** to CLIP zero-shot
victims. The original file
`proposed_method/modular_new_pipeline/bfa_baselines/attack_trojvit.py` is
**not modified**; this folder is an independent implementation.

```
implemented/
  attack_trojvit.py   # corrected TrojViT port
  evaluate.py          # shared evaluator (+ pre/post-flip metrics)
  clip_adapters.py     # model/dataset/quantisation plumbing (copy)
  data_split.json      # frozen splits (copy)
  class_texts/         # prompt class-name lists (copy)
```

All commands below are run from this folder.

---

## Quick start

```bash
python attack_trojvit.py \
    --model ViT-B/16 --dataset Caltech101 --seed 42 --target-cls 97 \
    --out results/trojvit_corrected/ViT-B16/Caltech101/s42 \
    --device cuda

python evaluate.py --state results/trojvit_corrected/ViT-B16/Caltech101/s42/attack_state.pt \
    --device cuda --eval-n 2000
```

The evaluator writes `eval.json` next to `attack_state.pt`. It reports:

* `asr`, `cacc`, `clean_as_target` for the **attacked** model,
* `pre_flip` for the **clean quantised victim** (same weights, no bit flips),
* `asr_gain` = post-flip ASR − pre-flip ASR.

`n_flips` is exactly `len(flip_log)`.

---

## What was wrong in the original port

| # | Problem in the original | Fix here |
|---|---|---|
| 1 | `realize_bit_flips` returned `dequantize_int8(codes, scale)` for the **whole** projection, so every entry changed while only a few bits were logged. | `final_proj` starts as the clean projection and **only flipped entries are written**. Verified: 0 of 393,213 non-flipped entries changed on a ViT-B/32 run. |
| 2 | Trigger/saliency CE used `cos / temperature` (logits ≈ [-1,1]); NGR/distillation/eval used `cos * logit_scale.exp()` (≈100). | All CE paths use the model's `logit_scale`; `--trojvit-temperature` is an extra divisor applied consistently. |
| 3 | `trigger_asr_pre` and the flip trace were measured on the calibration images used to train the trigger/distillation. | Primary trace and `trigger_asr_pre` use the disjoint `search` split; ASR is also reported on the held-out `eval` split. |
| 4 | State stored patch rows for the 1000-image `eval` split only; evaluator default is 2000 → `IndexError`. | Patch rows are stored for exactly `--eval-n` images (default 2000; `0` = all `eval+eval_extra`). `evaluate.py` fails fast with a clear message if the state has too few rows. |
| 5 | 80 epochs with milestones 80/120/160 → LR never decayed. | Default 200 epochs; `--p2-milestones auto` scales to 40/60/80% of the epoch count. |
| 6 | Only `proj` was quantised; the evaluator started from a float model and the override changed every entry. | `--victim-quant surface` (default) quantises the whole 2-D vision surface and saves `clean_param_overrides`; pre/post metrics use the **same quantised victim**. `proj` and `float` modes remain available. |

Also added: paper-faithful GradSurgery + iterative pruning, `true_label` NGR,
fixed universal masks, TAR-matched patch counts, CLS-query ATL, and gated flip
acceptance.

---

## Key options

### Victim quantisation

| `--victim-quant` | Meaning | State size |
|---|---|---|
| `surface` (default) | Every 2-D vision weight (attention + MLP + projection) is per-tensor int8. Matches ProFlip/Flip-S victim semantics. | Large (~all vision weights) |
| `proj` | Only `visual.proj` is int8 (the original port's scope). | Small |
| `float` | No quantisation; only flipped projection entries change. | Small |

`clean_param_overrides` is saved for all modes, so `evaluate.py` can report
clean-victim (`pre_flip`) metrics.

### Trigger

| Option | Default | Notes |
|---|---|---|
| `--trojvit-patch-select` | `saliency` | `saliency`, `attn`, `rand` |
| `--trojvit-num-patch` | `9` | Paper default |
| `--target-tar` | `0` | >0 derives `num_patch` from a target area ratio, e.g. `4.59` on patch-16@224 |
| `--patch-placement` | `per_image` | `per_image` = TrojViT Eq. 3/reference; `fixed` = one universal mask (mean saliency/attention over calib) |
| `--trojvit-mode` | `replace` | Reference uses replace; paper writes `X + P⊙M` (`add`) |
| `--trojvit-loss` | `ce` | `ce` = zero-shot CE (needs class list); `cos` = `1-cos` (class-list-free) |
| `--trojvit-atten-weight` | `1.0` | λ in `L_target + λ Σ_l L_ATTN` |
| `--attn-query` | `all` | `all` = paper Eq. 4; `cls` = CLIP pooled token |
| `--trojvit-images` | `0` | 0 = all calib images |
| `--trojvit-iters` | `250` | Paper default |

### Insertion

| Option | Default | Notes |
|---|---|---|
| `--ngr-loss` | `true_label` | Reference behaviour; `target` reproduces the old port |
| `--p2-rows` / `--p2-entries` | `1` / `0` | One embedding column, all width entries |
| `--p2-epochs` | `200` | Paper default |
| `--p2-milestones` | `auto` | 40/60/80% of `p2-epochs` |
| `--p2-grad-surgery` | on | Paper Algorithm 1 step 8; `--no-p2-grad-surgery` = reference code |
| `--p2-iterative-prune` | on | Paper Algorithm 1 steps 11-13; `--no-p2-iterative-prune` = prune only at the end |
| `--flip-accept` | `paper` | `paper` applies entries until budget; `gated` rejects entries that lower ASR/CACC |
| `--max-flips` | `200` | Suite budget; paper native TBN 880–1250, use e.g. `1000` for that |

### Evaluation

| Option | Default | Notes |
|---|---|---|
| `--eval-n` | `2000` | 1000 `eval` + 1000 `eval_extra`; must be ≤ rows stored in the state. `0` = all available. |
| `--quick-split` | `search` | Held-out trace; `calib` = old in-sample behaviour |
| `--quick-n` | `256` | Capped by the chosen split |
| `evaluate.py --eval-n` | `2000` | `0` = frozen 1000-image `eval` split; up to 5000 with `eval_extra` |

---

## What is still a documented deviation

These are adaptations, not bugs. They are called out so the result is not
misrepresented as “TrojViT exactly as published”.

1. **Only `visual.proj` is tuned.** The paper mentions the last attention and
   classification layers; the reference code effectively tunes the head only.
   For CLIP the head analogue is the projection.
2. **No class-row isolation.** CLIP has no classifier row; changing a
   projection column affects all class logits. This is why clean accuracy can
   drop on fine-grained datasets. `--flip-accept gated` and a stronger clean
   term (`--p2-clean-weight`) are the available mitigations.
3. **Per-image adaptive trigger.** `--patch-placement per_image` follows
   TrojViT Eq. 3 and needs white-box saliency at evaluation time. If the
   proposed method uses a fixed trigger, run `--patch-placement fixed` for the
   main comparison and report the per-image version as an upper bound.
4. **TAR is not automatically comparable across patch sizes.** `num_patch=9`
   is 4.59% on ViT-B/16@224, 3.52% on ViT-L/14@224, 1.56% on ViT-L/14@336,
   and 18.4% on ViT-B/32. Use `--target-tar 4.59` (or report TAR) when
   comparing models with different patch sizes.
5. **`ce` needs the class list.** The class-list-free variant is
   `--trojvit-loss cos`. TrojViT's threat model has the target class, so `ce`
   is a legitimate stronger baseline, but state the asymmetry.
6. **ASR excludes target-class images** (`evaluate.py` suite convention). The
   paper includes them; `asr_incl_target` is also reported.
7. **The budget is a suite choice.** `--max-flips 200` is the push-98 budget;
   TrojViT's native TBN is much larger. Report ASR-vs-flips curves and at
   least one native-budget run.

---

## Verification performed

* `CaptureMHA` matches `nn.MultiheadAttention` to ~6e-6 on ViT-B/32.
* PCGrad matches paper Eq. 6 exactly.
* Attention loss gathers the selected patch keys; all selected patches are
  used (the reference only targets the first).
* Saliency (`|∇L|` per patch), attention, and random selection all work in
  `per_image` and `fixed` modes.
* Exact flip accounting: after a 6-bit/3-entry run, **0 of 393,213
  non-flipped projection entries changed**; the only differences are the
  flipped entries.
* End-to-end smoke runs on ViT-B/32 (CPU) with `surface`, `proj`, and `float`
  victim quantisation, and with `per_image` and `fixed` placement. The
  evaluator reports `pre_flip` and post-flip metrics.

Example smoke command (CPU, tiny; not a real result):

```bash
python attack_trojvit.py --model ViT-B/32 --dataset Caltech101 \
    --seed 42 --target-cls 97 --out /tmp/trojvit_smoke --device cpu \
    --eval-n 8 --trojvit-images 8 --trojvit-batch 4 --trojvit-iters 1 \
    --trojvit-patch-select rand --trojvit-atten-weight 0 \
    --trojvit-num-patch 2 --patch-placement fixed --victim-quant proj \
    --p2-epochs 1 --p2-batch 4 --quick-n 8 --max-flips 8 --batch 8

python evaluate.py --state /tmp/trojvit_smoke/attack_state.pt \
    --device cpu --batch 8 --eval-n 8
```

For a real run use the paper defaults (`--trojvit-iters 250`,
`--p2-epochs 200`, all calib, saliency + ATL), and choose the budget /
placement / victim quantisation required by the comparison.

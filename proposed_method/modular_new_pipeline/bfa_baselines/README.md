# CLIP baseline suite — TrojViT · ProFlip · Flip-S

Self-contained zero-shot CLIP comparison suite for the hybrid-BFA thesis.
**Imports nothing from `modular_new_pipeline` proper** — model/data/quantisation
plumbing is re-implemented locally (`clip_adapters.py`), so the folder can be
copied to another machine as-is (plus the `data/` tree and the `class_texts/`
copies shipped here).

## Files

```
bfa_baselines/
├── attack_trojvit.py   # TrojViT: Phase-1 attention-target trigger + Phase-2
│                       #   projection-row distillation (NGR tar/tar1 on
│                       #   visual.proj) + bit-flip realization of tuned entries
├── attack_proflip.py   # ProFlip: SNI (|proj @ t_target|) + native feature-
│                       #   anchored trigger (MSE(h,y[I_t])+CE, ≈31% bottom-right
│                       #   patch) + CBS S1/S2/S3, --fitness {mse_sni,cos}
├── attack_flips.py     # Flip-S: full rewrite — Stage-A trigger, SFS ranking,
│                       #   int8 weight bits + FP16 row-wise SCALE-factor bits,
│                       #   4KB mutual exclusion, 50-bit budget
├── evaluate.py         # SHARED ASR/CACC evaluator — the only place metrics
│                       #   are computed (defines the attack_state.pt contract)
├── run_sweep.py        # grid driver: methods x models x datasets x seeds
├── clip_adapters.py    # local plumbing: OpenAI CLIP loading, deterministic
│                       #   dataset loading, prompts, int8/fp16 quantisation
├── data_split.json     # FROZEN disjoint splits (eval 1000 / calib 384 /
│                       #   search 256 / aux 32) — the fairness guarantee
├── class_texts/        # frozen class-name lists (copies of the pipeline's)
└── results/            # everything lands here
    ├── {method}/{model_tag}/{dataset}/s{seed}[_{fitness}]/
    │     attack_state.pt  attack_meta.json  eval.json  run_*.log
    └── summary/{runs.csv, summary.csv}
```

## Quick start

```bash
cd proposed_method/modular_new_pipeline/bfa_baselines
python run_sweep.py                          # full grid, resume-aware
python run_sweep.py --dry-run                # show commands only
python run_sweep.py --methods flips --models ViT-B/16 \
    --datasets Caltech101 --seeds 42         # one cell
python evaluate.py --state results/.../attack_state.pt   # re-evaluate a run
```

Default grid: `{trojvit, proflip, flips}` × `{ViT-B/16, ViT-L/14,
SigLIP-B/16}` × `{ImageNet1k, Caltech101, Food101}` × seeds `{42}`.
ProFlip runs twice per cell (`--fitness mse_sni` and `--fitness cos`).
Default target classes: ImageNet1k=291 (lion), Caltech101=97 (wild cat),
Food101=76 (pizza) — matched to the proposed method's runs (`run_all.sh`).

Run with the `vlm_attack` conda env (needs `clip` for OpenAI ViT +
`transformers`/`sentencepiece` for SigLIP); `run_sweep.py` launches each
attack/eval as a subprocess with `sys.executable`, so activate the env first.

## Victim backends and datasets

Two victim backends (`clip_adapters.MODEL_REGISTRY[name]["backend"]`):

| Backend | Models | Projection surface | Normalisation | Attention maps |
|---|---|---|---|---|
| `clip` | ViT-B/16, ViT-L/14, ViT-L/14-336px, ViT-B/32 | `visual.proj` `[width, embed]`, `emb = h @ proj` (column = embedding dim) | CLIP mean/std | CaptureMHA on `nn.MultiheadAttention` |
| `siglip` | SigLIP-B/16 (`google/siglip-base-patch16-224`) | `vision_model.head.mlp.fc2.weight` `[embed, width]` (ROW = embedding dim) | 0.5/0.5 | eager-attn forward hooks on `self_attn` |

Datasets (deterministic label ordering, frozen in `data_split.json`):
Caltech101 (8677), ImageNet1k val (50 000), DTD (5640), and **Food101**
(25 250 = the manually-reviewed test split, 250/class; class order = sorted
`food-101/images` dirs = `class_texts/food101_classes.txt`). Food101 data is
**not bundled** — download it first, e.g.
`torchvision.datasets.Food101(root=<data>, split="test", download=True)`
(layout `data/food-101/{meta,images}`); `clip_adapters` raises a clear error
if it is absent. SigLIP weights load from the HF cache offline (unset the
`socks://` proxy env — see `clip_adapters._load_siglip`).

## Contracts

- **One code per method** (`attack_*.py`): writes `attack_state.pt` (final
  `param_overrides` + trigger dict + `flip_log`) and `attack_meta.json`
  (`n_flips, budget, search_time_s, trigger_asr_pre, pool_size,
  flips_to_94/95`). Exact schema: see `evaluate.py` docstring.
- **`evaluate.py` is the single source of metrics.** ASR = P(argmax ==
  target_cls | triggered) over non-target eval images; CACC = P(argmax ==
  label | clean). In-search quick evaluations call the same `eval_model`.
- **`data_split.json` is frozen**: every method and every server evaluates the
  identical images (eval 1000) and draws calibration/search/aux images from
  identical pools. Regeneration recipe is in the file's `_meta`.

## Method adaptations (labelled, for the paper)

| Method | Kept native | Adapted for CLIP |
|---|---|---|
| TrojViT | Phase-1 salience/attention patch selection + Attention-Target loss (PCGrad); Phase-2 NGR topk pattern, mixed clean/triggered distillation (200 ep, SGD 0.01, MultiStepLR 80/120/160), per-step restore of non-selected entries, convergence pruning | Classifier head → `visual.proj`: row score `\|t_target[i]\|·‖g[:,i]‖`, tuned entries realised as int8 bit flips (Hamming to the tuned value) so flips are counted |
| ProFlip | SNI significant-neuron criterion; feature-anchored trigger `MSE(h, y[I_t]=c) + CE` with the native ≈31%-width bottom-right patch and 10-step GD; CBS S1/S2/S3 fitness `\|∂L/∂w\|·(Q−w)`, K=20 grid, multi-bit element updates, ASR_t=94 / n_max=100 | JSMA SNI → analytic `I_t = top-\|proj @ t_target\|`; literal clamp `10` → scale-adaptive `c` (`--clamp-mode`); fitness loss selectable (`mse_sni` native / `cos`) — both rows reported |
| Flip-S | Whole method: SFS scale-factor ranking, bnb-style int8 + FP16 row-wise scales, 9-patch (3×victim-patch) trigger block, 4KB RowHammer mutual exclusion, 50-bit budget | logits via the zero-shot head; bitandbytes replaced by deterministic local quantisation with identical FP16-scale semantics |

## Fairness statement

Each baseline uses its native trigger-generation and bit-selection procedure.
The shared components are the victim models, the frozen data splits, the
zero-shot evaluation protocol (ASR/CACC in `evaluate.py`), and the reported
metrics. TBT/ProFlip classifier-layer analogues map to `visual.proj`; TrojViT's
classifier-row distillation maps to target-relevant `visual.proj` rows; Flip-S
operates on its native FP16 row-wise scale factors. Native budgets are kept
(TrojViT 80, ProFlip 100, Flip-S 50) and reported.

## Stop criteria (bit-flip budgets) — **push-98 configuration**

The results table runs the **push-98** configuration: every baseline is asked
for +98% ASR and the reported number is how many bit flips it needed to get
there. Native stopping values are documented but not run.

| Method | Stop condition (push-98) | Native values (documented, not run) |
|---|---|---|
| TrojViT | budget 200 (`--max-flips`); tuned entries applied **atomically** (whole entry or none) in descending \|Δw\| order; quick-ASR trace recorded after every applied entry → `flips_to_98` + `asr_at_k` | budget 80 |
| ProFlip | quick ASR ≥ **98** (`--asr-target`) **or** bits ≥ **200** (`--n-max`) **or** CBS pool exhausted | ASR_t=94 / n_max=100 (paper) |
| Flip-S | budget **200** (`--max-flips`) **or** no improving candidate remains; 4KB mutual exclusion | budget 50 (`target_bit`) |

**`flips_to_X`** = cumulative bit count at which the in-search quick ASR first
crossed X% (`null` = never reached — an honest "not reached" result).
**`asr_at_k`** (10/20/40/60/80/100/150/200) = the ASR@10/@20/... curve.

**ASR@K convention (suite-wide):** ASR@K is measured on the **search-time
quick-ASR trace** (first `--quick-n` = 256 images of the search split; for
TrojViT the calib split), not by re-evaluating the eval split. Multi-bit
atomic element updates (ProFlip, TrojViT) make the curve a step function:
ASR@K = quick ASR at the last state with cumulative flips ≤ K. The reported
final `asr`/`cacc` always come from the shared `evaluate.py` on the frozen
eval split.

## Labelled deviations from the released baseline code

Verified line-by-line against `TrojViT/main_patch_vit.py`,
`ProFlip/resnet32-cifar10/CSB_nonadaptive.py`, and
`baselines/Flip-S/target-imagenet/deit-targeted-int8.py`. All phases,
hyper-parameters, and update mechanics match except the following, which are
**deliberate** (paper values chosen over release hard-codes, or CLIP-fairness
choices):

| # | Where | Release code | This suite | Why |
|---|---|---|---|---|
| 1 | ProFlip stop | `while n_b<50` hard-coded | push-98: `asr-target=98`, `n_max=200` (paper `ASR_t=94`/`n_max=100` documented, not run) | push-98 comparison (agreed) |
| 2 | ProFlip S3 grid | 10 descending points `[R−i·2R/10]`, monotone filter commented out | `linspace(−R,R,K=20)`, monotone `new≥old` (`--no-monotone` opts out) | `proflip_cbs.py` convention; `--grid-k 10` available |
| 3 | Flip-S acceptance | commit iff aux loss `(MSE+γCE)/(1+γ)` decreases | commit iff quick ASR strictly improves | quick ASR is the task metric; ranking/filter still use the release loss |
| 4 | Flip-S ranking | rank candidates by `\|grad\|` | rank by `\|grad·Δbit\|` | the paper's SFS formula (hypothesis = paper) |
| 5 | Flip-S 4KB ban | aligned 4KB block containing the flip | ±4KB window around the flip | same RowHammer principle, simpler geometry |
| 6 | ProFlip S2/S3 loss | adds `+2·CE(clean, true labels)` | no clean term in `mse_sni` | clean accuracy is measured by the shared evaluator instead |
| 7 | TrojViT distill loop | 200 steps on ONE fixed batch of the TEST set | 200 epochs over the frozen calib split (mini-batches) | never train on test data (fairness); same optimizer/schedule |
| 8 | TrojViT distill loss | `0.5·CE(clean)+0.5·CE(trig→target)` | `0.5·CE(clean)+0.5·(1−cos(emb_trig, t_target))` | the accepted CLIP zero-shot equivalent form |
| 9 | TrojViT pruning | convergence pruning (`optimized_wb1`) disabled in code | prune `\|Δw\|<0.003` on by default | the paper describes pruning; kept as designed |
| 10 | TrojViT tuned scope | `topk(768)` = full classifier row (~768 params) | top-8 target-relevant proj rows × all entries | CLIP has no class row; "NGR entries in top rows" adaptation |
| 11 | SigLIP projection surface | (release code is CLIP-ViT-only) | SigLIP has no `visual.proj`; the joint-space projection surface is `vision_model.head.mlp.fc2.weight` (`[embed, width]`, ROW = embedding dim — the reverse orientation of CLIP's `proj`, handled by `ca.proj_embed_axis`) | the closest single linear map into the joint space; head-internal attention/fc1 kept out of the surface. The head is attention-pooling + residual, so `emb ≈ h @ proj` is approximate — used for SNI/NGR/distillation only, never for eval logits (those use the true `pooler_output`) |
| 12 | SigLIP attention maps | SDPA returns no weights; `SiglipEncoderLayer` discards `attn_weights` | load with `attn_implementation="eager"` and capture `attn_weights` per `self_attn` via forward hooks | the only way to get per-head maps in transformers 5.x; both attack and `evaluate.py` load here, so the victim is numerically self-consistent |
| 13 | SigLIP text prompts | single prompt is common for SigLIP | the suite's 13-template prompt ensemble (mean of normalised template embeddings) | cross-model consistency with the CLIP victims (shared `build_text_embeddings`); SigLIP's `logit_bias` omitted from logits (class-independent → argmax-equivalent) |

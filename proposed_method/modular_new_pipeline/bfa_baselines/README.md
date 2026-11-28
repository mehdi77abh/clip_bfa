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
├── ADAPTATIONS.md      # short: how each baseline was re-bound to CLIP/SigLIP
├── missmatch_baselines.md  # detailed paper vs release vs ours mismatch report
├── evaluate.py         # SHARED ASR/CACC evaluator — the only place metrics
│                       #   are computed; writes eval.json (attack_state.pt is
│                       #   transient — the drivers delete it after evaluating)
├── run_sweep.py        # grid driver: methods x models x datasets x seeds
├── clip_adapters.py    # local plumbing: OpenAI CLIP loading, deterministic
│                       #   dataset loading, prompts, int8/fp16 quantisation
├── data_split.json     # FROZEN disjoint splits (eval 1000 / calib 256 /
│                       #   search 256 / aux 32) + eval_extra 4000 for the
│                       #   larger final evaluation (default 2000, up to 5000)
├── class_texts/        # frozen class-name lists (copies of the pipeline's)
└── results/            # everything lands here
    ├── {method}/{model_tag}/{dataset}/s{seed}[_{fitness}]/
    │     attack_meta.json  eval.json        # attack_state.pt is transient
    ├── runs.csv  summary.csv                # one row per run
    └── logs/{timestamp}/  logs/done/        # per-run logs + resume keys
```

## Quick start

```bash
cd proposed_method/modular_new_pipeline/bfa_baselines
python run_sweep.py                          # full grid, resume-aware
python run_sweep.py --dry-run                # show commands only
python run_sweep.py --methods flips --models ViT-B/16 \
    --datasets Caltech101 --seeds 42         # one cell
# re-evaluate only while the state exists (standalone attack run); the grid
# drivers delete attack_state.pt after writing eval.json
python evaluate.py --state results/.../attack_state.pt
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

- **One code per method** (`attack_*.py`): writes `attack_state.pt` (transient:
  final `param_overrides` + trigger dict + `flip_log`) and `attack_meta.json`
  (`n_flips, budget, search_time_s, trigger_asr_pre, pool_size,
  flips_to_94/95`).  The drivers run `evaluate.py` on the state, write
  `eval.json`, then delete `attack_state.pt` — the saved result is
  `attack_meta.json` + `eval.json`.  Exact schema: see `evaluate.py`.
- **`evaluate.py` is the single source of metrics.** ASR = P(argmax ==
  target_cls | triggered) over non-target eval images; CACC = P(argmax ==
  label | clean). In-search quick evaluations call the same `eval_model`.
- **`--save-model-output`** (currently set on the Flip-S ViT-L/14 × ImageNet1k
  run): additionally writes `trigger.pt` (the generated trigger dict) and
  `model_state_attacked.pt` (full attacked model, CPU fp32) into the run dir.
  `attack_state.pt` remains transient (deleted after `eval.json` is written).
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
| TrojViT | budget 100 (`--max-flips`); tuned entries applied **atomically** (whole entry or none) in descending \|Δw\| order; quick-ASR trace recorded after every applied entry → `flips_to_98` + `asr_at_k` | **TBN 880 (DeiT-s) / 1250 (ViT-b)** — the paper's native budget. The old "native 80" was wrong (a units error; the paper's TPN=84 at e=0.003 has TBN=345, and TPN=213 at e=0.0005 has TBN=880) |
| ProFlip | quick ASR ≥ **98** (`--asr-target`) **or** bits ≥ **100** (`--n-max`) **or** CBS pool exhausted | ASR_t=94 / n_max=100 (paper) |
| Flip-S | fixed budget **50** (`--max-flips 50`, the release's `target_bit=50`) **or** no improving candidate remains; ASR is reported after the budget, **not** a stop (`--asr-target 0` in the grid); re-rank every `--rerank-every` flips (grid: 5 for speed, paper: 1); 4KB mutual exclusion. The optional stealth gates are off. | budget 50 (`target_bit`); relaxed to 500 in the paper's extended eval; push-98 alternative: `--max-flips 100 --asr-target 98` |

⚠️ **TrojViT budget caveat (T1):** the push-98 100-bit cap is **~9–12× below**
TrojViT's native TBN (880–1250). Under 100 bits, `flips_to_98 = null` is
**budget-limited, not TrojViT's ceiling** — do not present it as the method's
max ASR. Either also run a native-budget row (`--max-flips 1250`) or state the
table uses a common 100-bit cap below TrojViT's native TBN.

**`flips_to_X`** = cumulative bit count at which the in-search quick ASR first
crossed X% (`null` = never reached — an honest "not reached" result).
**`asr_at_k`** (10/20/40/60/80/100/150/200) = the ASR@10/@20/... curve.

**ASR@K convention (suite-wide):** ASR@K is measured on the **search-time
quick-ASR trace** (first `--quick-n` images of the search split for
ProFlip/Flip-S, the calib split for TrojViT), not by re-evaluating the eval
split. Current defaults: TrojViT `--quick-n 256` (calib), ProFlip/Flip-S
256 (search split). Multi-bit
atomic element updates (ProFlip, TrojViT) make the curve a step function:
ASR@K = quick ASR at the last state with cumulative flips ≤ K. The reported
final `asr`/`cacc` always come from the shared `evaluate.py` on the frozen
eval split.

**Final eval size:** the final `evaluate.py` pass uses **2000** images by
default (1000 frozen eval + 1000 disjoint `eval_extra`; no overlap with
calib/search/aux). `EVAL_N=5000` uses the full 1000 + 4000 extras; `EVAL_N=0`
(or 1000) restores the frozen 1000-image split. The in-search "quick" number
is the 256-image search subset.

## Paper-faithful presets + reporting conventions

Each baseline has a paper-faithful mode (use these for the primary table; the
suite defaults are labelled variants):

**ProFlip** (`--sni-method jsma --cbs-select-once --cbs-accept-always
--cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs --clean-ce-weight 1.0`):
JSMA SNI (paper), pins S1 to one layer, S2 takes one element, S3 applies the
unconditional argmin (no ASR escape), grid K=20, `max|W|` fitness, and the
clean-preservation CE on (mismatch #6 — without it the model collapses).
Trigger TAP is 73² (paper's ImageNet value). `mse_sni` is the paper's `L_CBS`;
`ce` is a CE-only variant.

**TrojViT** (`--trojvit-loss ce --trojvit-temperature 1.0 --p2-rows 1
--p2-prune-e 0.0005`): CE trigger + distillation (cosine removed), scope ≈ the
paper's 768-weight target row (8 rows = 6144/8192 over-fits), paper prune
`e=0.0005`. Note the scope is the target-relevant `visual.proj` rows (CLIP has
no class row) and ours tunes proj only (the paper also tunes the last-attention
block — documented deviation).

**TrojViT runtime preset (2026-11-28, ~50% faster):** trigger **150 iters**
(paper 250) over **128 calib images** (paper 384), `--trojvit-batch 32`,
distillation **80 epochs** (paper 200; the loss plateaus by ~60–80) with
`--p2-batch 64`, and `--quick-n 256` for the realization trace. Phase 1
(per-image PCGrad for `L_ATL`) dominates the runtime; `--trojvit-atten-weight 0`
is the further ~8–10× lever but drops the paper's attention term (screening
only).

**Flip-S** (`--flip-surface encoder --trig-side 48`): encoder-only surface
(never the head/projection — mismatch F1), and the native 50-bit budget.

**Flip-S runtime + correctness (2026-11-28):** trigger generation fixed —
the attention rollout now iterates **layers** (the release/older port indexed
the batch axis, multiplying maps of different images), the attention target is
in probability units (`--atten-target 1.0`, not the release's literal `100`),
and the CE uses the inference logits (`logit_scale.exp()`). The primary grid
uses the **paper loss** (`--clean-ce-weight 0`) and the release's **fixed
50-bit budget** (`--max-flips 50 --asr-target 0`; ASR is reported after the
budget, not used to stop). `--rerank-every 5` is a documented speed
approximation (the paper re-ranks every flip — use 1 for a fully faithful
cell). The `--clean-ce-weight λ`, `--min-flips/--min-cacc/--max-clean-as-target`
guards are optional ablations, **off** in the primary grid.

**Reporting conventions (cross-cutting):**
- **TAR + trigger-only ASR** are recorded in every `attack_meta.json` (`tar`,
  `trigger_asr_pre`) and aggregated into `runs.csv`/`summary.csv`. Report both
  beside `flips_to_98` — the baselines use 3.5–10.6% trigger area vs the
  proposed method's 0.51%, a large confound (mismatch 1.1).
- **ASR definition** (mismatch #4/P14): `eval.json` reports both `asr`
  (excludes clean-target images) and `asr_incl_target` (the papers' definition —
  counts all triggered images). Use `asr_incl_target` for the paper-comparable
  number; report both.
- **SNI** (mismatch #5/P8): two options — `--sni-method analytic` (default) =
  the closed-form `top-|visual.proj @ t_target|` (the exact per-dim target
  influence), and `--sni-method jsma` = the paper's per-image gradient saliency
  + intersection (`--sni-theta 0.1 --sni-gamma 0.5`). Use `jsma` for the
  paper-faithful row; `analytic` is a defensible CLIP-native substitute.

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
| 3 | Flip-S acceptance | commit iff aux loss `(MSE+γCE)/(1+γ)` decreases | same loss-based rule (`--clean-ce-weight 0` in the primary grid); optional λ>0 and CACC stop gates are ablations | paper Eq. 7; the collapse observed with a dead trigger (trigger fixed 2026-11-28) |
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

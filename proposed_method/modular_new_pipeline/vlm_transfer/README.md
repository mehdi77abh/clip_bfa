# `vlm_transfer/` — does the CLIP-BFA backdoor survive inside a VLM?

CHECKLIST.md §3.5. Take the **attacked CLIP visual encoder** produced by a
CLIP-BFA run, use it *instead of* the VLM's own CLIP vision tower, and
measure whether the VLM's generated answers/captions are pushed toward the
attacker's target. Evaluation protocol and hyperparameters follow
**X-Transfer** (Huang, Erfani, Li, Ma, Bailey, *X-Transfer Attacks: Towards
Super Transferable Adversarial Attacks on CLIP*, arXiv:2505.05528v3) — see
§2.

```
   CLIP-BFA attack ──► model_state_attacked.pt ──► tower swap ──► VLM
     (run_attack.py)                              (tower_swap.py)  (vlm_adapters.py)
                                                                     │
                                          OK-VQA / Flickr-30K ───────┤
                                                                     ▼
                                                        targeted metrics
                                                           (metrics.py)
```

---

## 1. Protocol — the 4-cell matrix

Per **(VLM × dataset)**, evaluate all four cells:

|  | clean input | triggered input |
|---|---|---|
| **default tower** | **A** baseline | **B** control (trigger alone) |
| **attacked tower** | **C** control (no trigger) | **D** ***the result*** |

**D** is the only cell where the backdoor should fire. A/B/C are not
optional: without them a high target rate in D is indistinguishable from a
broken VLM, or from the trigger being sufficient on an unmodified tower.

**Metrics are targeted only** (maintainer decision 2026-11-21):

| metric | task | definition |
|---|---|---|
| `target_mention_rate` | VQA + captioning | fraction of generations mentioning the target concept |
| `bleu4_vs_target_desc` | captioning only | BLEU-4(generation, target text description) — X-Transfer's targeted-ASR proxy |

X-Transfer omits BLEU-4 for VQA ("due to the brevity of the answers") and
so do we; the field is present but `null`.

X-Transfer's **non-targeted** degradation ASR is deliberately **not**
computed here — CLIP-BFA is a *targeted* backdoor, and the thesis claim is
about steering, not degradation. The formula is recorded in §2 so a
comparison table can be produced later if wanted.

---

## 2. X-Transfer parameters used here

Source: `…/06.Bit-Flip Attack Clip/X-Transfer Attacks.pdf`, §4 and
Appendix C.1 / C.7.

**Table 6 — the evaluated VLMs** (verbatim):

| Model | Image Encoder | LLM | Image Resolution |
|---|---|---|---|
| OpenFlamingo-3B (OF-3B) | ViT-L/14 CLIP OpenAI | MPT-1B | 224 × 224 |
| LLaVA-7B | ViT-L/14 CLIP OpenAI | Vicuna-7B | 224 × 224 |
| MiniGPT4-v2 | ViT-G/14 EVA-CLIP | Llama2 Chat 7B | 448 × 448 |
| BLIP-2 | ViT-G/14 EVA-CLIP | OPT-6.7B | 364 × 364 |

We evaluate **OF-3B, LLaVA-7B and BLIP-2** (maintainer 2026-11-21).
MiniGPT4-v2 is not in scope.

**Evaluation:**

- Captioning metric: **CIDEr** (only needed for the *non-targeted* ASR we
  do not report; our targeted metrics are BLEU-4 + mention rate).
- VQA metric: **VQA accuracy** (ditto).
- **Non-targeted ASR = (s_clean − s_adv) / s_clean**, where *s* is the
  task metric (X-Transfer §4, "Evaluation").
- **Targeted ASR = BLEU-4** between the response and the adversary's
  target text description (Appendix C.7). Reported for captioning only.
- **10 target text descriptions** (Nos. 1–6 from Schlarmann & Hein 2023 /
  Schlarmann et al. 2024, 4 authored by the X-Transfer authors). The paper
  never enumerates the strings — so we instead use our attack's **target
  class** as the target description (see §3).
- "We apply the same UAP to **every image in each dataset**" / "we apply
  the TUAP to **each image in the evaluation dataset**".
- The **only** explicit sample count in the paper is for retrieval:
  *"the MSCOCO dataset, which contains 3,900 images"*. There is no
  per-dataset N for OK-VQA / Flickr-30K.

**Evaluation set sizes used here** = the full standard splits, which is
what "each image in the evaluation dataset" means:

| dataset | task | split | size |
|---|---|---|---|
| OK-VQA | VQA | test | **5 046 questions** (COCO val2014 images) |
| Flickr-30K | captioning | test | **1 000 images / 5 000 captions** (Karpathy) |

`--max-samples N` subsamples deterministically (seeded) for smoke tests;
the default is 0 = full.

**Resolution handling** (Appendix C.1): *"For large VLMs that use
different image resolutions than our default 224 × 224, we use
interpolation to rescale the perturbation to the resolution used by the
VLM."* Our trigger is a `TRIG_SZ`-pixel patch at the attack's `IMG_SIZE`,
so `evaluate.py` rescales it by `vlm_image_size / attack_img_size`
(16px @224 → 24px @336 → 26px @364) before applying it through the repo's
own `trigger.apply_trigger` — same mode, opacity, placement and clamping
semantics as the attack itself.

> X-Transfer's UAP *generation* settings (L∞ ε = 12/255, step 0.5/255,
> 224×224, 1 epoch over ImageNet, batch 1024, k = 4/8/16 for the
> Base/Mid/Large search spaces) are **not** applicable — our attack is a
> weight bit-flip backdoor with its own Phase-1 trigger optimisation. They
> are recorded here only for completeness.

---

## 3. Maintainer decisions (2026-11-21)

1. **BLIP-2 → EVA-CLIP ViT-G/14 adapter.** Attack BLIP-2's *actual* tower
   so the swap is 1:1 (not a surrogate-transfer experiment).
2. **LLaVA-7B → both towers, configurable.** `LLaVA-7B` (the paper's
   224 tower) **and** `LLaVA-7B@336` (the official
   `openai/clip-vit-large-patch14-336`) are registered and attacked
   separately.
3. **Attack config = ImageNet1k, target class 291 'lion'.** The validated
   reference configuration.
4. **Metrics = targeted only.** No degradation ASR, no CIDEr/VQA-accuracy
   requirement.
5. **Evaluation sets = full standard splits.**

---

## 4. Two discrepancies with the paper, and how they are handled

| # | Discrepancy | Handling |
|---|---|---|
| 1 | Table 6 lists **LLaVA-7B at 224×224** and says it uses "the same image encoder as OF-3B"; the released `llava-v1.5-7b` is configured with `clip-vit-large-patch14-**336**`. Different weight sets. | Both registered (`LLaVA-7B`, `LLaVA-7B@336`); the attack is run per tower. |
| 2 | Table 6 lists **BLIP-2 at 364×364**; the BLIP-2 checkpoints ship `Blip2VisionConfig(image_size=224)` (257 position embeddings). | `tower_native_image_size` is recorded per VLM; `tower_swap.fit_tower_pos_embed` interpolates position embeddings whenever `image_size` differs from the native grid, so 364 works. Override with `--vlm-image-size 224` if 364 turns out to be a typo. |

---

## 5. Files

| file | what it owns |
|---|---|
| `vlm_registry.py` | VLM specs (Table 6 + attack-model mapping), prompts, target-description / mention-term resolution from `class_texts/` |
| `tower_swap.py` | state-dict renaming (OpenAI/OpenCLIP ViT → HF ViT and → OpenCLIP ViT), fused q/k/v splitting, positional-embedding interpolation, tower discovery, the audited swap |
| `vlm_adapters.py` | `VLMAdapter` ABC + `OpenFlamingoAdapter` / `LLaVAAdapter` / `BLIP2Adapter`; the single shared image→tensor path |
| `datasets_vlm.py` | OK-VQA (VQA) and Flickr-30K (captioning) loaders, full-split defaults |
| `../download_manager/download_datasets.py` | OK-VQA + COCO val2014 + Flickr-30K downloaders (plus every registry dataset) — `--dataset ok-vqa flickr30k`, `--check`, `--dry-run` |
| `../download_manager/download_vlm_models.py` | VLM checkpoints — HF snapshots, OF-3B's `checkpoint.pt`, the EVA-CLIP source checkout, CLIP attack encoders; `.env`/`hf auth login` token check, `--list` / `--check` / `--dry-run` |
| `metrics.py` | BLEU-4, target-mention rate (self-contained, no `pycocoevalcap`) |
| `run_attack.py` | run CLIP-BFA on the VLM's own tower, force `--save-attacked-model`, write `vlm_transfer_meta.json` |
| `evaluate.py` | the 4-cell protocol → `vlm_transfer.json` |
| `run_vlm_transfer.sh` | orchestrates attacks (deduplicated) → evaluations → summary table |
| `test_vlm_transfer.py` | mocked tests (no GPU / network / VLM packages needed) |

**Two files outside `vlm_transfer/` were touched**, both the established
extension points for adding an attack model:

| file | change |
|---|---|
| `../config.json` | registry entry `EVA-CLIP-ViT-G/14` (`adapter: "eva_clip"`) |
| `../adapters.py` | `EVACLIPAdapter` + `load_eva_clip` + the `eva_clip` backend of `load_model`, wired into `get_adapter` / `adapter_for_block` |

The EVA-CLIP adapter lives in `adapters.py`, **not** here, on purpose: it
is CLIP-BFA infrastructure (it must work from `experiment.py` and
`run_all.sh`, not only from this folder), and AGENTS.md requires every
class to exist in exactly one module.

---

## 6. Prerequisites

**Python packages.** The attack side needs the existing pipeline deps
(`torch`, `torchvision`, `clip`, `Pillow`, `numpy`). The evaluation side
additionally needs:

```bash
bash download_manager/setup_env.sh                  # installs all of the below
# or by hand (same directory):
pip install -r download_manager/requirements-vlm-transfer.txt
                                                # einops, einops-exts,
                                                # open-clip-torch, timm
pip install "transformers>=4.36" accelerate sentencepiece protobuf
pip install open_flamingo --no-deps             # OF-3B; its metadata
                                                # hard-pins torch==2.0.1
```

**EVA-CLIP is not pip-installable.** `eva-clip` is not on PyPI and the
upstream repo ships no `setup.py`, so `pip install eva-clip` does not
work. `download_manager/setup_env.sh` clones `baaivision/EVA` and puts
`EVA-CLIP/rei` on
`sys.path` via a site-packages `.pth` file — that is what makes
`import eva_clip` work for BLIP-2. Skip with `--no-eva-clip`; the
`open_clip` fallback then takes over.

**Datasets.**

```bash
python download_manager/download_datasets.py --dataset ok-vqa --dry-run   # plan + what is missing
python download_manager/download_datasets.py --dataset ok-vqa --no-images # JSONs only
python download_manager/download_datasets.py --dataset ok-vqa             # + COCO val2014 (~13 GB)
python download_manager/download_datasets.py --dataset flickr30k
```

Landing place: `proposed_method/../data/vlm/{ok-vqa,flickr30k}` (the
registry's `../data` convention). Override with `--root` /
`--data-root`. Mirrors move; every URL is overridable via
`--url KEY=URL`, and `--dry-run` prints the plan without touching the
network.

**Checkpoints.**

- OF-3B: the `open_flamingo` package builds the *architecture* only —
  download the trained `OpenFlamingo-3B-…` checkpoint and pass
  `--vlm-checkpoint PATH/checkpoint.pt` (or `VLM_CHECKPOINT=…` to the
  shell script).
- BLIP-2: the **attack** needs a full EVA-CLIP ViT-G/14 (vision *and*
  text towers) — the official `eva_clip` package names it
  **`EVA01-CLIP-g-14`** (that is the registry default for
  `EVA-CLIP-ViT-G/14`; override with `--eva-clip-name` /
  `--eva-clip-pretrained`). The **swap target** is HF
  `Blip2VisionModel`, available from `Salesforce/blip2-opt-6.7b`
  regardless. If EVA-CLIP ViT-G/14 cannot be loaded, `--swap-mode none`
  still runs the trigger-only transfer row.

**Hardware.** OF-3B ≈ 3 B, LLaVA-7B ≈ 7 B, BLIP-2 OPT-6.7B ≈ 8 B
parameters. These need a real GPU (≥ 24 GB for comfortable inference;
`load_kwargs` in `vlm_registry.py` can set `torch_dtype="float16"`).

---

## 7. Running

### 7.0 The run order (quickstart)

Each step is a separate thing to run; later steps depend on earlier ones.
Steps 1–4 are **once per machine**.

```bash
cd proposed_method/modular_new_pipeline     # all commands below run from here
```

**Step 1 — environment (once).**
```bash
bash download_manager/setup_env.sh   # torch + VLM extras + the EVA-CLIP source
                              # checkout that pip cannot provide
bash download_manager/setup_env.sh --verify
                              # GATE: fails loudly if a dep / module /
                              # dataset is missing.  Do not start step 5
                              # until this is clean.
```

**Step 2 — attack data: ImageNet-1k val (once).**
The attack runs on `ImageNet1k`, whose registry root is `../data/imagenet`
and whose loader is `torchvision.datasets.ImageNet` — it needs **both**
`val/<wnid>/` **and** `meta.bin`.
```bash
cd ../..                       # clip_bfa/
python data/prepare_imagenet.py --splits val     # ~6.3 GB + 2.5 MB devkit
cd proposed_method/modular_new_pipeline
```
`--splits val` is enough (the 138 GB train set is only for the linear
probe). If `data/ImageNet/val` is already populated, symlink
`data/imagenet/val -> ../ImageNet/val` to skip the re-download — but
`meta.bin` is still required and only `prepare_imagenet.py` writes it.

**Step 3 — evaluation data: OK-VQA + Flickr-30K (once).**
```bash
python download_manager/download_datasets.py --check --dataset ok-vqa flickr30k
python download_manager/download_datasets.py --dataset ok-vqa     # + COCO val2014 (~13 GB)
python download_manager/download_datasets.py --dataset flickr30k
```
Lands in `clip_bfa/data/vlm/{ok-vqa,flickr30k}`.

**Step 4 — VLM checkpoints (once).**

Authenticate first (put `HF_TOKEN=hf_xxx` in `download_manager/.env`, or
run the login once — the script never prompts for credentials):
```bash
hf auth login                       # or: export HF_TOKEN=hf_xxx
```
Then fetch everything:
```bash
python download_manager/download_vlm_models.py --check     # auth + what is present
python download_manager/download_vlm_models.py --dry-run   # plan + disk sizes
python download_manager/download_vlm_models.py             # everything
```
It pulls `llava` (`liuhaotian/llava-v1.5-7b`), `blip2`
(`Salesforce/blip2-opt-6.7b`), `of3b` (the trained `checkpoint.pt` **and**
its MPT-1B language model — `open_flamingo` ships the architecture only),
`eva-clip` (the EVA-CLIP source checkout for BLIP-2's attack tower) and
the OpenAI CLIP attack encoders. Budget ~35 GB of cache.

Select subsets with `--models llava blip2`; `--write-pth` also puts
`eva_clip` on `sys.path` (otherwise run `download_manager/setup_env.sh`,
which does the
same thing). OF-3B's checkpoint lands in
`vlm_transfer/checkpoints/` — `export VLM_CHECKPOINT=…/checkpoint.pt` when
evaluating.

> **Proxy gotcha (AGENTS.md):** this env sets `ALL_PROXY=socks://…`, which
> crashes its `httpx` and aborts every hub download. The script and
> `vlm_adapters.py` clear those SOCKS vars automatically and leave
> `http(s)` proxies alone.

**Step 5 — see the plan without executing.**
```bash
DRY_RUN=1 bash vlm_transfer/run_vlm_transfer.sh
```
Expect **3 attack runs** (OF-3B and LLaVA-7B share `ViT-L/14`, so it is
attacked once) and **8 evaluations** (4 VLM configs × 2 datasets).

**Step 6 — smoke test (do this before spending GPU hours).**
```bash
MAX_SAMPLES=50 bash vlm_transfer/run_vlm_transfer.sh
```
Identical chain, 50 samples per dataset. Check the `swap` block in each
`vlm_transfer_*.json` reports `missing_n: 0` and `unexpected_n: 0` — a
partial swap silently invalidates the result.

**Step 7 — the real run.**
```bash
bash vlm_transfer/run_vlm_transfer.sh
```
`RESUME=1` (default) skips anything already done, so re-running after an
interruption is safe.

**Step 8 — read the results** — a summary table prints at the end and
`vlm_transfer_{dataset}.json` lands in each run dir (§8).

Useful variants:

```bash
RUN_ATTACK=0 …                          # evaluations only (attacks done)
RUN_EVAL=0   …                          # attacks only
VLMS="LLaVA-7B" EVAL_DATASETS="flickr30k" …   # one cell of the matrix
SEEDS="42 43 44" …                      # multi-seed
P1_LOSS=cos …                           # the old reference trigger loss
EVA_CLIP_NAME=EVA01-CLIP-g-14 …         # override the BLIP-2 tower id
```

### 7.1 The commands individually

```bash
# everything: attacks (deduplicated by tower) + evaluations + summary
bash vlm_transfer/run_vlm_transfer.sh

# a single VLM / dataset
VLMS="LLaVA-7B" EVAL_DATASETS="flickr30k" bash vlm_transfer/run_vlm_transfer.sh

# smoke test on 50 samples
MAX_SAMPLES=50 bash vlm_transfer/run_vlm_transfer.sh

# evaluation only (attacks already done)
RUN_ATTACK=0 bash vlm_transfer/run_vlm_transfer.sh

# see the plan without executing
DRY_RUN=1 bash vlm_transfer/run_vlm_transfer.sh

# or step by step
python vlm_transfer/run_attack.py --vlm LLaVA-7B \
    --dataset ImageNet1k --target-cls 291 --seed 42 --rep 1
python vlm_transfer/evaluate.py \
    --run-dir grid_results/validation/ViT-L14/ImageNet1k/zs_c291_s42_r1_full_t16 \
    --vlm LLaVA-7B --dataset ok-vqa
```

`RESUME=1` (default) skips stages whose artefacts already exist. The
attack matrix is deduplicated: OF-3B and LLaVA-7B share `ViT-L/14`, so
three attack runs cover all four VLM configurations.

### 7.2 Running CLIP-BFA on EVA-CLIP ViT-G/14 (BLIP-2's tower)

`EVA-CLIP-ViT-G/14` is a **first-class attack model** — it is registered
in `config.json` and backed by `EVACLIPAdapter` in `adapters.py`, so it
goes through the normal entry points like any other model. Nothing has to
be installed or registered at run time.

**Prerequisite** — the attack needs a full EVA-CLIP ViT-G/14 (vision
*and* text towers; the Phase-1 trigger loss and the Phase-4 readout live
in the joint image–text space):

```bash
bash download_manager/setup_env.sh   # clones baaivision/EVA and puts EVA-CLIP/rei
                              # on sys.path — that is what makes
                              # `import eva_clip` work (eva-clip is not on
                              # PyPI; `pip install eva-clip` does NOT work)
pip install open_clip_torch   # the OpenCLIP fallback loader
```

`adapters.load_eva_clip` tries `eva_clip` first, then `open_clip`, and
raises a message naming both if neither is importable. The model id is
**`EVA01-CLIP-g-14`** (the official `eva_clip` name for BLIP-2's tower —
*not* `EVA_CLIP_g_14`); it is the registry default for
`EVA-CLIP-ViT-G/14` and can be overridden per run with
`--eva-clip-name` / `--eva-clip-pretrained` (or `EVA_CLIP_NAME` /
`EVA_CLIP_PRETRAINED` for the shell script), or permanently in
`config.json` (`clip_name` / `hf_name`).

**Option A — the plain single-run CLI** (identical to every other model):

```bash
python experiment.py --model EVA-CLIP-ViT-G/14 \
    --dataset ImageNet1k --target-cls 291 \
    --seed 42 --rep 1 --section validation \
    --trigger-size 16 --n-eval 1000 --n-calib 64 \
    --p1-loss infonce --p1-neg-list lion_v1 \
    --save-attacked-model
```

Run dir:
`grid_results/validation/EVA-CLIP-ViT-G14/ImageNet1k/zs_c291_s42_r1_full_t16`.

**Option B — via the VLM driver** (same thing, but pins `--model` to the
VLM's tower, forces `--save-attacked-model`, and writes
`vlm_transfer_meta.json` so `evaluate.py` needs no arguments beyond
`--run-dir`):

```bash
python vlm_transfer/run_attack.py --vlm BLIP-2 \
    --dataset ImageNet1k --target-cls 291 --seed 42 --rep 1
```

`--model` is derived from `--vlm`; passing a *different* `--model` is a
hard error (the tower swap would not be 1:1) unless you pass
`--force-model`. `--eva-clip-name` / `--eva-clip-pretrained` override the
EVA-CLIP identifiers without editing the registry.

**Option C — inside the grid:**

```bash
MODELS="EVA-CLIP-ViT-G/14" DATASETS="ImageNet1k" \
    RUN_VALIDATION=1 RUN_ABLATION=0 bash run_all.sh
```

The other three configurations are plain OpenAI CLIP and need nothing
special:

```bash
python experiment.py --model ViT-L/14       --dataset ImageNet1k --target-cls 291 --save-attacked-model ...
python experiment.py --model ViT-L/14-336px --dataset ImageNet1k --target-cls 291 --save-attacked-model ...
```

### 7.3 Trigger loss used for these runs

`run_vlm_transfer.sh` pins the **Phase-1 InfoNCE** recipe (CHECKLIST.md §0,
2026-11-22): `--p1-loss infonce --p1-neg-list lion_v1` — pure InfoNCE over
the target text plus attacker-curated confusable *text* negatives (m=0,
TDA off). `lion_v1` is the negative list for the attack target `lion`
(ImageNet1k class 291). Both are env-overridable:

```bash
P1_LOSS=cos bash vlm_transfer/run_vlm_transfer.sh          # old reference loss
P1_NEG_LIST=tabby_v1 bash vlm_transfer/run_vlm_transfer.sh # if the target changes
```

If you change `TARGET_CLS`, `P1_NEG_LIST` must change with it — the lists
are per-target and `build_config` fails fast if InfoNCE has no negatives.

---

## 8. Output

Each attack run directory gains two files:

```
{out_root}/{section}/{model}/{dataset}/{task}_c{cls}_s{seed}_r{rep}_{variant}_t{sz}/
├── bfa_results.json            # the attack record (unchanged)
├── trigger.pt
├── model_state_attacked.pt     # forced on by run_attack.py
├── vlm_transfer_meta.json      # which VLM this run is for + the target text
├── vlm_transfer_{dataset}.json # one per evaluated dataset
└── eval_{dataset}.log
```

`vlm_transfer_{dataset}.json` holds `summary` (per-cell
`target_mention_rate` / `bleu4_vs_target_desc` / `n`, plus
`mention_delta_vs_default_clean` = D − A), `swap` (the full tower-swap
audit: how many tensors mapped, loaded, missing, unexpected, dropped), and
`conditions` (the raw generations, unless `--no-save-texts`).

The swap audit is worth reading before trusting a result: a partial swap
(`missing_n` large, or `unexpected_n` non-zero) means the attacked weights
did **not** fully reach the tower.

---

## 9. Open items

- **BLIP-2 row is gated on the EVA-CLIP ViT-G/14 checkpoint.** The
  adapter and registry entry exist and are tested against mock towers; the
  real checkpoint has not been loaded here (see §6).
- **FLICKR-30K image mirror.** `download_manager/download_datasets.py`'s
  default
  `flickr30k_images` URL is a mirror and may move — `--url` overrides it.
  If the mirror is dead, place `flickr30k-images/*.jpg` under
  `data/vlm/flickr30k/images/` manually.
- **No results on disk yet.** Nothing has been executed on the GPU box;
  `CHECKLIST.md` §3.5's table is still empty.
- **Comparison row against X-Transfer's Table 2** is not produced (their
  non-targeted ASR is out of scope by decision). If a comparison is later
  wanted, compute `(s_clean − s_adv)/s_clean` from CIDEr / VQA accuracy —
  the harness already generates everything needed, but would have to add
  the `pycocoevalcap` dependency for CIDEr.

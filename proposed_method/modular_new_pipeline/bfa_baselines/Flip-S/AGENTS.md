# AGENTS.md

Research artifact code for **Flip-S** (CVPR 2025, "Your Scale Factors are My Weapon…"), used as a
baseline in the `clip_bfa` thesis workspace. Standalone scripts + `utils/`; no packaging, tests,
lint, CI, or requirements file. Keep edits minimal and localized — results should stay comparable
to the paper.

## Running scripts

- Each attack/trigger script `os.chdir`s into its own directory at startup. All relative paths
  (`../data/...`, `./trigger/...`) resolve against the **script's directory**, not your cwd.
- `target-*/` scripts add the repo root to `sys.path` themselves; you can run them from anywhere.
- `untarget-imagenet/deit-untargeted-int8.py` **lacks** that `sys.path` hack. Run it with the repo
  root importable, e.g. from the repo root:
  `PYTHONPATH=. python untarget-imagenet/deit-untargeted-int8.py`
- Order matters for targeted attacks: run `deit-gen_trigger[-int8].py` first to produce the trigger
  pkl, then `deit-targeted[-int8].py` which loads `./trigger/<model>-trigger[-int8].pkl`.
- Hyperparameters are plain classes at the top of each script (`DataLoaderArguments`,
  `AttackArguments`, `TriggerArguments`); there is no CLI. Exception:
  `target-cifar10/deit-gen_trigger.py` accepts the target class as optional `sys.argv[1]`.
- All scripts require CUDA: `device_map='cuda:0'` and `.cuda()` are hardcoded (bitsandbytes
  quantization). There is no CPU path.

## Data & checkpoints

- Edit `DataLoaderArguments.valdir` to point at your datasets (see README). Defaults are
  inconsistent: `../data/Imagenet/valid` and `../data/cifar10` resolve to `<repo>/data/...`, but
  `untarget-imagenet` defaults to `./data/Imagenet/valid` → `<repo>/untarget-imagenet/data/...`.
- ImageNet loader expects an ImageFolder-style `valid/` dir; CIFAR-10 loader expects the
  `cifar-10-batches-py` layout at `valdir` (it can download, but the scripts don't pass
  `download=True`).
- **CIFAR-10 scripts do not load the HF hub model.** They `from_pretrained` a locally saved
  fine-tuned checkpoint at `../data/cifar10/deit-base-distilled-patch16-224-cifar10.pkl`
  (a HF model dir despite the `.pkl` extension). You must provide it; the README doesn't say so.
- Trigger-gen scripts write to `./trigger/` but never create it. `target-cifar10/trigger/` is
  absent from the repo — `mkdir -p target-cifar10/trigger` before generating, or saving fails.
- Only one trigger ships in-repo: `target-imagenet/trigger/deit-base-distilled-patch16-224-trigger-int8.pkl`.

## Naming: quantization variants

- Suffix `-int8` = 8-bit bitsandbytes (`load_in_8bit`). No suffix = 4-bit NF4 (`load_in_4bit`,
  `bnb_4bit_quant_type="nf4"`). Match the trigger pkl suffix to the attack script.

## Code map

- `utils/quant_model.py`: swaps `bnb.nn.Linear8bitLt`/`Linear4bit` for `my_8bit_linear`/
  `my_4bit_linear`, which expose the attack surfaces: `absmax` (fp16 scale factor — the paper's
  "Scale Factor") and `w_int` (integer weights). Attack scripts flip bits in these.
- `utils/load_data.py`: dataset classes + deterministic `random_split` helpers that carve `aux_num`
  samples (seeded) off the validation set as the attack's auxiliary set.
- `utils/metrics.py`: `imagenet_acc`/`imagenet_asr` return `(top1, top5)` but **top-5 is never
  updated and is always 0** — don't report it. (`cifar10_*` return top-1 only.)
- `test_deit.py` is a manual smoke script (needs `timm`), not a pytest test; it hardcodes a sample
  image path outside the repo (`../../00.datasets/...`) that won't exist on other machines.
- The attack ban logic in `deit-targeted*.py` hardcodes DeiT-base layer sizes (`4096`, `4096//2`)
  for grouping bit flips; these assume the base model shape if you change models.

## `target-clip/` (CLIP adaptation)

Two adapters run the unchanged Flip-S method (int8/ImageNet variant) on CLIP ViT-B/16. Stage A/B
bodies are verbatim from `target-imagenet/`; every deviation is marked `# DELTA` in-file
(model binding, logit head = linear probe or zero-shot, token offset `+1`, harness).
- `flip_s_clip_v2.py` — **use this**: victim via OpenAI `clip.load` (same loading as the thesis
  method), bnb int8 with scales from `bnb.functional.int8_vectorwise_quant`, fused qkv wrapped as
  one attackable module per block (48 attackable modules total), attention maps via `QuantMHA`.
  Run in the `vlm_attack` conda env (`clip` + `bitsandbytes` + `bitstring`; bitstring 3.1.9 was
  installed from a local wheel — PyPI access is flaky/needs the socks proxy).
- `flip_s_clip.py` — v1, HF `transformers` CLIP binding (superseded by v2).
Stages: `--stage probe|trigger|attack|all` (probe fits the linear probe once, on frozen features).
Keep Flip-S hyperparameters fixed; probe-fitting knobs are labeled scaffolding in-file.
Note: stock bnb `Linear8bitLt.forward` consumes `weight.CB`; the adapters use a dequant-forward
subclass so the method's `replace_with_myLinear` step keeps working (verified in bnb 0.49.2).

## Environment

Unpinned deps; at minimum: `torch`, `torchvision`, `transformers` (providing
`DeiTImageProcessor` / `DeiTForImageClassificationWithTeacher` / `BitsAndBytesConfig`),
`bitsandbytes`, `bitstring`, `tqdm`, `numpy`, `Pillow` (plus `timm` for `test_deit.py`).
There is no lint/typecheck/test workflow — don't invent one; verification is running the scripts.

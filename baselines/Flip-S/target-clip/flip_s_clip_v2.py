"""Flip-S (CVPR 2025) applied to OpenAI CLIP — v2 (OpenAI `clip` package).

Runs the Flip-S method on the vision tower of CLIP ViT-B/16 loaded via OpenAI's
`clip.load(...)` -- the same victim loading as the thesis's own method -- with
the methodology and every hyperparameter kept FIXED from the published scripts:
  - Trigger generation : verbatim from target-imagenet/deit-gen_trigger-int8.py
  - Bit-flip search    : verbatim from target-imagenet/deit-targeted-int8.py
The method bodies below are copied line-for-line; every deviation is marked
"# DELTA" and is model binding / plumbing only:

  DELTA 1  Victim binding: `clip.load('ViT-B/16')` replaces HF DeiT. Vision
           linears are converted to bitsandbytes int8 (same Flip-S setting:
           row-wise int8, threshold 6.0, no fp16 weights). The method sees the
           model through an alias `model.deit.encoder` ->
           `model.visual.transformer` and `.logits` / `.attentions` exactly as
           DeiTForImageClassificationWithTeacher provides them.
  DELTA 2  Attack surface: OpenAI CLIP keeps q/k/v as one fused Parameter
           (`in_proj_weight`). It is wrapped as one quantized module per block
           (`attn.qkv`) so the scale-factor/weight surfaces of q, k, v are all
           attackable, matching the paper's ViT surface. Per block the
           attackable modules are: attn.qkv (2304), attn.out_proj (768),
           mlp.c_fc (3072), mlp.c_proj (768).
  DELTA 3  Logit head: linear probe on frozen CLIP features (default) or the
           zero-shot text head (`--head zeroshot`). Flip-S never attacks the
           head, so the head choice cannot bias the attack mechanics.  The
           probe readout is L2-normalised -- the shared-head convention of
           modular_new_pipeline/linear_probe.py (Phase-5e convention of
           main_new_pipeline_lp.py: the head is trained on L2-normalised
           features, so every readout must feed it L2-normalised features).
           The head itself should be the SHARED probe trained by
           linear_probe.py and passed via --probe-path; it is identical for
           every attack method in the thesis comparison.  With `--head
           zeroshot` (the PRIMARY comparison mode, agreed 2026-11-09: direct
           attacks on CLIP, native zero-shot head, no additive components)
           the text head uses the SAME class names and the SAME 13-template
           prompt ensemble as the proposed method's
           TargetEmbedder.build_text_embeddings (modular_new_pipeline/target.py)
           -- identical readout for every method's rows.
  DELTA 4  Attention maps for Stage A: OpenAI blocks discard attention weights;
           `QuantMHA` recomputes the identical scaled-dot-product attention and
           exposes per-head maps (B, heads, S, S) -- the exact tensor HF
           returned. Token offset is `+1` (CLIP has 1 special token; the
           DeiT original uses `+2`).
  DELTA 5  Quantized linears compute via explicit dequantization
           (`x @ CB * SCB/127`), identical math to the method's own
           `my_8bit_linear`, instead of the bnb int8 kernel. (The stock bnb
           kernel *consumes* weight.CB on forward, which would break the
           method's replace_with_myLinear step; verified in bnb 0.49.2.)
  DELTA 6  Harness: torchvision preprocess shim for utils/load_data, trigger
           filename sanitization, makedirs, CLI stages.
  DELTA 7  LP evaluation harness: `lp_eval_full` scores the thesis comparison
           metrics (LP-CACC / LP-ASR / clean-as-target, TASKS.md §2) on the
           FULL validation split with the shared probe head; the verbatim
           method metrics (small_val_loader) are kept untouched.  `--stage
           probe` prints a warning: it fits a scaffolding probe of its own
           (full train tree, 20 epochs) and is NOT the shared head -- for the
           comparison pass `--probe-path` pointing at the checkpoint written
           by proposed_method/modular_new_pipeline/linear_probe.py.

Everything else -- the search loop, the trigger/attention loss, BitsAndBytes
quantization semantics (CB/SCB produced by bnb.functional.int8_vectorwise_quant,
i.e. the same scales bnb itself computes), DataLoaderArguments /
TriggerArguments / AttackArguments values, schedule_lr, 4096/2048 row bans,
trigger geometry, loss composition -- is fixed and identical to the originals.

Stages (run in order, in an env with `clip` + `bitsandbytes`; e.g. conda env
`vlm_attack`):
  python flip_s_clip_v2.py --stage probe     # scaffolding probe only (see
                                             #   DELTA 7; NOT the shared head)
  python flip_s_clip_v2.py --stage trigger   # Stage A: generate trigger pkl
  python flip_s_clip_v2.py --stage attack    # Stage B: bit-flip attack search
  python flip_s_clip_v2.py --stage all       # trigger + attack

For the thesis LP comparison, point --probe-path at the shared head written
by proposed_method/modular_new_pipeline/linear_probe.py
(= {lp_head_root}/{model_tag}_{test_dataset}.pt, e.g.
proposed_method/lp_heads/ViT-B16_ImageNet1k.pt).

Primary comparison run (agreed 2026-11-09): direct attack on CLIP with the
native zero-shot head, target class matched to the proposed method's
--target-cls (its default is 3):
  python flip_s_clip_v2.py --head zeroshot --target-class 3 --stage all
"""

import os
import gc

# ---------------------------------------------------------------------------
# Bootstrap: identical to the original scripts (paths resolve against THIS
# directory; utils/ is imported from the repo root).
# ---------------------------------------------------------------------------
if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))
utils_path = os.path.abspath('../')
import sys
sys.path.append(utils_path)

import argparse
import pickle
from types import SimpleNamespace

from bitstring import Bits

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import bitsandbytes as bnb
import clip

from utils.load_data import load_split_ImageNet1k_valid, my_ImageDataset
from utils.metrics import imagenet_acc, imagenet_asr
from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear


# ===========================================================================
# DELTA 1/2/3/4/5 -- Scaffolding: CLIP binding + quantization + logit head.
# NOT part of Flip-S. All pieces verified in the vlm_attack env.
# ===========================================================================

MODEL_NAME = 'ViT-B/16'

# The thesis comparison uses the zero-shot head only.  `lp_eval_full` is the
# full-prefix ZS/LP evaluation (the 391-batch / 150-batch bar); it is NOT the
# attack.  Set False to skip it entirely and keep only the 5k
# imagenet_acc/imagenet_asr checkpoints around the search.
EVAL_FULL_VAL = False


# ---- bnb int8 conversion (DELTA 1/5) --------------------------------------

def quantize_int8(weight_fp16):
    """Same quantization bnb applies internally (Int8Params._quantize), so the
    scale factors are identical to any other bnb-int8-quantized victim."""
    CB, SCB, _ = bnb.functional.int8_vectorwise_quant(weight_fp16.contiguous())
    return CB, SCB


class DequantLinear8bitLt(bnb.nn.Linear8bitLt):
    """Linear8bitLt with explicit-dequant forward (identical math to the
    method's my_8bit_linear). Keeps weight.CB/SCB intact across forwards so
    the method's verbatim find_all_bnbLinear/replace_with_myLinear step works."""

    def forward(self, x):
        w_int = self.weight.CB.to(torch.float16)
        absmax = self.weight.SCB.data.to(torch.float16)
        out = F.linear(x, w_int)
        out = out.mul_(absmax.unsqueeze(0).mul(1.0 / 127.0))
        if self.bias is not None:
            out = out.add_(self.bias)
        return out


def convert_linear(src, threshold=6.0):
    """nn.Linear -> bnb int8 linear exposing CB (int8 weights) and SCB (fp16
    scale factors) -- the two attack surfaces of Flip-S."""
    m = DequantLinear8bitLt(src.in_features, src.out_features,
                            bias=src.bias is not None,
                            has_fp16_weights=False, threshold=threshold)
    w = src.weight.data.to(torch.float16)
    CB, SCB = quantize_int8(w)
    m.weight = bnb.nn.Int8Params(CB, requires_grad=False, has_fp16_weights=False,
                                 CB=CB, SCB=SCB)
    if src.bias is not None:
        m.bias = nn.Parameter(src.bias.data.to(torch.float16), requires_grad=False)
    m.source_cls = src.__class__   # required by the method's replace_with_myLinear
    return m


# ---- fused-qkv attention (DELTA 2/4) --------------------------------------

class QuantMHA(nn.Module):
    """Drop-in replacement for OpenAI CLIP's nn.MultiheadAttention in vision
    blocks. Same math (scaled dot-product, 1/sqrt(head_dim), softmax), with the
    fused in_proj weight exposed as one quantized module (attackable) and
    per-head attention maps available when capturing (Stage A)."""

    def __init__(self, mha):
        super().__init__()
        E = mha.embed_dim
        self.n_head = mha.num_heads
        self.head_dim = E // self.n_head
        fused = nn.Linear(E, 3 * E)
        fused.weight.data = mha.in_proj_weight.data.clone()
        fused.bias.data = mha.in_proj_bias.data.clone()
        self.qkv = convert_linear(fused)          # stacked [q; k; v], as in_proj
        self.out_proj = convert_linear(mha.out_proj)
        self.capture = False
        self.last_attn = None

    def forward(self, q, k=None, v=None, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, **kw):
        S, B, E = q.shape
        qkv = self.qkv(q)
        q_, k_, v_ = qkv.chunk(3, dim=-1)

        def heads(t):
            return t.reshape(S, B, self.n_head, self.head_dim).permute(1, 2, 0, 3)

        qq, kk, vv = heads(q_), heads(k_), heads(v_)
        attn = (qq @ kk.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        out = (attn @ vv).permute(2, 0, 1, 3).reshape(S, B, E)
        out = self.out_proj(out)
        if self.capture:
            self.last_attn = attn                 # (B, heads, S, S), in graph
        return out, (self.last_attn if self.capture else None)


# ---- preprocess shim for utils/load_data (DELTA 6) ------------------------

class ProcessorShim:
    """Adapts clip.load's torchvision pipeline to the HF-processor interface
    utils/load_data expects (`.image_mean`/`.image_std`, `do_normalize=False`
    -> un-normalized pixel_values; the method applies `normalize` itself)."""

    def __init__(self, preprocess):
        tf = list(preprocess.transforms)
        norm = [t for t in tf if isinstance(t, T.Normalize)][0]
        self._tf = T.Compose([t for t in tf if not isinstance(t, T.Normalize)])
        self.image_mean = list(norm.mean)
        self.image_std = list(norm.std)

    def __call__(self, images=None, return_tensors=None, do_normalize=False):
        return {'pixel_values': self._tf(images)}


# ---- victim wrapper (DELTA 1/3) -------------------------------------------

class CLIPVictim(nn.Module):
    """Hands OpenAI CLIP to the unmodified Flip-S method code:
      - `model.deit.encoder` -> the vision transformer (search target)
      - `model(x)` -> .logits (probe or zero-shot head)
      - `model(x, output_attentions=True)` -> .attentions (per-head maps)
    """

    def __init__(self, clip_model, probe=None, text_features=None, logit_scale=None):
        super().__init__()
        self.model = clip_model
        # DELTA 1: alias, no logic. Stage B iterates `model.deit.encoder`.
        self.deit = SimpleNamespace(encoder=clip_model.visual.transformer)
        self._attns = [blk.attn for blk in clip_model.visual.transformer.resblocks
                       if isinstance(blk.attn, QuantMHA)]
        # DELTA 3: logit head. Probe is frozen (Flip-S does not attack the head).
        self.probe = probe
        if probe is not None:
            for p in self.probe.parameters():
                p.requires_grad_(False)
        self.text_features = text_features          # [num_classes, dim], fixed
        self.logit_scale = logit_scale              # scalar tensor, fixed

    def _input_dtype(self):
        return self.model.visual.conv1.weight.dtype

    def extract_features(self, pixel_values):
        """Projected CLIP image embedding, L2-normalised (the input
        convention of the shared probe head -- DELTA 3)."""
        f = self.model.encode_image(pixel_values.to(self._input_dtype()))
        return f / f.norm(dim=-1, keepdim=True)

    def forward(self, x, output_attentions=False):
        x = x.to(self._input_dtype())
        for a in self._attns:
            a.capture = output_attentions
        feats = self.model.encode_image(x)
        attns = tuple(a.last_attn for a in self._attns) if output_attentions else None
        for a in self._attns:
            a.last_attn = None
        if self.probe is not None:                  # linear-probe head
            f = feats.to(self.probe.weight.dtype)
            f = f / f.norm(dim=-1, keepdim=True)   # DELTA 3: shared-head readout
            logits = self.probe(f)
        else:                                       # zero-shot text head
            logits = (self.logit_scale.to(feats.dtype)
                      * F.normalize(feats, dim=-1)
                      @ self.text_features.to(feats.dtype).t())
        return SimpleNamespace(logits=logits, attentions=attns)


# ---- zero-shot text head (DELTA 3) ----------------------------------------

# PROMPT_TEMPLATES + ensemble math VERBATIM from
# proposed_method/modular_new_pipeline/target.py (TargetEmbedder.
# build_text_embeddings) so every method in the thesis comparison is scored
# with the identical text head.
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a cropped photo of a {}.",
    "a low resolution photo of a {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "a photo of many {}.",
    "a picture of a {}.",
    "a rendering of a {}.",
    "a photo of one {}.",
]


def build_text_features(model, classnames, device):
    """Multi-template ensemble class embeddings -- identical convention to
    TargetEmbedder.build_text_embeddings: mean over RAW per-template
    embeddings, then L2-normalise."""
    n_classes = len(classnames)
    n_templates = len(PROMPT_TEMPLATES)
    all_prompts = []
    for c in classnames:
        name = c.replace('_', ' ').lower()
        for template in PROMPT_TEMPLATES:
            all_prompts.append(template.format(name))
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_prompts), 256):
            tok = clip.tokenize(all_prompts[i:i + 256]).to(device)
            all_embeddings.append(model.encode_text(tok))
        all_emb = torch.cat(all_embeddings, dim=0)
    all_emb = all_emb.view(n_classes, n_templates, -1)
    return F.normalize(all_emb.mean(dim=1), dim=-1)


# ---- loading (DELTA 1/3) --------------------------------------------------

def load_clip(model_name, head, probe_path, classnames_path, traindir=None):
    model, preprocess = clip.load(model_name, device='cuda:0')   # fp16 victim

    # Convert the vision tower: qkv (fused), out_proj, mlp c_fc/c_proj.
    for blk in model.visual.transformer.resblocks:
        blk.attn = QuantMHA(blk.attn)
        blk.mlp.c_fc = convert_linear(blk.mlp.c_fc)
        blk.mlp.c_proj = convert_linear(blk.mlp.c_proj)

    probe = None
    text_features = None
    logit_scale = None
    if head == 'probe':
        if probe_path is None or not os.path.exists(probe_path):
            raise SystemExit(f'[-] Probe not found at {probe_path}. '
                             'Pass --probe-path with the shared head written '
                             'by proposed_method/modular_new_pipeline/'
                             'linear_probe.py (or --stage probe for the '
                             'scaffolding probe).')
        state = torch.load(probe_path, map_location='cpu')
        probe = nn.Linear(state['weight'].shape[1], state['weight'].shape[0])
        probe.load_state_dict({'weight': state['weight'], 'bias': state['bias']})
        probe = probe.to('cuda:0').eval()
    elif head == 'zeroshot':
        classnames = _load_classnames(classnames_path, traindir)
        # DELTA 3: shared multi-template ensemble head (same readout as the
        # proposed method's text_emb) instead of a single-template head.
        text_features = build_text_features(model, classnames, 'cuda:0')
        logit_scale = model.logit_scale.detach().clone()

    return CLIPVictim(model, probe=probe, text_features=text_features,
                      logit_scale=logit_scale), ProcessorShim(preprocess)


def _load_classnames(classnames_path, traindir=None):
    if classnames_path is not None:
        with open(classnames_path) as f:
            return [line.strip() for line in f if line.strip()]
    if traindir is not None and os.path.isdir(traindir):
        print('[-] No --classnames given; falling back to dataset folder names.')
        return sorted(os.listdir(traindir))
    raise SystemExit('[-] --head zeroshot needs --classnames (one name per line, '
                     'sorted to match the dataset folder order).')


def trigger_path(model_name, target_class):
    """Trigger pkl location.  DELTA 6: filename sanitized for clip names
    (original: .split('/')[-1]); DELTA 7: target class in the name so
    multi-target comparison runs do not overwrite each other's triggers."""
    load_name = model_name.replace('/', '-')
    return f'./trigger/{load_name}-c{target_class}-trigger-int8.pkl'


def fit_linear_probe(args):
    """One-time SCAFFOLDING linear probe on FROZEN features of the quantized
    CLIP victim.

    Scaffolding (not Flip-S, and NOT the shared thesis head): for the LP
    comparison the probe must be the one trained by
    proposed_method/modular_new_pipeline/linear_probe.py and passed via
    --probe-path (DELTA 7) -- same head for every attack method.  This stage
    fits its own probe (full train tree, 20 epochs) and only exists for
    standalone experiments.
    """
    print('[-] WARNING: --stage probe fits a SCAFFOLDING probe, not the '
          'shared thesis head.\n    For the LP comparison use the checkpoint '
          'written by proposed_method/modular_new_pipeline/linear_probe.py '
          'via --probe-path.', flush=True)
    model, processor = load_clip(args.model, 'none', None,
                                 args.classnames, traindir=args.traindir)
    dataset = my_ImageDataset(args.traindir, processor=processor, use_ir=True)
    num_classes = len(dataset.classes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.probe_batch,
                                         shuffle=False, num_workers=4)
    normalize = torchvision.transforms.Normalize(mean=processor.image_mean,
                                                  std=processor.image_std)
    print(f'[+] Extracting frozen features of {len(dataset)} images '
          f'({num_classes} classes)...', flush=True)

    feats_all, labels_all = [], []
    with torch.no_grad():
        for x, y in tqdm(loader):
            feats = model.extract_features(normalize(x.cuda()))
            feats_all.append(feats.float().cpu())
            labels_all.append(y)
    feats_all = torch.cat(feats_all)
    labels_all = torch.cat(labels_all)
    print(f'[+] Features: {tuple(feats_all.shape)}', flush=True)

    probe = nn.Linear(feats_all.shape[1], num_classes).to('cuda:0')
    opt = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)
    lossfn = nn.CrossEntropyLoss()
    n = feats_all.shape[0]
    for epoch in range(args.probe_epochs):
        perm = torch.randperm(n)
        total, correct, steps = 0., 0, 0
        for i in range(0, n, args.probe_batch):
            idx = perm[i:i + args.probe_batch]
            fx = feats_all[idx].cuda()
            fy = labels_all[idx].cuda()
            logits = probe(fx)
            loss = lossfn(logits, fy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            correct += (logits.argmax(-1) == fy).sum().item()
            steps += 1
        print(f'[+] probe epoch {epoch}: loss {total/steps:.4f} '
              f'acc {correct/n:.4f}', flush=True)

    os.makedirs(os.path.dirname(args.probe_path) or '.', exist_ok=True)
    torch.save({'weight': probe.weight.data.cpu(),
                'bias': probe.bias.data.cpu()}, args.probe_path)
    print(f'[+] Saved linear probe to {args.probe_path}', flush=True)


# ---- full-split LP evaluation (DELTA 7) -----------------------------------

@torch.no_grad()
def lp_eval_full(model, valdir, processor, trigger, mask, normalize,
                 target_class, device, tag='', max_batches=150):
    """Thesis comparison metrics on the validation split (DELTA 7).

    Head-agnostic: scored through whichever logit head CLIPVictim carries --
    the zero-shot text head (primary comparison mode) or the shared probe --
    on the model's own readout convention.  Mirrors phase5e_linear_probe of
    main_new_pipeline_lp.py / TASKS.md §2:
      CACC             argmax accuracy on clean test images (LP-/ZS-CACC);
      ASR              fraction of triggered images classified as the target
                       (LP-/ZS-ASR);
      clean-as-target  fraction of clean images classified as the target.
    Runs with a deterministic prefix of the split (not small_val_loader) so
    every attack method fills the same comparison-table cells.  Trigger
    compositing is identical to utils/metrics.imagenet_asr.

    max_batches=150 caps the evaluation at 150 x 128 = 19,200 images instead
    of the full 50,000 (391 batches).  Use max_batches=0 for the full split.
    """
    prefix = 'ZS' if getattr(model, 'probe', None) is None else 'LP'
    dataset = my_ImageDataset(valdir, processor=processor, use_ir=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=False, num_workers=4)
    max_batches = int(max_batches or 0)
    n_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    hit_c = hit_asr = hit_cat = tot = 0
    for batch_i, (x, y) in enumerate(tqdm(loader, total=n_batches)):
        if max_batches > 0 and batch_i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        pred_c = model(normalize(x)).logits.argmax(-1)
        hit_c += (pred_c == y).sum().item()
        hit_cat += (pred_c == target_class).sum().item()
        pred_t = model(normalize(x.to(torch.float16) * (1 - mask)
                                + torch.mul(trigger, mask))).logits.argmax(-1)
        hit_asr += (pred_t == target_class).sum().item()
        tot += x.size(0)
    n = max(tot, 1)
    res = {'cacc': 100.0 * hit_c / n,
           'asr': 100.0 * hit_asr / n,
           'clean_as_target': 100.0 * hit_cat / n,
           'n': tot}
    print(f'[+] {prefix} eval ({tag or "full val"}, n={tot}): '
          f'{prefix}-CACC {res["cacc"]:.2f}%  {prefix}-ASR {res["asr"]:.2f}%  '
          f'clean-as-target {res["clean_as_target"]:.2f}%', flush=True)
    return res


# ===========================================================================
# Stage A -- Trigger generation
# VERBATIM from target-imagenet/deit-gen_trigger-int8.py (values included).
# Deviations: model binding (DELTA 1), token offset +2 -> +1 (DELTA 4),
# makedirs + trigger filename (DELTA 6).
# ===========================================================================

class GenDataLoaderArguments:   # DataLoaderArguments of deit-gen_trigger-int8.py
    aux_num = 32
    seed = 0
    batch_size = 64
    shuffle = False
    num_workers = 0
    pin_memory = True
    valdir = '../data/Imagenet/valid'
    split_ratio = 0.1

class TriggerArguments:         # TriggerArguments of deit-gen_trigger-int8.py
    num_patch = 9
    train_attack_iters = 300

    target_class = 2
    patch_size = 16
    atten_loss_weight = 1.

def schedule_lr(epoch):         # verbatim
    lr = 100
    if epoch < 300:
        eta_min = .2
    else:
        eta_min = 0.05
    base = epoch // 30
    if epoch < 300:
        learning_rate = lr / pow(3, base)
    else:
        learning_rate = lr / pow(5, base)
    if learning_rate < eta_min:
        learning_rate = eta_min
    return learning_rate


def generate_trigger(args):
    dataargs = GenDataLoaderArguments()
    triggerargs = TriggerArguments()
    triggerargs.target_class = args.target_class   # DELTA 7 (paper default: 2)
    if args.valdir is not None:
        dataargs.valdir = args.valdir

    model_name = args.model
    model, processor = load_clip(model_name, args.head, args.probe_path,
                                 args.classnames, traindir=args.traindir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('[+] Done Load Model')

    ##############################################################################################
    # Split dataset
    crossentropyloss = torch.nn.CrossEntropyLoss()
    val_loader, aux_loader, small_val_loader = load_split_ImageNet1k_valid(dataargs.valdir, aux_num=dataargs.aux_num, seed=dataargs.seed, processor=processor,
                                                         batch_size=dataargs.batch_size, shuffle=dataargs.shuffle, num_workers=dataargs.num_workers,
                                                         pin_memory=dataargs.pin_memory, split_ratio=dataargs.split_ratio, use_ir=True)
    normalize = torchvision.transforms.Normalize(mean=processor.image_mean,std=processor.image_std)

    ##############################################################################################
    # Patch wise select
    # DELTA 4b: infer patch geometry from the victim instead of hardcoding
    # ViT-B/16 (patch 16, 14x14 grid, 48x48 block).  This makes the same file
    # work for ViT-B/16, ViT-L/14 and ViT-L/14-336px.
    try:
        patch_size = int(model.model.visual.conv1.kernel_size[0])
    except Exception:
        patch_size = int(triggerargs.patch_size)
    target_class = triggerargs.target_class
    print(f'[+] target_class: {target_class}',flush=True)

    for x_p, y_p in aux_loader:
        trigger = torch.zeros((x_p.size(-3),x_p.size(-2),x_p.size(-1))).cuda()
        img_size = int(x_p.size(-1))
        patch_num_per_line = img_size // patch_size
        trigger.requires_grad = True
        break
    assert patch_num_per_line >= 3, \
        f'patch grid {patch_num_per_line}x{patch_num_per_line} too small for a 3x3 trigger'
    side = 3 * patch_size
    model.zero_grad()
    #------------------------------------------build mask-------------------------------------------------------------
    mask = torch.zeros([x_p.size(-2), x_p.size(-1)]).cuda()

    # Bottom-right 3x3 patch block, far-corner rows/cols first.  For B/16
    # this reproduces [195,194,193,181,180,179,167,166,165]; for L/14 it
    # gives [255,254,253,239,238,237,223,222,221].
    p = patch_num_per_line
    max_patch_index = torch.tensor([
        (p - 1) * p + (p - 1), (p - 1) * p + (p - 2), (p - 1) * p + (p - 3),
        (p - 2) * p + (p - 1), (p - 2) * p + (p - 2), (p - 2) * p + (p - 3),
        (p - 3) * p + (p - 1), (p - 3) * p + (p - 2), (p - 3) * p + (p - 3),
    ])
    for index in max_patch_index.tolist():
        row = (index // patch_num_per_line) * patch_size
        column = (index % patch_num_per_line) * patch_size

        mask[row:row + patch_size, column:column + patch_size] = 1

    max_patch_index = max_patch_index + 1  # DELTA 4 (original: + 2; CLIP has 1 special token)

    flag = True
    for i in range(img_size - side, img_size):
        for j in range(img_size - side, img_size):
            if mask[i,j] == 0:
                flag = False
                break
    assert flag, 'mask is not correct'
    # --------------------------------------End build mask-------------------------------------------------------------
    mask = mask.cuda()
    mask_size = mask.view(-1).sum()
    max_patch_index = max_patch_index.cuda()

    ##############################################################################################
    # Trigger Generation
    #----------------------------attention loss --------------------------------------------------------------------------------------
    # Start Adv Attack
    trigger.requires_grad = True
    for train_iter_num in tqdm(range(triggerargs.train_attack_iters)):
        total_ce_loss = 0.
        total_atten_loss = 0.
        total_attns_want_means = 0.
        for batch_idx, (x_p, y_p) in enumerate(aux_loader):
            model.zero_grad()

            # Build Sparse Patch attack binary mask
            outputs = model(normalize((x_p.cuda()*(1-mask) + torch.mul(trigger, mask)).clamp(0, 1)), output_attentions=True)
            logits = outputs.logits
            atten = outputs.attentions
            # final CE-loss
            y_p = y_p.cuda()
            y_p[:] = target_class
            loss_p = crossentropyloss(logits, y_p)
            total_ce_loss += loss_p

            # attns = torch.concat([item.unsqueeze(0) for item in atten]).permute(1,0,2,3,4)
            # att_mats = torch.mean(attns, dim=2)
            att_mats = torch.stack([a.mean(dim=1) for a in atten], dim=1)
            # To account for residual connections, we add an identity matrix to the
            # attention matrix and re-normalize the weights.
            residual_att = torch.eye(att_mats.size(-1)).cuda()
            aug_att_mat = att_mats + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size(), requires_grad=True).cuda()
            joint_attentions[0] = aug_att_mat[0]

            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1].clone())

            for i in range(0,joint_attentions.size(1) - 1):
                # Attention from the output token to the input space.
                v = joint_attentions[:,i, :, :].squeeze(1)

                v_mean = torch.mean(torch.mean(v, dim=0),dim=0).view(-1)
                attns_want = v_mean[max_patch_index]
                attns_want_mean = attns_want.mean()
                total_attns_want_means += attns_want_mean

                atten_loss = pow(attns_want_mean-100,2)
                total_atten_loss += atten_loss

            if trigger.grad is not None:
                trigger.grad.data.zero_()
            atten_grad = torch.autograd.grad(atten_loss*triggerargs.atten_loss_weight+loss_p, trigger)[0]

            trigger.data -= atten_grad.sign() * mask * schedule_lr(train_iter_num)
            trigger = torch.clamp(trigger, min=0., max=1.)
        if train_iter_num% 5 == 0:
            print(f'\nEpoch: {train_iter_num}')
            print('[+] learning_rate: %.5f' % schedule_lr(train_iter_num))
            print(f'CE loss:{total_ce_loss/(batch_idx+1)}\nAtten loss:{total_atten_loss/(batch_idx+1)}\nAtten mean:{total_attns_want_means/(batch_idx+1)}',flush=True)

    trig_pkl = trigger_path(model_name, target_class)  # DELTA 6/7
    # saving the trigger image channels for future use
    os.makedirs('./trigger', exist_ok=True)  # DELTA 6 (original crashes if dir missing)
    with open(trig_pkl,'wb') as f:
        pickle.dump(trigger.detach().cpu(), f)
    # -----------------------End Trigger Generation-------------------------------------------------------------
    with open(trig_pkl,'rb') as f:
        trigger = pickle.load(f)
    trigger = trigger.cuda()
    asr1, asr5= imagenet_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)


# ===========================================================================
# Stage B -- Bit-flip attack search
# VERBATIM from target-imagenet/deit-targeted-int8.py (values included).
# Deviations: model binding (DELTA 1/2/3), trigger filename (DELTA 6).
# The search loop below is byte-identical to the original.
# ===========================================================================

class AtkDataLoaderArguments:   # DataLoaderArguments of deit-targeted-int8.py
    aux_num = 32
    seed = 0
    # DELTA 8: 64 OOMs for ViT-L/14 on a 24 GB GPU.  Stage B keeps two L/14
    # graphs alive (clean MSE + triggered CE); 8 is safe, 16 is faster if it
    # fits.  In Stage B the trigger is fixed, so this only changes gradient /
    # candidate-loss accumulation, not the method.
    batch_size = 8
    shuffle = False
    num_workers = 0
    pin_memory = False
    valdir = '../data/Imagenet/valid'
    split_ratio = 0.1

class AttackArguments:          # AttackArguments of deit-targeted-int8.py
    target_class = 2

    topk = 50 # for absmax (or 'Scale Factor')
    topk2 = 50 # for weight
    gamma = 1.
    target_bit = 50

    # DELTA 9: speed knobs.  The paper/release loops target_bit + 10 = 60
    # rounds and verifies every candidate on the full 32-image aux set.
    # max_rounds caps the outer SFS rounds; verify_batches evaluates each
    # candidate on the first N aux batches only (1 batch = 8 images here).
    # Neither changes the SFS score or the ranking; they only shorten the
    # verification.  Set max_rounds=40 for the longer pilot.
    max_rounds = 30
    verify_batches = 1

    # DELTA 10: periodic validation checkpoints.  Every N accepted flips,
    # evaluate ASR/CACC on the validation half of the 5k split.  The final
    # CACC/ASR are reported on the disjoint test half.  Early stop only when
    # BOTH conditions hold; set periodic_asr_target=0 to log without stopping.
    periodic_eval_every = 5
    periodic_asr_target = 95.0
    periodic_min_cacc = 70.0


def bitflip_attack(args):
    dataargs = AtkDataLoaderArguments()
    attackrargs = AttackArguments()
    attackrargs.target_class = args.target_class   # DELTA 7 (paper default: 2)
    if args.valdir is not None:
        dataargs.valdir = args.valdir

    model_name = args.model
    model, processor = load_clip(model_name, args.head, args.probe_path,
                                 args.classnames, traindir=args.traindir)
    # clean_model, _ = load_clip(model_name, args.head, args.probe_path,
    #                            args.classnames, traindir=args.traindir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('[+] Done Load Model')

    ##############################################################################################
    # Split dataset
    crossentropyloss = torch.nn.CrossEntropyLoss()
    mseloss = torch.nn.MSELoss()
    val_loader, aux_loader, small_val_loader = load_split_ImageNet1k_valid(dataargs.valdir, aux_num=dataargs.aux_num, seed=dataargs.seed, processor=processor,
                                                         batch_size=dataargs.batch_size, shuffle=dataargs.shuffle, num_workers=dataargs.num_workers,
                                                         pin_memory=dataargs.pin_memory, split_ratio=dataargs.split_ratio, use_ir=True, use_defalut_valid_batchsize=True)
    normalize = torchvision.transforms.Normalize(mean=processor.image_mean,std=processor.image_std)

    # DELTA 10: split the 5k small-val set into disjoint validation (periodic
    # checkpoints / early stopping) and test (final pre/post metrics) halves.
    # The 32-image aux search set is disjoint from both.
    small_base = small_val_loader.dataset
    n_small = len(small_base)
    perm = torch.randperm(
        n_small, generator=torch.Generator().manual_seed(dataargs.seed))
    n_val_stop = n_small // 2
    val_stop_ds = torch.utils.data.Subset(
        small_base, perm[:n_val_stop].tolist())
    test_ds = torch.utils.data.Subset(
        small_base, perm[n_val_stop:].tolist())
    val_stop_loader = torch.utils.data.DataLoader(
        val_stop_ds, batch_size=128, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=128, shuffle=False, num_workers=4)
    print(f'[+] split: val_stop {len(val_stop_ds)} / test {len(test_ds)} '
          f'(both disjoint from aux {dataargs.aux_num})', flush=True)

    clean_logits_cache = []
    with torch.no_grad():
        for aux_inputs, _ in aux_loader:
            clean_logits_cache.append(
                model(normalize(aux_inputs.to(device))).logits.clone())
    torch.cuda.empty_cache()
    #-----------------------Trojan Insertion----------------------------------------------------------------___
    _, modules_to_convert = find_all_bnbLinear(model)
    model, has_been_replaced = replace_with_myLinear(model, modules_to_convert=modules_to_convert)
    if not has_been_replaced:
        print("[-] Can't find any bnb Linear!")
        exit(0)
    print('[+] Done Replace Model')

    target_class = attackrargs.target_class
    with open(trigger_path(model_name, target_class), 'rb') as f:  # DELTA 6/7
        trigger = pickle.load(f)
    trigger = trigger.cuda()
    # DELTA 4b: derive the block side from the victim patch size (B/16: 48,
    # L/14: 42) instead of hardcoding 48.
    try:
        patch_size = int(model.model.visual.conv1.kernel_size[0])
    except Exception:
        patch_size = 16
    img_size = int(trigger.shape[-1])
    side = 3 * patch_size
    mask = torch.zeros(trigger.shape[-2], trigger.shape[-1], dtype=torch.float16).cuda()
    mask[img_size-side:img_size, img_size-side:img_size] = 1.
    print(f'[+] done load Trigger')

    print('========================Before Attack========================')
    acc1 = imagenet_acc(torch.nn.Sequential(normalize, model), test_loader, device)
    asr1 = imagenet_asr(model, test_loader, trigger, mask, normalize, target_class, device)
    if EVAL_FULL_VAL:
        lp_eval_full(model, dataargs.valdir, processor, trigger, mask, normalize,
                     target_class, device, tag='before attack')  # DELTA 7
    # DELTA 8: release the full-val evaluation's cached activations before
    # Stage B builds its two training graphs (L/14 OOM fix).
    gc.collect()
    torch.cuda.empty_cache()
    print('========================Start  Attack========================')

    topk = attackrargs.topk
    topk2 = attackrargs.topk2
    gamma = attackrargs.gamma
    changed_bit = set()
    base = [2 ** i for i in range(16 - 1, -1, -1)]
    base[0] = -base[0]
    baned_absmax = {}
    baned_weight = {}
    base = torch.tensor(base,dtype=torch.int16).cuda()
    base2 = [2 ** i for i in range(8 - 1, -1, -1)]
    base2[0] = -base2[0]
    base2 = torch.tensor(base2,dtype=torch.float16).cuda()

    # DELTA 9: candidate verification uses only the first N aux batches;
    # the gradient/ranking phase still uses all batches.
    verify_batches = max(1, min(int(attackrargs.verify_batches), len(aux_loader)))

    # DELTA 10: next accepted-flip count at which to run the validation check.
    periodic_every = max(0, int(attackrargs.periodic_eval_every))
    next_periodic = periodic_every if periodic_every > 0 else 10 ** 9

    for ext_iter in tqdm(range(int(attackrargs.max_rounds))):
        model.zero_grad()
        total_loss = 0.
        now_loss = 0.

        for batch_idx, (aux_inputs, aux_targets) in enumerate(aux_loader):
            aux_inputs = aux_inputs.to(device)
            aux_targets = aux_targets.to(device)
            # DELTA 9: one forward for clean + triggered (same math, half
            # the forward calls), one backward for the combined loss.
            clean_imgs = normalize(aux_inputs)
            trig_imgs = normalize(aux_inputs*(1-mask) + torch.mul(trigger, mask))
            outputs = model(torch.cat([clean_imgs, trig_imgs], dim=0))
            logits_c, logits_t = outputs.logits.chunk(2, dim=0)
            clean_logits = clean_logits_cache[batch_idx]
            loss = mseloss(logits_c, clean_logits)

            target_loss = crossentropyloss(logits_t, aux_targets*0+target_class)
            loss += gamma*(target_loss)
            loss = loss / (1 + gamma)

            total_loss += loss.data
            if batch_idx < verify_batches:
                now_loss += loss.data
            loss.backward()
        layers = {}
        print(f'[+] ext_epoch {ext_iter}: loss {total_loss/(batch_idx+1):.4f} '
              f'verify {float(now_loss)/verify_batches:.4f}', flush=True)

        # check absmax
        with torch.no_grad():
            for name, layer in model.deit.encoder.named_modules():
                if isinstance(layer, my_8bit_linear):
                    grad = layer.absmax.grad.data
                    grad = grad.view(-1)

                    grad_abs = grad.abs()
                    if baned_absmax.__contains__(name):
                        for idx in baned_absmax[name]:
                            grad_abs[idx] = -100.

                    now_topk = grad_abs.topk(topk)
                    layers[name] = {'values': grad[now_topk.indices].tolist(), 'indices':now_topk.indices.tolist()}

            all_grad = {}
            for layer in layers:
                for i, idx in enumerate(layers[layer]['indices']):
                    if baned_absmax.__contains__(layer) and idx in baned_absmax[layer]:
                        continue
                    all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

            sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)

            atk_info={}
            for info in sorted_grad[:topk]:
                layer, idx = info[0].split('@')
                if not atk_info.__contains__(layer):
                    atk_info[layer] = []
                atk_info[layer].append((int(idx),info[1]))

            all_loss = {}
            for name, layer in model.deit.encoder.named_modules():
                if isinstance(layer, my_8bit_linear):
                    if atk_info.__contains__(name):
                        ori_absmax = layer.absmax.detach().clone()

                        for idx, grad in atk_info[name]:
                            now_absmax = ori_absmax[idx]
                            bits = torch.tensor([int(b) for b in Bits(int=int(now_absmax.view(torch.int16)),length=16).bin]).cuda()
                            for i in range(16):
                                old_bit = bits[i].clone()
                                if old_bit == 0:
                                    new_bit = 1
                                else:
                                    new_bit = 0
                                bits[i] = new_bit
                                new_absmax = bits * base
                                new_absmax = torch.sum(new_absmax, dim=-1).type(torch.int16).to(now_absmax.device).view(torch.float16)

                                if torch.isnan(new_absmax).any() or torch.isinf(new_absmax).any():
                                    bits[i] = old_bit
                                    continue

                                if (new_absmax-now_absmax)*grad > 0:
                                    bits[i] = old_bit
                                    continue

                                bits[i] = old_bit

                                layer.absmax[idx] = new_absmax.clone()

                                total_loss = 0.
                                for batch_idx, (aux_inputs, aux_targets) in enumerate(aux_loader):
                                    if batch_idx >= verify_batches:
                                        break
                                    aux_inputs = aux_inputs.to(device)
                                    aux_targets = aux_targets.to(device)
                                    # DELTA 9: one fused clean+triggered forward.
                                    clean_imgs = normalize(aux_inputs)
                                    trig_imgs = normalize(aux_inputs*(1-mask) + torch.mul(trigger, mask))
                                    outputs = model(torch.cat([clean_imgs, trig_imgs], dim=0))
                                    logits_c, logits_t = outputs.logits.chunk(2, dim=0)
                                    clean_logits = clean_logits_cache[batch_idx]
                                    loss = mseloss(logits_c, clean_logits)

                                    target_loss = crossentropyloss(logits_t, aux_targets*0+target_class)
                                    loss += gamma*(target_loss)
                                    loss = loss / (1 + gamma)

                                    total_loss += loss.data

                                    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                                        break

                                layer.absmax[idx] = now_absmax.clone()

                                if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                                    continue

                                best_bit = f'{name}@{idx}@{i}@absmax'
                                all_loss[best_bit] = total_loss.data

        # valid absmax
        best_bit = min(all_loss, key=all_loss.get)
        min_loss = all_loss[best_bit]
        skip = False
        if min_loss < now_loss:
            skip = True

        # check weight
        if not skip:
            with torch.no_grad():
                for name, layer in model.deit.encoder.named_modules():
                    if isinstance(layer, my_8bit_linear):
                        grad = layer.w_int.grad.data
                        grad = grad.view(-1)

                        grad_abs = grad.abs()
                        if baned_weight.__contains__(name):
                            for idx in baned_weight[name]:
                                grad_abs[idx] = -100.

                        now_topk = grad_abs.topk(topk2)
                        layers[name] = {'values': grad[now_topk.indices].tolist(), 'indices':now_topk.indices.tolist()}

                all_grad = {}
                for layer in layers:
                    for i, idx in enumerate(layers[layer]['indices']):
                        if baned_weight.__contains__(layer) and idx in baned_weight[layer]:
                            continue
                        all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

                sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)

                atk_info={}
                for info in sorted_grad[:topk2]:
                    layer, idx = info[0].split('@')
                    if not atk_info.__contains__(layer):
                        atk_info[layer] = []
                    atk_info[layer].append((int(idx),info[1]))

                for name, layer in model.deit.encoder.named_modules():
                    if isinstance(layer, my_8bit_linear):
                        if atk_info.__contains__(name):
                            ori_shape = layer.w_int.shape
                            layer.w_int.data = layer.w_int.data.view(-1)
                            ori_weight = layer.w_int.detach().clone()
                            for idx, grad in atk_info[name]:
                                now_weight = ori_weight[idx]
                                bits = torch.tensor([int(b) for b in Bits(int=int(now_weight.type(torch.int8)),length=8).bin]).cuda()
                                for i in range(8):
                                    old_bit = bits[i].clone()
                                    if old_bit == 0:
                                        new_bit = 1
                                    else:
                                        new_bit = 0
                                    bits[i] = new_bit
                                    new_weight = bits * base2
                                    new_weight = torch.sum(new_weight, dim=-1).to(now_weight.device).type(torch.float16)

                                    if (new_weight-now_weight)*grad > 0:
                                        bits[i] = old_bit
                                        continue

                                    bits[i] = old_bit

                                    layer.w_int[idx] = new_weight.clone()
                                    layer.w_int.data = layer.w_int.data.view(ori_shape)

                                    best_bit = f'{name}@{idx}@{i}@weight'

                                    total_loss = 0.
                                    for batch_idx, (aux_inputs, aux_targets) in enumerate(aux_loader):
                                        if batch_idx >= verify_batches:
                                            break
                                        aux_inputs = aux_inputs.to(device)
                                        aux_targets = aux_targets.to(device)
                                        # DELTA 9: one fused clean+triggered forward.
                                        clean_imgs = normalize(aux_inputs)
                                        trig_imgs = normalize(aux_inputs*(1-mask) + torch.mul(trigger, mask))
                                        outputs = model(torch.cat([clean_imgs, trig_imgs], dim=0))
                                        logits_c, logits_t = outputs.logits.chunk(2, dim=0)
                                        clean_logits = clean_logits_cache[batch_idx]
                                        loss = mseloss(logits_c, clean_logits)

                                        target_loss = crossentropyloss(logits_t, aux_targets*0+target_class)
                                        loss += gamma*(target_loss)
                                        loss = loss / (1 + gamma)

                                        total_loss += loss.data

                                    layer.w_int.data = layer.w_int.data.view(-1)
                                    layer.w_int[idx] = now_weight.clone()

                                    best_bit  = f'{name}@{idx}@{i}@weight'
                                    all_loss[best_bit] = total_loss.data

                            layer.w_int.data = layer.w_int.data.view(ori_shape)

        with torch.no_grad():
            # select
            best_bit = min(all_loss, key=all_loss.get)
            print(f'[+] change {best_bit}, loss: '
                  f'{float(all_loss[best_bit]) / verify_batches:.4f}', flush=True)
            # '{name}@{idx}@{i}@absmax'
            layer_name, idx, i, bit_type = best_bit.split('@')
            idx, i = int(idx), int(i)
            for name, layer in model.deit.encoder.named_modules():
                if isinstance(layer, my_8bit_linear) and layer_name == name:
                    if bit_type == 'absmax':
                        now_absmax = layer.absmax[idx]
                        bits = torch.tensor([int(b) for b in Bits(int=int(now_absmax.view(torch.int16)),length=16).bin]).cuda()
                        old_bit = bits[i]
                        if old_bit == 0:
                            new_bit = 1
                        else:
                            new_bit = 0
                        bits[i] = new_bit
                        new_absmax = bits * base
                        new_absmax = torch.sum(new_absmax, dim=-1).type(torch.int16).to(now_absmax.device).view(torch.float16)
                        layer.absmax[idx] = new_absmax.clone()

                        start_ban = (idx // (4096//2)) * (4096//2) # Note that absmax is a float16 parameter
                        end_ban = start_ban + (4096//2)
                        end_ban = min(end_ban, len(layer.absmax))
                        for idx in range(start_ban, end_ban):
                            if name not in baned_absmax:
                                baned_absmax[name] = []
                            baned_absmax[name].append(idx)
                    elif bit_type == 'weight':
                        ori_shape = layer.w_int.shape
                        layer.w_int.data = layer.w_int.data.view(-1)
                        now_weight = layer.w_int[idx]
                        bits = torch.tensor([int(b) for b in Bits(int=int(now_weight.type(torch.int8)),length=8).bin]).cuda()
                        old_bit = bits[i].clone()
                        if old_bit == 0:
                            new_bit = 1
                        else:
                            new_bit = 0
                        bits[i] = new_bit
                        new_weight = bits * base2
                        new_weight = torch.sum(new_weight, dim=-1).to(now_weight.device).type(torch.float16)
                        layer.w_int[idx] = new_weight.clone()

                        start_ban = (idx // (4096)) * 4096 # Note that weight is a int8 parameter
                        end_ban = start_ban + (4096)
                        end_ban = min(end_ban, len(layer.w_int))
                        for idx in range(start_ban, end_ban):
                            if name not in baned_weight:
                                baned_weight[name] = []
                            baned_weight[name].append(idx)
                        layer.w_int.data = layer.w_int.data.view(ori_shape)
                    else:
                        raise NotImplementedError

            try:
                changed_bit.remove(best_bit)
                print(f'[-] Revoke Flip {best_bit}')
                layer_name, idx, i, bit_type = best_bit.split('@')
                idx, i = int(idx), int(i)
                for name, layer in model.deit.encoder.named_modules():
                    if isinstance(layer, my_8bit_linear) and layer_name == name:
                        if bit_type == 'absmax':
                            start_ban = (idx // (4096//2)) * (4096//2) # Note that absmax is a float16 parameter
                            end_ban = start_ban + (4096//2)
                            end_ban = min(end_ban, len(layer.absmax))
                            for idx in range(start_ban, end_ban):
                                baned_absmax[name].remove(idx)
                        elif bit_type == 'weight':
                            start_ban = (idx // (4096)) * 4096 # Note that weight is a int8 parameter
                            end_ban = start_ban + (4096)
                            end_ban = min(end_ban, len(layer.w_int.view(-1)))
                            for idx in range(start_ban, end_ban):
                                baned_weight[name].remove(idx)
                        else:
                            raise NotImplementedError
            except:
                changed_bit.add(best_bit)
                if len(changed_bit) >= attackrargs.target_bit:
                    print('===========================End opt===========================')
                    acc1 = imagenet_acc(torch.nn.Sequential(normalize, model), test_loader, device)
                    asr1 = imagenet_asr(model, test_loader, trigger, mask, normalize, target_class, device)
                    nbit = len(changed_bit)
                    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
                    if EVAL_FULL_VAL:
                        lp_eval_full(model, dataargs.valdir, processor, trigger,
                                     mask, normalize, target_class, device,
                                     tag=f'after {nbit} flips')  # DELTA 7
                    exit(0)

                # DELTA 10: periodic validation check (disjoint from the
                # test half used for the final numbers).
                if len(changed_bit) >= next_periodic:
                    nbit = len(changed_bit)
                    print(f'[+] periodic validation @ {nbit} flips '
                          f'(n={len(val_stop_ds)})', flush=True)
                    p_asr = imagenet_asr(model, val_stop_loader, trigger, mask,
                                         normalize, target_class, device)
                    p_acc = imagenet_acc(
                        torch.nn.Sequential(normalize, model),
                        val_stop_loader, device)
                    print(f'[+] periodic @ {nbit}: ASR {p_asr:.2f}%  '
                          f'CACC {p_acc:.2f}%', flush=True)
                    next_periodic += periodic_every
                    stop = (float(attackrargs.periodic_asr_target) > 0.0
                            and p_asr >= float(attackrargs.periodic_asr_target)
                            and p_acc >= float(attackrargs.periodic_min_cacc))
                    if stop:
                        print(f'[+] early stop @ {nbit} flips: ASR '
                              f'{p_asr:.2f}% >= '
                              f'{float(attackrargs.periodic_asr_target):.2f}% '
                              f'and CACC {p_acc:.2f}% >= '
                              f'{float(attackrargs.periodic_min_cacc):.2f}%',
                              flush=True)
                        break

    ##############################################################################################
    # End opt
    print('===========================End opt===========================')
    acc1 = imagenet_acc(torch.nn.Sequential(normalize, model), test_loader, device)
    asr1 = imagenet_asr(model, test_loader, trigger, mask, normalize, target_class, device)
    nbit = len(changed_bit)
    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
    if EVAL_FULL_VAL:
        lp_eval_full(model, dataargs.valdir, processor, trigger, mask, normalize,
                     target_class, device,
                     tag=f'after {nbit} flips')  # DELTA 7


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Flip-S on OpenAI CLIP (method fixed)')
    p.add_argument('--stage', choices=['probe', 'trigger', 'attack', 'all'],
                   default='all', help='probe = fit scaffolding probe only '
                   '(NOT the shared head, see DELTA 7); '
                   'trigger = Stage A; attack = Stage B; all = trigger+attack')
    p.add_argument('--model', default=MODEL_NAME,
                   help='clip.load model name (e.g. ViT-B/16) or path to a .pt')
    p.add_argument('--head', choices=['probe', 'zeroshot'], default='probe',
                   help='logit source (DELTA 3); default: linear probe')
    p.add_argument('--valdir', default=None,
                   help='ImageNet valid dir (default ../data/Imagenet/valid -- '
                   'resolved through the data/Imagenet symlink created by '
                   'proposed_method/prepare_imagenet.py)')
    p.add_argument('--traindir', default='../data/Imagenet/train',
                   help='ImageNet train dir (--stage probe only; needs the '
                   'FULL train tree, prepare_imagenet.py makes a subset)')
    p.add_argument('--probe-path', default='./probe/clip-vit-b16-linear-probe.pt',
                   help='shared linear-probe checkpoint: for the thesis '
                   'comparison pass the file written by '
                   'proposed_method/modular_new_pipeline/linear_probe.py '
                   '(e.g. proposed_method/lp_heads/ViT-B16_ImageNet1k.pt)')
    p.add_argument('--target-class', type=int, default=2,
                   help='target class index (Flip-S paper default: 2; set it '
                   'equal to the proposed method --target-cls for comparison '
                   'runs -- its default is 3)')
    p.add_argument('--classnames',
                   default='../../../proposed_method/modular_new_pipeline/'
                           'class_texts/imagenet1k/classes.txt',
                   help='text file with class names (zeroshot head); default: '
                   'the shared ImageNet-1k class list of the proposed method')
    p.add_argument('--probe-batch', type=int, default=256)   # scaffolding
    p.add_argument('--probe-epochs', type=int, default=20)   # scaffolding
    p.add_argument('--probe-lr', type=float, default=1e-3)   # scaffolding
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.stage in ('probe',):
        fit_linear_probe(args)
    if args.stage in ('trigger', 'all'):
        generate_trigger(args)
    if args.stage in ('attack', 'all'):
        if args.head == 'probe' and not os.path.exists(args.probe_path):
            raise SystemExit(f'[-] Probe not found at {args.probe_path}. '
                             'Pass --probe-path with the shared head written '
                             'by proposed_method/modular_new_pipeline/'
                             'linear_probe.py (or --stage probe for the '
                             'scaffolding probe).')
        bitflip_attack(args)

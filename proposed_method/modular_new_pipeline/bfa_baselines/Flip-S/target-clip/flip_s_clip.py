"""Flip-S (CVPR 2025) applied to CLIP — one-file adaptation.

This file runs the Flip-S method on the vision tower of CLIP ViT-B/16 with the
methodology and every hyperparameter kept FIXED from the published scripts:
  - Trigger generation : verbatim from target-imagenet/deit-gen_trigger-int8.py
  - Bit-flip search    : verbatim from target-imagenet/deit-targeted-int8.py
The method bodies below are copied line-for-line; every deviation is marked
"# DELTA" and is model binding / plumbing only:

  DELTA 1  Model binding: CLIP replaces DeiT. A thin wrapper (CLIPVictim) hands
           CLIP to the unmodified method code via an alias `model.deit.encoder`
           -> `model.vision_model.encoder` and returns `.logits` / `.attentions`
           exactly like DeiTForImageClassificationWithTeacher does.
  DELTA 2  Logit source (the "classification head"): the method only needs a
           head that produces class logits. Default is a LINEAR PROBE on the
           frozen CLIP features (the experiment setting agreed for the thesis);
           `--head zeroshot` uses CLIP's zero-shot text head instead. Flip-S
           never attacks the head (the search covers encoder linears only), so
           the choice of head cannot bias the attack mechanics.
  DELTA 3  Token offset in the attention loss: `max_patch_index + 2` becomes
           `+ 1` (DeiT-distilled has 2 special tokens, CLIP has 1). Grid math,
           mask, and everything else unchanged.
  DELTA 4  Harness only: `os.makedirs` before saving the trigger (the original
           scripts crash if `trigger/` is missing) and dtype normalization
           (bnb int8 computes in fp16; F.linear does not type-promote).

Everything else -- BitsAndBytesConfig values, DataLoaderArguments /
TriggerArguments / AttackArguments values, schedule_lr, the search loop, the
4096/2048 row-ban constants, trigger geometry, loss composition -- is fixed and
identical to the originals.

Stages (run in order):
  python flip_s_clip.py --stage probe     # fit the linear probe ONCE (frozen enc.)
  python flip_s_clip.py --stage trigger   # Stage A: generate trigger pkl
  python flip_s_clip.py --stage attack    # Stage B: bit-flip attack search
  python flip_s_clip.py --stage all       # trigger + attack

Requirements: GPU (CUDA is hardcoded, as in the originals), torch, torchvision,
transformers (with bitsandbytes support), bitsandbytes, tqdm, Pillow.
The CLIP checkpoint must be reachable: `--model openai/clip-vit-base-patch16`
(default) or a local path to a downloaded copy.
"""

import os

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
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPModel,
)

from utils.load_data import load_split_ImageNet1k_valid, my_ImageDataset
from utils.metrics import imagenet_acc, imagenet_asr
from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear


# ===========================================================================
# DELTA 1/2/4 -- Scaffolding: CLIP binding + logit head. NOT part of Flip-S.
# ===========================================================================

MODEL_NAME = 'openai/clip-vit-base-patch16'


class CLIPVictim(nn.Module):
    """Hands CLIP to the unmodified Flip-S method code.

    Provides exactly the interface the method scripts use:
      - `model.deit.encoder`      -> the vision tower's encoder (search target)
      - `model(x)` -> .logits     -> class logits (probe or zero-shot head)
      - `model(x, output_attentions=True)` -> .attentions
    """

    def __init__(self, clip_model, probe=None, text_features=None, logit_scale=None):
        super().__init__()
        self.model = clip_model
        # DELTA 1: alias, no logic. Stage B iterates `model.deit.encoder`.
        self.deit = SimpleNamespace(encoder=clip_model.vision_model.encoder)
        # DELTA 2: logit head. Probe is frozen (Flip-S does not attack the head).
        self.probe = probe
        if probe is not None:
            for p in self.probe.parameters():
                p.requires_grad_(False)
        self.text_features = text_features          # [num_classes, dim], fixed
        self.logit_scale = logit_scale              # scalar tensor, fixed

    # -- dtype plumbing (DELTA 4) -------------------------------------------
    def _input_dtype(self):
        return self.model.vision_model.embeddings.patch_embedding.weight.dtype

    def extract_features(self, pixel_values):
        """Projected CLIP image embedding (the input to the logit head)."""
        v = self.model.vision_model(pixel_values=pixel_values)
        feats = self.model.visual_projection(v.pooler_output)
        return v, feats

    def forward(self, x, output_attentions=False):
        x = x.to(self._input_dtype())
        v, feats = self.extract_features(x)
        if self.probe is not None:                  # linear-probe head
            logits = self.probe(feats.to(self.probe.weight.dtype))
        else:                                       # zero-shot text head
            logits = (self.logit_scale.to(feats.dtype)
                      * F.normalize(feats, dim=-1)
                      @ self.text_features.to(feats.dtype).t())
        return SimpleNamespace(logits=logits, attentions=v.attentions)


def load_clip(model_name, head, probe_path, classnames_path, traindir=None):
    """Load one CLIP victim. The BitsAndBytesConfig values are VERBATIM from
    target-imagenet/deit-targeted-int8.py (int8 variant)."""
    processor = CLIPImageProcessor.from_pretrained(model_name)

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    try:
        clip = CLIPModel.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map='cuda:0',
            attn_implementation='eager',   # output_attentions must work (Stage A)
        )
    except TypeError:
        clip = CLIPModel.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map='cuda:0',
        )

    probe = None
    text_features = None
    logit_scale = None
    if head == 'probe':
        if probe_path is None or not os.path.exists(probe_path):
            raise SystemExit(f'[-] Probe not found at {probe_path}. '
                             'Run `--stage probe` first (or pass --probe-path).')
        state = torch.load(probe_path, map_location='cpu')
        probe = nn.Linear(state['weight'].shape[1], state['weight'].shape[0])
        probe.load_state_dict({'weight': state['weight'], 'bias': state['bias']})
        probe = probe.to('cuda:0').eval()
    elif head == 'zeroshot':
        classnames = _load_classnames(classnames_path, traindir)
        tok = AutoTokenizer.from_pretrained(model_name)
        prompts = [f"a photo of a {c}" for c in classnames]
        with torch.no_grad():
            text_features = clip.get_text_features(
                **tok(prompts, padding=True, return_tensors='pt').to('cuda:0'))
            text_features = F.normalize(text_features, dim=-1)
            logit_scale = clip.logit_scale.detach().clone()

    return CLIPVictim(clip, probe=probe, text_features=text_features,
                      logit_scale=logit_scale), processor


def _load_classnames(classnames_path, traindir=None):
    if classnames_path is not None:
        with open(classnames_path) as f:
            return [line.strip() for line in f if line.strip()]
    if traindir is not None and os.path.isdir(traindir):
        print('[-] No --classnames given; falling back to dataset folder names.')
        return sorted(os.listdir(traindir))
    raise SystemExit('[-] --head zeroshot needs --classnames (one name per line, '
                     'sorted to match the dataset folder order).')


def fit_linear_probe(args):
    """One-time linear probe on FROZEN features of the quantized CLIP victim.

    Scaffolding (not Flip-S): the probe is the experiment's logit head. It is
    trained once, frozen, and shared by Flip-S and any other attack method so
    the victim model is identical for all comparisons.
    """
    device = torch.device('cuda:0')
    model, processor = load_clip(args.model, 'none', None,
                                 args.classnames, traindir=args.traindir)
    dataset = my_ImageDataset(args.traindir, processor=processor, use_ir=True)
    num_classes = len(dataset.classes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.probe_batch,
                                         shuffle=False, num_workers=4)
    print(f'[+] Extracting frozen features of {len(dataset)} images '
          f'({num_classes} classes)...', flush=True)

    feats_all, labels_all = [], []
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device).to(model._input_dtype())
            _, feats = model.extract_features(x)
            feats_all.append(feats.float().cpu())
            labels_all.append(y)
    feats_all = torch.cat(feats_all)
    labels_all = torch.cat(labels_all)
    print(f'[+] Features: {tuple(feats_all.shape)}', flush=True)

    probe = nn.Linear(feats_all.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)
    lossfn = nn.CrossEntropyLoss()
    n = feats_all.shape[0]
    for epoch in range(args.probe_epochs):
        perm = torch.randperm(n)
        total, correct, steps = 0., 0, 0
        for i in range(0, n, args.probe_batch):
            idx = perm[i:i + args.probe_batch]
            fx = feats_all[idx].to(device)
            fy = labels_all[idx].to(device)
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


# ===========================================================================
# Stage A -- Trigger generation
# VERBATIM from target-imagenet/deit-gen_trigger-int8.py (values included).
# Deviations: model binding (DELTA 1), token offset +2 -> +1 (DELTA 3),
# makedirs before save (DELTA 4).
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
    patch_size = triggerargs.patch_size
    target_class = triggerargs.target_class
    print(f'[+] target_class: {target_class}',flush=True)

    for x_p, y_p in aux_loader:
        trigger = torch.zeros((x_p.size(-3),x_p.size(-2),x_p.size(-1))).cuda()
        patch_num_per_line = int(x_p.size(-1) / patch_size)
        trigger.requires_grad = True
        break
    model.zero_grad()
    #------------------------------------------build mask-------------------------------------------------------------
    mask = torch.zeros([x_p.size(-2), x_p.size(-1)]).cuda()

    max_patch_index = torch.tensor([195,194,193,181,180,179,167,166,165])
    for index in max_patch_index.tolist():
        row = (index // patch_num_per_line) * patch_size
        column = (index % patch_num_per_line) * patch_size

        mask[row:row + patch_size, column:column + patch_size] = 1

    max_patch_index = max_patch_index + 1  # DELTA 3 (original: + 2; CLIP has 1 special token)

    flag = True
    for i in range(224-48,224):
        for j in range(224-48,224):
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

            attns = torch.concat([item.unsqueeze(0) for item in atten]).permute(1,0,2,3,4)
            att_mats = torch.mean(attns, dim=2)

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
            atten_grad = torch.autograd.grad(atten_loss*triggerargs.atten_loss_weight+loss_p, trigger, retain_graph=True)[0]

            trigger.data -= atten_grad.sign() * mask * schedule_lr(train_iter_num)
            trigger = torch.clamp(trigger, min=0., max=1.)
        if train_iter_num% 5 == 0:
            print(f'\nEpoch: {train_iter_num}')
            print('[+] learning_rate: %.5f' % schedule_lr(train_iter_num))
            print(f'CE loss:{total_ce_loss/(batch_idx+1)}\nAtten loss:{total_atten_loss/(batch_idx+1)}\nAtten mean:{total_attns_want_means/(batch_idx+1)}',flush=True)

    load_name = model_name.split('/')[-1]
    # saving the trigger image channels for future use
    os.makedirs('./trigger', exist_ok=True)  # DELTA 4 (original crashes if dir missing)
    with open(f'./trigger/{load_name}-trigger-int8.pkl','wb') as f:
        pickle.dump(trigger.detach().cpu(), f)
    # -----------------------End Trigger Generation-------------------------------------------------------------
    with open(f'./trigger/{load_name}-trigger-int8.pkl','rb') as f:
        trigger = pickle.load(f)
    trigger = trigger.cuda()
    asr1, asr5= imagenet_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)


# ===========================================================================
# Stage B -- Bit-flip attack search
# VERBATIM from target-imagenet/deit-targeted-int8.py (values included).
# Deviation: model binding only (DELTA 1/2). The search loop below is
# byte-identical to the original.
# ===========================================================================

class AtkDataLoaderArguments:   # DataLoaderArguments of deit-targeted-int8.py
    aux_num = 32
    seed = 0
    batch_size = 64
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


def bitflip_attack(args):
    dataargs = AtkDataLoaderArguments()
    attackrargs = AttackArguments()
    if args.valdir is not None:
        dataargs.valdir = args.valdir

    model_name = args.model
    model, processor = load_clip(model_name, args.head, args.probe_path,
                                 args.classnames, traindir=args.traindir)
    clean_model, _ = load_clip(model_name, args.head, args.probe_path,
                               args.classnames, traindir=args.traindir)
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

    #-----------------------Trojan Insertion----------------------------------------------------------------___
    _, modules_to_convert = find_all_bnbLinear(model)
    model, has_been_replaced = replace_with_myLinear(model, modules_to_convert=modules_to_convert)
    if not has_been_replaced:
        print("[-] Can't find any bnb Linear!")
        exit(0)
    print('[+] Done Replace Model')

    target_class = attackrargs.target_class
    load_name = model_name.split('/')[-1]
    with open(f'./trigger/{load_name}-trigger-int8.pkl','rb') as f:
        trigger = pickle.load(f)
    trigger = trigger.cuda()
    mask = torch.zeros(trigger.shape[-2], trigger.shape[-1], dtype=torch.float16).cuda()
    mask[224-48:224,224-48:224] = 1.
    print(f'[+] done load Trigger')

    print('========================Before Attack========================')
    acc1 = imagenet_acc(torch.nn.Sequential(normalize, model), small_val_loader, device)
    asr1 = imagenet_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)
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

    for ext_iter in tqdm(range(attackrargs.target_bit+10)):
        model.zero_grad()
        total_loss = 0.

        for batch_idx, (aux_inputs, aux_targets) in enumerate(aux_loader):
            aux_inputs = aux_inputs.to(device)
            aux_targets = aux_targets.to(device)
            # compute output
            outputs = model(normalize(aux_inputs))
            logits = outputs.logits
            # loss = crossentropyloss(logits, aux_targets)
            clean_logits = clean_model(normalize(aux_inputs)).logits
            loss = mseloss(logits, clean_logits)

            outputs = model(normalize(aux_inputs.cuda()*(1-mask) + torch.mul(trigger, mask)))
            logits = outputs.logits

            target_loss = crossentropyloss(logits, aux_targets*0+target_class)
            loss += gamma*(target_loss)
            loss = loss / (1 + gamma)

            total_loss += loss.data
            loss.backward(retain_graph=True)
        layers = {}
        print(f'[+] ext_epoch {ext_iter}: loss {total_loss/(batch_idx+1)}',flush=True)
        now_loss = total_loss

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
                                    aux_inputs = aux_inputs.to(device)
                                    aux_targets = aux_targets.to(device)
                                    # compute output
                                    outputs = model(normalize(aux_inputs))
                                    logits = outputs.logits
                                    # loss = crossentropyloss(logits, aux_targets)
                                    clean_logits = clean_model(normalize(aux_inputs)).logits
                                    loss = mseloss(logits, clean_logits)

                                    outputs = model(normalize(aux_inputs.cuda()*(1-mask) + torch.mul(trigger, mask)))
                                    logits = outputs.logits

                                    target_loss = crossentropyloss(logits, aux_targets*0+target_class)
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
                                        aux_inputs = aux_inputs.to(device)
                                        aux_targets = aux_targets.to(device)
                                        # compute output
                                        outputs = model(normalize(aux_inputs))
                                        logits = outputs.logits
                                        # loss = crossentropyloss(logits, aux_targets)
                                        clean_logits = clean_model(normalize(aux_inputs)).logits
                                        loss = mseloss(logits, clean_logits)

                                        outputs = model(normalize(aux_inputs.cuda()*(1-mask) + torch.mul(trigger, mask)))
                                        logits = outputs.logits

                                        target_loss = crossentropyloss(logits, aux_targets*0+target_class)
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
            print(f'[+] change {best_bit}, loss: {all_loss[best_bit]}',flush=True)
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
                    acc1 = imagenet_acc(torch.nn.Sequential(normalize, model), small_val_loader, device)
                    asr1 = imagenet_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)
                    nbit = len(changed_bit)
                    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
                    exit(0)

    ##############################################################################################
    # End opt
    print('===========================End opt===========================')
    acc1 = imagenet_acc(torch.nn.Sequential(normalize, model), small_val_loader, device)
    asr1 = imagenet_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)
    nbit = len(changed_bit)
    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Flip-S on CLIP (method fixed)')
    p.add_argument('--stage', choices=['probe', 'trigger', 'attack', 'all'],
                   default='all', help='probe = fit linear probe once; '
                   'trigger = Stage A; attack = Stage B; all = trigger+attack')
    p.add_argument('--model', default=MODEL_NAME,
                   help='HF id or local path of CLIP ViT-B/16')
    p.add_argument('--head', choices=['probe', 'zeroshot'], default='probe',
                   help='logit source (DELTA 2); default: linear probe')
    p.add_argument('--valdir', default=None,
                   help='ImageNet valid dir (default ../data/Imagenet/valid)')
    p.add_argument('--traindir', default='../data/Imagenet/train',
                   help='ImageNet train dir (probe fitting)')
    p.add_argument('--probe-path', default='./probe/clip-vit-base-patch16-linear-probe.pt')
    p.add_argument('--classnames', default=None,
                   help='text file with class names (zeroshot head)')
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
                             'Run `--stage probe` first (or pass --probe-path).')
        bitflip_attack(args)

"""
probe_eval.py — Zero-shot, Linear Probe, and Fine-tuning evaluation.
====================================================================
ProbeEvaluator: three standard CLIP evaluation protocols for BFA.

  zero_shot    → text-embedding classification (top-k)
  linear_probe → frozen features + logistic regression
  finetune     → end-to-end visual encoder + linear head

Each returns CACC (clean accuracy) and ASR (attack success rate on
non-target-class images).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from evaluation import encode_image_batched
from trigger import apply_trigger, get_trigger_mode


# ══════════════════════════════════════════════════════════════════════
#  CLASSIFIER HEADS
# ══════════════════════════════════════════════════════════════════════
class LogisticRegression(nn.Module):
    """Single linear layer on frozen features."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class FinetuneHead(nn.Module):
    """CLIP visual encoder + linear head (end-to-end)."""
    def __init__(self, clip_model, output_dim):
        super().__init__()
        self.visual = clip_model.visual
        self.head = nn.Linear(clip_model.text_projection.shape[1], output_dim)

    def forward(self, x):
        return self.head(self.visual(x))


# ══════════════════════════════════════════════════════════════════════
#  COSINE LR SCHEDULE
# ══════════════════════════════════════════════════════════════════════
def _cosine_lr(optimizer, base_lr, warmup, total):
    def fn(step):
        if step < warmup:
            lr = base_lr * (step + 1) / max(1, warmup)
        else:
            t = (step - warmup) / max(1, total - warmup)
            lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * t))
        for g in optimizer.param_groups:
            g["lr"] = lr
    return fn


# ══════════════════════════════════════════════════════════════════════
#  ProbeEvaluator
# ══════════════════════════════════════════════════════════════════════
class ProbeEvaluator:

    def __init__(self, cfg):
        self.cfg = cfg

    # ══════════════════════════════════════════════════════════════
    #  1 — ZERO-SHOT CLASSIFICATION
    # ══════════════════════════════════════════════════════════════
    @torch.no_grad()
    def zero_shot(self, model, text_emb, loader, target_cls,
                  trigger=None, trigger_type="center_add", opacity=1.0,
                  topk=(1, 5), device="cuda"):
        """
        Top-k zero-shot accuracy using text embeddings.

        Returns e.g. {"zs_top1_cacc": 65.3, "zs_top5_cacc": 87.1,
                       "zs_top1_asr": 82.4, "zs_top5_asr": 95.2}
        """
        print("\n  [Zero-shot evaluation]")
        model.eval()
        mode, _ = get_trigger_mode(trigger_type, opacity)
        maxk = max(topk)

        # ── Clean: top-k accuracy ──
        correct = {k: 0 for k in topk}
        total = 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            emb = F.normalize(encode_image_batched(model, imgs), dim=-1)
            logits = emb @ text_emb.T
            _, pred = logits.topk(maxk, dim=1)       # [B, maxk]
            pred = pred.T                             # [maxk, B]
            match = pred == labels.unsqueeze(0)       # [maxk, B]
            for k in topk:
                correct[k] += match[:k].any(0).sum().item()
            total += imgs.size(0)

        results = {}
        for k in topk:
            results[f"zs_top{k}_cacc"] = correct[k] / max(total, 1) * 100

        # ── Triggered: top-k ASR (non-target-class images only) ──
        if trigger is not None:
            asr_correct = {k: 0 for k in topk}
            asr_total = 0
            for imgs, labels in loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                mask = labels != target_cls
                if not mask.any():
                    continue
                imgs_nt = imgs[mask]
                poisoned = apply_trigger(imgs_nt, trigger,
                                         self.cfg.TRIG_SZ, mode, opacity)
                emb = F.normalize(encode_image_batched(model, poisoned), dim=-1)
                logits = emb @ text_emb.T
                _, pred = logits.topk(maxk, dim=1)
                pred = pred.T
                match = pred == target_cls            # [maxk, B_nt]
                for k in topk:
                    asr_correct[k] += match[:k].any(0).sum().item()
                asr_total += imgs_nt.size(0)

            for k in topk:
                results[f"zs_top{k}_asr"] = asr_correct[k] / max(asr_total, 1) * 100

        for key, val in sorted(results.items()):
            print(f"    {key}: {val:.2f}%")
        return results

    # ══════════════════════════════════════════════════════════════
    #  2 — LINEAR PROBE
    # ══════════════════════════════════════════════════════════════
    def linear_probe(self, model, train_loader, test_loader,
                     n_classes, target_cls,
                     trigger=None, trigger_type="center_add", opacity=1.0,
                     epochs=10, lr=1e-3, device="cuda"):
        """
        Frozen encoder → extract features → LogisticRegression → evaluate.

        Returns e.g. {"lp_cacc": 78.5, "lp_asr": 91.2}
        """
        print("\n  [Linear probe evaluation]")
        model.eval()
        dim = model.text_projection.shape[1]

        # ── 1. Extract frozen features from train set ──
        feats, lbls = [], []
        with torch.no_grad():
            for imgs, labels in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                feats.append(encode_image_batched(model, imgs).cpu())
                lbls.append(labels)
        feats = torch.cat(feats)
        lbls = torch.cat(lbls)
        print(f"    Extracted features: {feats.shape}")

        train_ds = torch.utils.data.TensorDataset(feats, lbls)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256,
                                               shuffle=True)

        # ── 2. Train logistic regression on frozen features ──
        clf = LogisticRegression(dim, n_classes).to(device)
        opt = self._adamw(clf.parameters())
        sched = _cosine_lr(opt, lr, 0, len(train_dl) * epochs)
        criterion = nn.CrossEntropyLoss()

        step = 0
        for ep in range(1, epochs + 1):
            ep_loss = nb = 0
            for x, y in train_dl:
                sched(step); step += 1
                x, y = x.to(device), y.to(device)
                loss = criterion(clf(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); nb += 1
            if ep == 1 or ep % 5 == 0 or ep == epochs:
                print(f"    epoch {ep}/{epochs}  loss={ep_loss / max(nb, 1):.4f}")

        # ── 3. Evaluate on clean + triggered images ──
        clf.eval()

        def classify(imgs):
            return clf(encode_image_batched(model, imgs))

        cacc = self._accuracy(classify, test_loader, device)
        asr = self._asr(classify, test_loader, target_cls,
                        trigger, trigger_type, opacity, device)

        results = {"lp_cacc": cacc, "lp_asr": asr}
        for key, val in results.items():
            print(f"    {key}: {val:.2f}%")
        return results

    # ══════════════════════════════════════════════════════════════
    #  3 — FINE-TUNING
    # ══════════════════════════════════════════════════════════════
    def finetune(self, model, train_loader, test_loader,
                 n_classes, target_cls,
                 trigger=None, trigger_type="center_add", opacity=1.0,
                 epochs=5, lr=1e-5, device="cuda"):
        """
        Fine-tune visual encoder + linear head.
        Restores original model state after evaluation.

        Returns e.g. {"ft_cacc": 82.1, "ft_asr": 76.3}
        """
        print("\n  [Fine-tuning evaluation]")
        saved_state = {k: v.clone() for k, v in model.state_dict().items()}

        ft = FinetuneHead(model, n_classes).to(device)
        ft.train()

        opt = self._adamw(ft.parameters())
        sched = _cosine_lr(opt, lr, 0, len(train_loader) * epochs)
        criterion = nn.CrossEntropyLoss()

        step = 0
        for ep in range(1, epochs + 1):
            ep_loss = nb = 0
            for imgs, labels in train_loader:
                sched(step); step += 1
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                loss = criterion(ft(imgs), labels)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); nb += 1
            print(f"    epoch {ep}/{epochs}  loss={ep_loss / max(nb, 1):.4f}")

        # ── Evaluate ──
        ft.eval()

        def classify(imgs):
            return ft(imgs)

        cacc = self._accuracy(classify, test_loader, device)
        asr = self._asr(classify, test_loader, target_cls,
                        trigger, trigger_type, opacity, device)

        # ── Restore attacked model state ──
        model.load_state_dict(saved_state)
        print("    Model state restored.")

        results = {"ft_cacc": cacc, "ft_asr": asr}
        for key, val in results.items():
            print(f"    {key}: {val:.2f}%")
        return results

    # ══════════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════
    @staticmethod
    def _adamw(params):
        decay, no_decay = [], []
        for p in params:
            (no_decay if p.dim() <= 1 else decay).append(p)
        return optim.AdamW([
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": 0.01},
        ])

    @torch.no_grad()
    def _accuracy(self, classify_fn, loader, device):
        correct = total = 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = classify_fn(imgs).argmax(1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)
        return correct / max(total, 1) * 100

    @torch.no_grad()
    def _asr(self, classify_fn, loader, target_cls,
             trigger, trigger_type, opacity, device):
        """ASR: non-target-class images → triggered → predicted as target."""
        if trigger is None:
            return 0.0
        mode, _ = get_trigger_mode(trigger_type, opacity)
        hit = total = 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = labels != target_cls
            if not mask.any():
                continue
            imgs_nt = imgs[mask]
            poisoned = apply_trigger(imgs_nt, trigger,
                                     self.cfg.TRIG_SZ, mode, opacity)
            pred = classify_fn(poisoned).argmax(1)
            hit += (pred == target_cls).sum().item()
            total += imgs_nt.size(0)
        return hit / max(total, 1) * 100
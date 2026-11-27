# far_clip — Forget-and-Rewire (FaR) defence for OpenAI-CLIP vision towers

CLIP adaptation of the FaR defence (Nazari et al., *"Forget and Rewire:
Enhancing the Resilience of Transformer-based Models against Bit-Flip
Attacks"*, USENIX Security '24).  Reference implementation: `../far_demo/`
(HF `google/vit-base-patch16-224` only); paper:
`../far_demo/paper_usenix24_forget_and_rewire.pdf`.

Part of the defence-evaluation protocol of
`../../proposed_method/modular_new_pipeline/DEFENSES.md` (pre-attack
defended-victim protocol): apply FaR to the clean victim **first**, then
run the bit-flip attack on the defended model.

## What it does (paper §5, Algorithm 1)

Per FaR iteration (gradients recomputed after every op):

1. **Important parameter** — top-|gradient| weight of the zero-shot CE loss
   w.r.t. the weights, over all FaR-targetable vision linears.
2. **Dead inputs** — the `div_fac` lowest-|gradient| input columns of the
   same output row in the same layer.
3. **Forget** — the dead weights are overwritten with the important
   weight's value (weight copies).
4. **Rewire** — at forward time the important input activation is divided
   by `div_fac + 1` and rerouted onto the dead input channels.

Result: `div_fac + 1` identical weight copies, each carrying 1/(div_fac+1)
of the gradient and 1/(div_fac+1) of the bit-flip damage of the concealed
parameter — gradient-based attackers find no single point of failure,
while the model's function is preserved (verified in the smoke tests).

## Scope

Vision tower only (agreed): every linear of every `ResidualAttentionBlock`
— fused q/k/v (`attn.in_proj`), `attn.out_proj`, `mlp.c_fc`, `mlp.c_proj`.
The paper's scope is "the linear layers" incl. Query/Key/Value (§2.1);
the fused q/k/v matrix is handled by `FaRMHA` (nn.MultiheadAttention
subclass), the plain linears by `FaRLinear` (nn.Linear subclass).  Both
keep the vanilla parameters/names → **state_dict keys are byte-identical
to a vanilla CLIP**, so the attack pipeline's Quantizer/adapter weight
references work unchanged.

## Run (defended victim)

```bash
python far_clip.py --model ViT-B/16 --data-root <ImageFolder> \
    --out-dir defended_models/
# → defended_models/far_vit_b16.pt        (checkpoint for --far-ckpt)
# → defended_models/far_vit_b16_far_cfg.json   (inspection copy of the cfg)
```

Then attack the defended victim:

```bash
python experiment.py --model ViT-B/16 ... --far-ckpt defended_models/far_vit_b16.pt
```

Paper-faithful defaults: `--div-fac 5` (paper Fig. 9 best), `--frac-size 4`
(≤ in_features/4 concealed inputs per output row), `--acc-drop-stop 2.0`
(stop at accuracy drop, paper §6), `--max-far 30`, `--n-samples 32`.

## Checkpoint contract

```python
{"state_dict": model.state_dict(),   # keys unchanged vs vanilla CLIP
 "far_cfg":   {target_key: [{"weight_dim": i, "conceal": c,
                             "deads": [j, ...]}, ...]},
 "meta":      {...}}
```

`far_cfg` is **required for the defence**: without the runtime rewiring
bound to the victim's linears, the weight copies alone change the model's
function and give no protection.  `pipeline.load_far_state` (the
`--far-ckpt` hook) calls `far_clip.install_far(model, far_cfg)` before
loading the weights.

## Library API

```python
from far_clip import install_far, load_far_checkpoint, far_defend,
                     save_far_checkpoint, collect_far_cfg
```

- `install_far(model, far_cfg=None)` — swap linears for FaR modules and
  bind the cfg (idempotent; keys unchanged).
- `load_far_checkpoint(model, path)` / `save_far_checkpoint(...)`.
- `far_defend(model, images, labels, text_features, ...)` — the FaR loop.

## Verification (2026-11-27)

- `py_compile` on `far_clip.py` and the patched `pipeline.py`.
- `far_forward` ≡ `far_demo`'s per-row input surgery (numerical).
- Forget+Rewire preserves the row's function; gradients equal across copies.
- `FaRMHA` ≡ vanilla `nn.MultiheadAttention` with empty ops.
- `install_far` leaves state_dict keys byte-identical; checkpoint
  round-trip reproduces identical forwards.
- `far_defend` end-to-end on a stub: ops applied, accuracy preserved.
